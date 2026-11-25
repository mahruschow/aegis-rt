#if defined(AEGIS_ENABLE_TENSORRT)

#include "aegis/backends/trt_backend.hpp"
#include "aegis/core/telemetry.hpp"

#include <NvOnnxParser.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace aegis::backends {

namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << "\n";
        }
    }
} g_trt_logger;

/**
 * @brief INT8 Post-Training Quantization Entropy Calibrator (`IInt8EntropyCalibrator2`).
 * Reads raw 640x480 RGB calibration frames captured from MuJoCo and runs our fused CUDA
 * letterbox kernel so calibration tensor statistics match deployment bit-for-bit.
 */
class AegisInt8EntropyCalibrator final : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    AegisInt8EntropyCalibrator(const std::string& calib_dir,
                               const std::string& cache_file,
                               int input_w,
                               int input_h)
        : cache_file_(cache_file),
          input_w_(input_w),
          input_h_(input_h),
          preproc_(core::kCameraWidth, core::kCameraHeight, input_w, input_h, false) {

        if (std::filesystem::exists(calib_dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(calib_dir)) {
                if (entry.path().extension() == ".rgb") {
                    frame_files_.push_back(entry.path().string());
                }
            }
            std::sort(frame_files_.begin(), frame_files_.end());
        }
        cudaMalloc(reinterpret_cast<void**>(&d_calib_input_),
                   static_cast<std::size_t>(3) * input_w_ * input_h_ * sizeof(float));
        cudaStreamCreate(&stream_);
    }

    ~AegisInt8EntropyCalibrator() override {
        if (d_calib_input_) cudaFree(d_calib_input_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    int getBatchSize() const noexcept override { return 1; }

    bool getBatch(void* bindings[], const char* /*names*/[], int /*nbBindings*/) noexcept override {
        if (cursor_ >= frame_files_.size()) {
            return false;
        }
        std::ifstream in(frame_files_[cursor_++], std::ios::binary);
        if (!in.is_open()) {
            return false;
        }
        std::vector<uint8_t> raw_rgb(core::kRgbFrameBytes);
        in.read(reinterpret_cast<char*>(raw_rgb.data()), static_cast<std::streamsize>(raw_rgb.size()));

        const auto meta = compute_letterbox_meta(core::kCameraWidth, core::kCameraHeight, input_w_, input_h_);
        preproc_.launch_fp32(raw_rgb.data(), meta, d_calib_input_, stream_);
        cudaStreamSynchronize(stream_);
        bindings[0] = d_calib_input_;
        return true;
    }

    const void* readCalibrationCache(std::size_t& length) noexcept override {
        calib_cache_.clear();
        std::ifstream in(cache_file_, std::ios::binary);
        if (in.good()) {
            calib_cache_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        length = calib_cache_.size();
        return length ? calib_cache_.data() : nullptr;
    }

    void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override {
        std::ofstream out(cache_file_, std::ios::binary);
        out.write(static_cast<const char*>(ptr), static_cast<std::streamsize>(length));
    }

private:
    std::string cache_file_;
    int input_w_;
    int input_h_;
    kernels::CudaPreprocessor preproc_;
    std::vector<std::string> frame_files_;
    std::size_t cursor_{0};
    float* d_calib_input_{nullptr};
    cudaStream_t stream_{nullptr};
    std::vector<char> calib_cache_;
};

}  // namespace

TrtBackend::TrtBackend(const std::string& onnx_path,
                       const std::string& engine_cache_path,
                       Precision precision,
                       const std::string& calib_dir,
                       bool zero_copy_unified,
                       int input_width,
                       int input_height)
    : input_w_(input_width), input_h_(input_height) {

    switch (precision) {
        case Precision::kFp32: backend_name_ = "TRT-FP32-FusedCUDA"; break;
        case Precision::kFp16: backend_name_ = "TRT-FP16-FusedCUDA"; break;
        case Precision::kInt8: backend_name_ = "TRT-INT8-FusedCUDA"; break;
    }
    if (zero_copy_unified) {
        backend_name_ += "-ZeroCopy";
    }

    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_pre_done_);
    cudaEventCreate(&ev_inf_done_);

    cuda_preproc_ = std::make_unique<kernels::CudaPreprocessor>(
        core::kCameraWidth, core::kCameraHeight, input_w_, input_h_, zero_copy_unified);

    build_or_load_engine(onnx_path, engine_cache_path, precision, calib_dir);
    context_.reset(engine_->createExecutionContext());

    // Resolve input/output tensor names & output grid shape
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* tname = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(tname) == nvinfer1::TensorIOMode::kINPUT) {
            input_tensor_name_ = tname;
        } else {
            output_tensor_name_ = tname;
            const auto dims = engine_->getTensorShape(tname);
            if (dims.nbDims == 4) {
                grid_h_ = static_cast<int>(dims.d[2]);
                grid_w_ = static_cast<int>(dims.d[3]);
            }
        }
    }

    output_floats_ = static_cast<std::size_t>(5) * grid_h_ * grid_w_;
    cudaMalloc(reinterpret_cast<void**>(&d_input_),
               static_cast<std::size_t>(3) * input_h_ * input_w_ * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output_), output_floats_ * sizeof(float));
    cudaMallocHost(reinterpret_cast<void**>(&h_output_pinned_), output_floats_ * sizeof(float));

    context_->setTensorAddress(input_tensor_name_.c_str(), d_input_);
    context_->setTensorAddress(output_tensor_name_.c_str(), d_output_);

    // Warmup GPU clocks & TensorRT execution context
    for (int i = 0; i < 10; ++i) {
        context_->enqueueV3(stream_);
    }
    cudaStreamSynchronize(stream_);
}

TrtBackend::~TrtBackend() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (h_output_pinned_) cudaFreeHost(h_output_pinned_);
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_pre_done_) cudaEventDestroy(ev_pre_done_);
    if (ev_inf_done_) cudaEventDestroy(ev_inf_done_);
    if (stream_) cudaStreamDestroy(stream_);
}

void TrtBackend::build_or_load_engine(const std::string& onnx_path,
                                      const std::string& engine_cache_path,
                                      Precision precision,
                                      const std::string& calib_dir) {
    runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));

    if (!engine_cache_path.empty() && std::filesystem::exists(engine_cache_path)) {
        std::ifstream in(engine_cache_path, std::ios::binary);
        std::vector<char> plan((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
        if (engine_) {
            return;
        }
    }

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_trt_logger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_trt_logger));

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX model for TensorRT: " + onnx_path);
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 29);  // 512 MB

    std::unique_ptr<AegisInt8EntropyCalibrator> calibrator;
    if (precision == Precision::kFp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (precision == Precision::kInt8 && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        calibrator = std::make_unique<AegisInt8EntropyCalibrator>(
            calib_dir, engine_cache_path + ".calib", input_w_, input_h_);
        config->setInt8Calibrator(calibrator.get());
    }

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        throw std::runtime_error("TensorRT buildSerializedNetwork failed.");
    }

    if (!engine_cache_path.empty()) {
        std::filesystem::create_directories(std::filesystem::path(engine_cache_path).parent_path());
        std::ofstream out(engine_cache_path, std::ios::binary);
        out.write(static_cast<const char*>(serialized->data()),
                  static_cast<std::streamsize>(serialized->size()));
    }

    engine_.reset(runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
}

core::Detection2D TrtBackend::infer(
    const uint8_t* rgb_hwc,
    int src_width,
    int src_height,
    BackendTimingsUs& out_timings) {

    const LetterboxMeta meta = compute_letterbox_meta(src_width, src_height, input_w_, input_h_);

    cudaEventRecord(ev_start_, stream_);
    cuda_preproc_->launch_fp32(rgb_hwc, meta, d_input_, stream_);
    cudaEventRecord(ev_pre_done_, stream_);

    {
        AEGIS_NVTX_RANGE("TRT_EnqueueV3");
        context_->enqueueV3(stream_);
        cudaMemcpyAsync(h_output_pinned_, d_output_, output_floats_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
    }
    cudaEventRecord(ev_inf_done_, stream_);
    cudaStreamSynchronize(stream_);

    float pre_ms = 0.0f, inf_ms = 0.0f;
    cudaEventElapsedTime(&pre_ms, ev_start_, ev_pre_done_);
    cudaEventElapsedTime(&inf_ms, ev_pre_done_, ev_inf_done_);
    out_timings.preprocess_us = static_cast<double>(pre_ms) * 1000.0;
    out_timings.inference_us  = static_cast<double>(inf_ms) * 1000.0;

    return decode_anchor_free_grid(h_output_pinned_, grid_h_, grid_w_, meta);
}

}  // namespace aegis::backends

#endif  // AEGIS_ENABLE_TENSORRT
