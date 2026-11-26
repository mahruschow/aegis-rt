#include "aegis/backends/ort_backend.hpp"
#include "aegis/core/telemetry.hpp"

#include <stdexcept>
#include <unordered_map>

namespace aegis::backends {

OrtBackend::OrtBackend(const std::string& onnx_path,
                       ExecutionProvider provider,
                       int input_width,
                       int input_height,
                       int intra_op_threads)
    : input_w_(input_width),
      input_h_(input_height),
      env_(ORT_LOGGING_LEVEL_WARNING, "aegis_ort"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

    session_opts_.SetIntraOpNumThreads(intra_op_threads);
    session_opts_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_opts_.EnableMemPattern();
    session_opts_.EnableCpuMemArena();

    backend_name_ = "ORT-CPU-FP32";
    if (provider == ExecutionProvider::kCoreML) {
#if defined(__APPLE__)
        try {
            std::unordered_map<std::string, std::string> coreml_opts;
            coreml_opts["MLComputeUnits"] = "ALL";
            session_opts_.AppendExecutionProvider("CoreML", coreml_opts);
            backend_name_ = "ORT-CoreML-ANE";
        } catch (const Ort::Exception&) {
            backend_name_ = "ORT-CPU-FP32 (CoreML fallback)";
        }
#endif
    } else if (provider == ExecutionProvider::kCuda) {
#if defined(AEGIS_ENABLE_CUDA)
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            session_opts_.AppendExecutionProvider_CUDA(cuda_opts);
            backend_name_ = "ORT-CUDA-FP32";
        } catch (const Ort::Exception&) {
            backend_name_ = "ORT-CPU-FP32 (CUDA fallback)";
        }
#else
        backend_name_ = "ORT-CPU-FP32 (CUDA disabled at compile time)";
#endif
    }

    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_opts_);

    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name_alloc = session_->GetInputNameAllocated(0, allocator);
    auto out_name_alloc = session_->GetOutputNameAllocated(0, allocator);
    input_name_ = in_name_alloc.get();
    output_name_ = out_name_alloc.get();
    input_names_[0] = input_name_.c_str();
    output_names_[0] = output_name_.c_str();

    // Query output grid shape [1, 5, GridH, GridW]
    auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int out_channels = 5;
    if (out_shape.size() == 4 && out_shape[2] > 0 && out_shape[3] > 0) {
        if (out_shape[1] > 0) {
            out_channels = static_cast<int>(out_shape[1]);
        }
        grid_h_ = static_cast<int>(out_shape[2]);
        grid_w_ = static_cast<int>(out_shape[3]);
    } else {
        grid_h_ = input_h_ / 16;
        grid_w_ = input_w_ / 16;
    }

    // Precompute separable letterbox/bilinear LUT for standard camera geometry (zero alloc in infer())
    preprocess_plan_.configure(core::kCameraWidth, core::kCameraHeight, input_w_, input_h_);

    // Pre-allocate contiguous NCHW input & output float buffers and bind persistent Ort::Value wrappers
    input_nchw_buffer_.resize(static_cast<std::size_t>(3) * input_h_ * input_w_, 0.0f);
    const std::array<int64_t, 4> in_shape{1, 3, input_h_, input_w_};
    input_tensor_ = Ort::Value::CreateTensor<float>(
        mem_info_,
        input_nchw_buffer_.data(),
        input_nchw_buffer_.size(),
        in_shape.data(),
        in_shape.size());

    output_nchw_buffer_.resize(static_cast<std::size_t>(out_channels) * grid_h_ * grid_w_, 0.0f);
    const std::array<int64_t, 4> out_tensor_shape{1, out_channels, grid_h_, grid_w_};
    output_tensor_ = Ort::Value::CreateTensor<float>(
        mem_info_,
        output_nchw_buffer_.data(),
        output_nchw_buffer_.size(),
        out_tensor_shape.data(),
        out_tensor_shape.size());

    // Warm up execution provider (triggers CoreML / XNNPACK / CUDA kernel compilation)
    for (int i = 0; i < 5; ++i) {
        session_->Run(
            run_options_,
            input_names_.data(), &input_tensor_, 1,
            output_names_.data(), &output_tensor_, 1);
    }
}

core::Detection2D OrtBackend::infer(
    const uint8_t* rgb_hwc,
    int src_width,
    int src_height,
    BackendTimingsUs& out_timings) {

    if (preprocess_plan_.meta().src_w != src_width ||
        preprocess_plan_.meta().src_h != src_height) {
        preprocess_plan_.configure(src_width, src_height, input_w_, input_h_);
    }
    const LetterboxMeta& meta = preprocess_plan_.meta();

    const uint64_t t0 = core::now_ns();
    {
        AEGIS_NVTX_RANGE("CPU_Preprocess_Letterbox");
        preprocess_plan_.execute(rgb_hwc, input_nchw_buffer_.data());
    }
    const uint64_t t1 = core::now_ns();

    {
        AEGIS_NVTX_RANGE("ORT_Session_Run");
        session_->Run(
            run_options_,
            input_names_.data(),
            &input_tensor_,
            1,
            output_names_.data(),
            &output_tensor_,
            1);
    }
    const uint64_t t2 = core::now_ns();

    out_timings.preprocess_us = static_cast<double>(t1 - t0) * 1e-3;
    out_timings.inference_us  = static_cast<double>(t2 - t1) * 1e-3;

    return decode_anchor_free_grid(output_nchw_buffer_.data(), grid_h_, grid_w_, meta);
}

}  // namespace aegis::backends
