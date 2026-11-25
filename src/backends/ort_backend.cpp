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
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

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
    if (out_shape.size() == 4 && out_shape[2] > 0 && out_shape[3] > 0) {
        grid_h_ = static_cast<int>(out_shape[2]);
        grid_w_ = static_cast<int>(out_shape[3]);
    } else {
        grid_h_ = input_h_ / 16;
        grid_w_ = input_w_ / 16;
    }

    // Pre-allocate contiguous NCHW float buffer & bind persistent Ort::Value wrapper
    input_nchw_buffer_.resize(static_cast<std::size_t>(3) * input_h_ * input_w_, 0.0f);
    const std::array<int64_t, 4> in_shape{1, 3, input_h_, input_w_};
    input_tensor_ = Ort::Value::CreateTensor<float>(
        mem_info_,
        input_nchw_buffer_.data(),
        input_nchw_buffer_.size(),
        in_shape.data(),
        in_shape.size());

    // Warm up execution provider (triggers CoreML / XNNPACK / CUDA kernel compilation)
    for (int i = 0; i < 5; ++i) {
        auto warmup_out = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &input_tensor_, 1,
            output_names_.data(), 1);
        (void)warmup_out;
    }
}

core::Detection2D OrtBackend::infer(
    const uint8_t* rgb_hwc,
    int src_width,
    int src_height,
    BackendTimingsUs& out_timings) {

    const LetterboxMeta meta = compute_letterbox_meta(src_width, src_height, input_w_, input_h_);

    const uint64_t t0 = core::now_ns();
    {
        AEGIS_NVTX_RANGE("CPU_Preprocess_Letterbox");
        cpu_preprocess_letterbox_nchw(rgb_hwc, meta, input_nchw_buffer_.data());
    }
    const uint64_t t1 = core::now_ns();

    Ort::Value output_tensor{nullptr};
    {
        AEGIS_NVTX_RANGE("ORT_Session_Run");
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor_,
            1,
            output_names_.data(),
            1);
        output_tensor = std::move(outputs[0]);
    }
    const uint64_t t2 = core::now_ns();

    out_timings.preprocess_us = static_cast<double>(t1 - t0) * 1e-3;
    out_timings.inference_us  = static_cast<double>(t2 - t1) * 1e-3;

    const float* out_ptr = output_tensor.GetTensorData<float>();
    return decode_anchor_free_grid(out_ptr, grid_h_, grid_w_, meta);
}

}  // namespace aegis::backends
