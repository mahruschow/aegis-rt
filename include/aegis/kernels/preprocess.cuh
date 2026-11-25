#pragma once

#if defined(AEGIS_ENABLE_CUDA)

#include "aegis/backends/i_backend.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace aegis::kernels {

/**
 * @brief Hardware-accelerated CUDA preprocessing buffer manager & kernel launcher.
 *
 * Key optimizations over CPU OpenCV (`cv::dnn::blobFromImage`):
 *  1. 4x PCIe bandwidth reduction: transfers compact uint8 HWC frame instead of float32 NCHW.
 *  2. Pinned DMA host memory (`cudaMallocHost`) on discrete x86_64 GPUs, OR zero-copy
 *     mapped LPDDR5 memory (`cudaHostAllocMapped`) on NVIDIA Jetson Orin Unified Memory.
 *  3. Single fused 2D CUDA kernel performs aspect-preserving letterbox, sub-pixel bilinear
 *     interpolation, [0, 1] scaling, and coalesced planar CHW writes directly into
 *     TensorRT's bound input tensor address.
 */
class CudaPreprocessor {
public:
    CudaPreprocessor(int max_src_w, int max_src_h, int dst_w, int dst_h, bool zero_copy_unified = false);
    ~CudaPreprocessor();

    CudaPreprocessor(const CudaPreprocessor&) = delete;
    CudaPreprocessor& operator=(const CudaPreprocessor&) = delete;

    // Access pinned/mapped host staging buffer (for direct zero-copy sensor writes)
    [[nodiscard]] uint8_t* host_staging_ptr() noexcept { return h_src_rgb_; }

    // Async stage + launch fused kernel writing FP32 planar NCHW into `d_dst_nchw` on `stream`
    void launch_fp32(const uint8_t* src_rgb_hwc,
                     const backends::LetterboxMeta& meta,
                     float* d_dst_nchw,
                     cudaStream_t stream);

    // Async stage + launch fused kernel writing FP16 planar NCHW into `d_dst_nchw` on `stream`
    void launch_fp16(const uint8_t* src_rgb_hwc,
                     const backends::LetterboxMeta& meta,
                     __half* d_dst_nchw,
                     cudaStream_t stream);

private:
    std::size_t max_src_bytes_{0};
    int dst_w_{0};
    int dst_h_{0};
    bool zero_copy_{false};

    uint8_t* h_src_rgb_{nullptr};
    uint8_t* d_src_rgb_{nullptr};
};

}  // namespace aegis::kernels

#endif  // AEGIS_ENABLE_CUDA
