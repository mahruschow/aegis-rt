#if defined(AEGIS_ENABLE_CUDA)

#include "aegis/kernels/preprocess.cuh"
#include "aegis/core/telemetry.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#define AEGIS_CUDA_CHECK(expr)                                                             \
    do {                                                                                   \
        cudaError_t _err = (expr);                                                         \
        if (_err != cudaSuccess) {                                                         \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +      \
                                     std::to_string(__LINE__) + " -> " +                   \
                                     cudaGetErrorString(_err));                            \
        }                                                                                  \
    } while (0)

namespace aegis::kernels {

namespace {

template <typename OutT>
__device__ __forceinline__ OutT cast_output(float v);

template <>
__device__ __forceinline__ float cast_output<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ __half cast_output<__half>(float v) {
    return __float2half_rn(v);
}

/**
 * @brief Fused 2D CUDA kernel: Letterbox + Subpixel Bilinear Resize + Normalize + HWC->NCHW.
 *
 * Memory access characteristics:
 *  - Threads within a warp (consecutive `dx`) write to contiguous global addresses
 *    `ch_r[out_idx]`, `ch_g[out_idx]`, `ch_b[out_idx]`, achieving 100% coalesced global stores.
 *  - Read-only `__ldg()` texture/L1 cache path accelerates 2x2 neighborhood fetches.
 */
template <typename OutT>
__global__ void fused_letterbox_bilinear_nchw_kernel(
    const uint8_t* __restrict__ src_rgb,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int resized_w,
    int resized_h,
    int pad_x,
    int pad_y,
    float inv_scale,
    OutT* __restrict__ dst_nchw) {

    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) {
        return;
    }

    const int plane_area = dst_w * dst_h;
    const int out_idx = dy * dst_w + dx;

    OutT* ch_r = dst_nchw;
    OutT* ch_g = dst_nchw + plane_area;
    OutT* ch_b = dst_nchw + 2 * plane_area;

    const int inner_x = dx - pad_x;
    const int inner_y = dy - pad_y;

    if (inner_x < 0 || inner_x >= resized_w || inner_y < 0 || inner_y >= resized_h) {
        constexpr float kPadNorm = 114.0f * (1.0f / 255.0f);
        const OutT pad_val = cast_output<OutT>(kPadNorm);
        ch_r[out_idx] = pad_val;
        ch_g[out_idx] = pad_val;
        ch_b[out_idx] = pad_val;
        return;
    }

    // Exact pixel-center alignment matching cpu_preprocess_letterbox_nchw()
    float src_xf = ((static_cast<float>(inner_x) + 0.5f) * inv_scale) - 0.5f;
    float src_yf = ((static_cast<float>(inner_y) + 0.5f) * inv_scale) - 0.5f;
    src_xf = fminf(fmaxf(src_xf, 0.0f), static_cast<float>(src_w - 1));
    src_yf = fminf(fmaxf(src_yf, 0.0f), static_cast<float>(src_h - 1));

    const int x0 = static_cast<int>(floorf(src_xf));
    const int y0 = static_cast<int>(floorf(src_yf));
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);

    const float wx1 = src_xf - static_cast<float>(x0);
    const float wy1 = src_yf - static_cast<float>(y0);
    const float wx0 = 1.0f - wx1;
    const float wy0 = 1.0f - wy1;

    const float w00 = wy0 * wx0;
    const float w01 = wy0 * wx1;
    const float w10 = wy1 * wx0;
    const float w11 = wy1 * wx1;

    const int idx00 = (y0 * src_w + x0) * 3;
    const int idx01 = (y0 * src_w + x1) * 3;
    const int idx10 = (y1 * src_w + x0) * 3;
    const int idx11 = (y1 * src_w + x1) * 3;

    constexpr float kInv255 = 1.0f / 255.0f;

    const float r = (w00 * __ldg(&src_rgb[idx00 + 0]) + w01 * __ldg(&src_rgb[idx01 + 0]) +
                     w10 * __ldg(&src_rgb[idx10 + 0]) + w11 * __ldg(&src_rgb[idx11 + 0])) * kInv255;
    const float g = (w00 * __ldg(&src_rgb[idx00 + 1]) + w01 * __ldg(&src_rgb[idx01 + 1]) +
                     w10 * __ldg(&src_rgb[idx10 + 1]) + w11 * __ldg(&src_rgb[idx11 + 1])) * kInv255;
    const float b = (w00 * __ldg(&src_rgb[idx00 + 2]) + w01 * __ldg(&src_rgb[idx01 + 2]) +
                     w10 * __ldg(&src_rgb[idx10 + 2]) + w11 * __ldg(&src_rgb[idx11 + 2])) * kInv255;

    ch_r[out_idx] = cast_output<OutT>(r);
    ch_g[out_idx] = cast_output<OutT>(g);
    ch_b[out_idx] = cast_output<OutT>(b);
}

}  // namespace

CudaPreprocessor::CudaPreprocessor(int max_src_w, int max_src_h, int dst_w, int dst_h, bool zero_copy_unified)
    : max_src_bytes_(static_cast<std::size_t>(max_src_w) * max_src_h * 3),
      dst_w_(dst_w),
      dst_h_(dst_h),
      zero_copy_(zero_copy_unified) {

    if (zero_copy_) {
        // Jetson Orin Nano Unified LPDDR5 Memory: CPU & GPU share physical memory with 0-byte copy
        AEGIS_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_src_rgb_),
                                       max_src_bytes_,
                                       cudaHostAllocMapped | cudaHostAllocWriteCombined));
        AEGIS_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_src_rgb_), h_src_rgb_, 0));
    } else {
        // Discrete PCIe GPU: Pinned DMA host buffer + dedicated device buffer
        AEGIS_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_src_rgb_), max_src_bytes_));
        AEGIS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_src_rgb_), max_src_bytes_));
    }
}

CudaPreprocessor::~CudaPreprocessor() {
    if (!zero_copy_ && d_src_rgb_ != nullptr) {
        cudaFree(d_src_rgb_);
    }
    if (h_src_rgb_ != nullptr) {
        cudaFreeHost(h_src_rgb_);
    }
}

void CudaPreprocessor::launch_fp32(
    const uint8_t* src_rgb_hwc,
    const backends::LetterboxMeta& meta,
    float* d_dst_nchw,
    cudaStream_t stream) {

    AEGIS_NVTX_RANGE("CUDA_Fused_Preprocess_FP32");
    const std::size_t frame_bytes = static_cast<std::size_t>(meta.src_w) * meta.src_h * 3;
    if (src_rgb_hwc != h_src_rgb_) {
        std::memcpy(h_src_rgb_, src_rgb_hwc, frame_bytes);
    }
    if (!zero_copy_) {
        AEGIS_CUDA_CHECK(cudaMemcpyAsync(d_src_rgb_, h_src_rgb_, frame_bytes, cudaMemcpyHostToDevice, stream));
    }

    const dim3 block(16, 16);
    const dim3 grid((meta.dst_w + block.x - 1) / block.x,
                    (meta.dst_h + block.y - 1) / block.y);
    fused_letterbox_bilinear_nchw_kernel<float><<<grid, block, 0, stream>>>(
        d_src_rgb_, meta.src_w, meta.src_h, meta.dst_w, meta.dst_h,
        meta.resized_w, meta.resized_h, meta.pad_x, meta.pad_y,
        1.0f / meta.scale, d_dst_nchw);
    AEGIS_CUDA_CHECK(cudaGetLastError());
}

void CudaPreprocessor::launch_fp16(
    const uint8_t* src_rgb_hwc,
    const backends::LetterboxMeta& meta,
    __half* d_dst_nchw,
    cudaStream_t stream) {

    AEGIS_NVTX_RANGE("CUDA_Fused_Preprocess_FP16");
    const std::size_t frame_bytes = static_cast<std::size_t>(meta.src_w) * meta.src_h * 3;
    if (src_rgb_hwc != h_src_rgb_) {
        std::memcpy(h_src_rgb_, src_rgb_hwc, frame_bytes);
    }
    if (!zero_copy_) {
        AEGIS_CUDA_CHECK(cudaMemcpyAsync(d_src_rgb_, h_src_rgb_, frame_bytes, cudaMemcpyHostToDevice, stream));
    }

    const dim3 block(16, 16);
    const dim3 grid((meta.dst_w + block.x - 1) / block.x,
                    (meta.dst_h + block.y - 1) / block.y);
    fused_letterbox_bilinear_nchw_kernel<__half><<<grid, block, 0, stream>>>(
        d_src_rgb_, meta.src_w, meta.src_h, meta.dst_w, meta.dst_h,
        meta.resized_w, meta.resized_h, meta.pad_x, meta.pad_y,
        1.0f / meta.scale, d_dst_nchw);
    AEGIS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace aegis::kernels

#endif  // AEGIS_ENABLE_CUDA
