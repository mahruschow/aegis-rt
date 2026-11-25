#if defined(AEGIS_ENABLE_CUDA)

#include "aegis/backends/i_backend.hpp"
#include "aegis/kernels/preprocess.cuh"

#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    using namespace aegis;
    constexpr int kSrcW = 640;
    constexpr int kSrcH = 480;
    constexpr int kDstW = 320;
    constexpr int kDstH = 320;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> src_rgb(kSrcW * kSrcH * 3);
    for (auto& px : src_rgb) {
        px = static_cast<uint8_t>(dist(rng));
    }

    const auto meta = backends::compute_letterbox_meta(kSrcW, kSrcH, kDstW, kDstH);
    std::vector<float> cpu_ref(3 * kDstW * kDstH, 0.0f);
    backends::cpu_preprocess_letterbox_nchw(src_rgb.data(), meta, cpu_ref.data());

    kernels::CudaPreprocessor cuda_pre(kSrcW, kSrcH, kDstW, kDstH, false);
    float* d_out = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_out), cpu_ref.size() * sizeof(float));
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cuda_pre.launch_fp32(src_rgb.data(), meta, d_out, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> gpu_out(cpu_ref.size(), 0.0f);
    cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float max_abs_diff = 0.0f;
    for (std::size_t i = 0; i < cpu_ref.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::abs(cpu_ref[i] - gpu_out[i]));
    }

    cudaFree(d_out);
    cudaStreamDestroy(stream);

    std::cout << "[PASS] test_cuda_preprocess: max_abs_diff vs CPU golden = " << max_abs_diff << "\n";
    assert(max_abs_diff < 1e-5f);
    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "[SKIP] test_cuda_preprocess: AEGIS_ENABLE_CUDA=OFF on this host.\n";
    return 0;
}
#endif
