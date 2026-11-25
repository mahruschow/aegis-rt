#!/usr/bin/env bash
# ==============================================================================
# Aegis-RT NVIDIA Jetson Orin Nano Edge Deployment & FPS/Watt Profiler
# Uses LPDDR5 Unified Memory Zero-Copy (`cudaHostAllocMapped`) + `tegrastats`
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p results

echo "==> Building Aegis-RT for Jetson Orin aarch64 (SM 87) with Unified Memory Zero-Copy..."
cmake -S . -B build -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DAEGIS_ENABLE_CUDA=ON \
    -DAEGIS_ENABLE_TENSORRT=ON \
    -DAEGIS_ENABLE_JETSON_ZEROCOPY=ON
cmake --build build -j

TEGRA_LOG="results/jetson_tegrastats.log"
if command -v tegrastats >/dev/null 2>&1; then
    echo "==> Starting background tegrastats power telemetry (100ms interval)..."
    tegrastats --interval 100 --logfile "${TEGRA_LOG}" &
    TEGRA_PID=$!
    trap 'kill ${TEGRA_PID} 2>/dev/null || true' EXIT
fi

for BACKEND in trt-fp16 trt-int8; do
    echo "==> Running Zero-Copy Edge Autonomy on ${BACKEND}..."
    ./build/aegis_rt \
        --backend "${BACKEND}" \
        --jetson-zerocopy \
        --frames 1000 \
        --calib-dir assets/calib_frames \
        --csv "results/jetson_${BACKEND}_zerocopy.csv"
done

echo "==> Jetson Orin Nano benchmark complete. Power log: ${TEGRA_LOG}"
