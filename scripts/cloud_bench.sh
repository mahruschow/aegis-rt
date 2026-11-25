#!/usr/bin/env bash
# ==============================================================================
# Aegis-RT One-Click Cloud GPU Benchmark & Nsight Systems Profiling Suite
# Runs:
#   1. Lock-free SPSC, Kalman/SE(3), and CUDA Fused Kernel numerical parity tests
#   2. Synthetic MuJoCo RGB-D dataset + INT8 calibration frame capture
#   3. AegisNanoDetector training & static ONNX export
#   4. 1,000-frame closed-loop benchmarks across ORT-CPU, ORT-CUDA, TRT-FP32/FP16/INT8
#   5. NVIDIA Nsight Systems (`nsys`) timeline & Nsight Compute (`ncu`) kernel metrics
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
mkdir -p results assets/models assets/calib_frames

echo "==> [1/6] Compiling Aegis-RT with CUDA + TensorRT enabled..."
cmake -S . -B build -G Ninja \
    -DAEGIS_ENABLE_CUDA=ON \
    -DAEGIS_ENABLE_TENSORRT=ON
cmake --build build -j

echo "==> [2/6] Running C++20 & CUDA Numerical Parity Unit Tests..."
ctest --test-dir build --output-on-failure -V

echo "==> [3/6] Capturing 450 MuJoCo FPV frames for detector training & INT8 PTQ calibration..."
xvfb-run -a ./build/aegis_rt \
    --backend gt \
    --frames 450 \
    --dump-dataset assets/calib_frames \
    --csv results/gt_oracle.csv

echo "==> [4/6] Training AegisNanoDetector & exporting static opset-17 ONNX..."
"${PYTHON_BIN}" python/export_model.py \
    --dataset-dir assets/calib_frames \
    --out assets/models/aegis_detector.onnx \
    --epochs 60

echo "==> [5/6] Benchmarking Closed-Loop 6-DOF Flight across Precision Tiers (1,000 frames each)..."
for BACKEND in ort-cpu ort-cuda trt-fp32 trt-fp16 trt-int8; do
    echo "--- Running backend: ${BACKEND} ---"
    xvfb-run -a ./build/aegis_rt \
        --backend "${BACKEND}" \
        --frames 1000 \
        --calib-dir assets/calib_frames \
        --csv "results/${BACKEND}.csv"
done

echo "==> [6/6] Capturing NVIDIA Nsight Systems NVTX Trace & Generating Plots..."
if command -v nsys >/dev/null 2>&1; then
    xvfb-run -a nsys profile \
        --trace=cuda,nvtx,osrt \
        --output=results/aegis_trt_int8_trace \
        --force-overwrite=true \
        ./build/aegis_rt --backend trt-int8 --frames 300
fi

if command -v ncu >/dev/null 2>&1; then
    ncu --target-processes all \
        --kernel-name regex:fused_letterbox_bilinear_nchw_kernel \
        --launch-skip 5 --launch-count 1 \
        --export results/aegis_preprocess_kernel \
        --force-overwrite \
        ./build/test_cuda_preprocess
fi

"${PYTHON_BIN}" python/plot_benchmarks.py \
    --inputs \
      "ORT-CPU=results/ort-cpu.csv" \
      "ORT-CUDA=results/ort-cuda.csv" \
      "TRT-FP32=results/trt-fp32.csv" \
      "TRT-FP16=results/trt-fp16.csv" \
      "TRT-INT8=results/trt-int8.csv" \
    --out results/aegis_benchmark_report.png

echo "==> Complete! Artifacts saved under ${ROOT_DIR}/results/"
