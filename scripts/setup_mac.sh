#!/usr/bin/env bash
# ==============================================================================
# Aegis-RT Local Apple Silicon macOS Setup & End-to-End Build Script
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "==> Checking Homebrew dependencies..."
brew install cmake ninja onnxruntime glfw

if [ ! -d ".venv" ]; then
    echo "==> Creating Python virtual environment..."
    /opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv || python3 -m venv .venv
fi

echo "==> Installing Python packages (MuJoCo, PyTorch, ONNX, Matplotlib)..."
.venv/bin/pip install --upgrade mujoco onnx numpy matplotlib torch

echo "==> Configuring & Building Aegis-RT C++20 runtime..."
cmake -S . -B build -G Ninja
cmake --build build -j

echo "==> Running unit tests..."
ctest --test-dir build --output-on-failure -V
