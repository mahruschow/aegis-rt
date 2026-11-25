# Contributing to Aegis-RT

First off — **welcome, and thank you for checking out Aegis-RT!** Whether you're a student exploring robotics for the first time, a systems engineer, or someone tinkering with CUDA kernels and edge hardware, contributions, questions, and ideas are genuinely welcome here.

## Ways to Get Involved

* **Add a New Vehicle Plant:** Want to add a bipedal robot, Ackermann racecar with tire slip dynamics, or submarine in `assets/scenes/`? Follow the pattern in `rover_arena.xml` and `src/control/rover_controller.cpp`.
* **New Inference Backends:** Metal compute shaders, AMD ROCm/MIGraphX, Vulkan Kompute, or hailo/NPU backends implementing `IPerceptionBackend` (`include/aegis/backends/i_backend.hpp`).
* **Sim-to-Real Hardware Drivers:** V4L2 / GStreamer CSI camera producers and MAVLink / ROS2 actuator bridges for physical drones and rovers.
* **Documentation & Tutorials:** Clarifying math derivations, setup scripts, or benchmark reproductions on new GPUs/SBCs.

## Local Development Workflow

1. Fork and clone the repository.
2. Run `bash scripts/setup_mac.sh` (macOS Apple Silicon) or use `Dockerfile.cuda` (Linux GPU).
3. Verify all unit tests pass before submitting a pull request:
   ```bash
   cmake --build build -j && ctest --test-dir build --output-on-failure -V
   ```
4. Open a Pull Request or Discussion — no question is too small!
