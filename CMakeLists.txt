cmake_minimum_required(VERSION 3.24)
project(aegis_rt VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AEGIS_ENABLE_CUDA "Enable custom fused CUDA kernels and NVTX profiling" OFF)
option(AEGIS_ENABLE_TENSORRT "Enable TensorRT C++ FP32/FP16/INT8 backend" OFF)
option(AEGIS_ENABLE_JETSON_ZEROCOPY "Default to Jetson Orin Unified Memory zero-copy" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# ------------------------------------------------------------------------------
# Dependencies: Threads, GLFW, ONNX Runtime, MuJoCo, and optional CUDA/TensorRT
# ------------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(glfw3 REQUIRED)

# Discover ONNX Runtime (Homebrew / Linux / Jetson paths)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
        /opt/homebrew/opt/onnxruntime/include/onnxruntime
        /opt/homebrew/include/onnxruntime
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime
        $ENV{ONNXRUNTIME_ROOT}/include
)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS
        /opt/homebrew/opt/onnxruntime/lib
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib
        $ENV{ONNXRUNTIME_ROOT}/lib
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Install via Homebrew (`brew install onnxruntime`) or set ONNXRUNTIME_ROOT.")
endif()
message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY} (include: ${ONNXRUNTIME_INCLUDE_DIR})")

# Discover MuJoCo C Library (supports Python wheel bundle in .venv, macOS framework, or Linux /opt/mujoco)
set(MUJOCO_SEARCH_HINTS
    "${CMAKE_SOURCE_DIR}/.venv/lib"
    "/opt/mujoco"
    "/usr/local"
    "/opt/homebrew"
    "$ENV{MUJOCO_DIR}"
)
file(GLOB MUJOCO_VENV_SITE "${CMAKE_SOURCE_DIR}/.venv/lib/python*/site-packages/mujoco")

find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
    HINTS
        ${MUJOCO_VENV_SITE}/include
        /Library/Frameworks/mujoco.framework/Headers
        /Applications/MuJoCo.app/Contents/Frameworks/mujoco.framework/Headers
        ${MUJOCO_SEARCH_HINTS}
    PATH_SUFFIXES include
)

find_library(MUJOCO_LIBRARY
    NAMES mujoco
    HINTS
        ${MUJOCO_VENV_SITE}
        ${MUJOCO_VENV_SITE}/mujoco.framework
        /Library/Frameworks
        /Applications/MuJoCo.app/Contents/Frameworks
        ${MUJOCO_SEARCH_HINTS}
    PATH_SUFFIXES lib
)

# Handle Python macOS wheel where libmujoco.*.dylib lives inside mujoco.framework/Versions/A/
if(NOT MUJOCO_LIBRARY AND MUJOCO_VENV_SITE)
    file(GLOB MUJOCO_DYLIB_CANDIDATES
        "${MUJOCO_VENV_SITE}/mujoco.framework/Versions/Current/libmujoco*.dylib"
        "${MUJOCO_VENV_SITE}/libmujoco*.dylib"
        "${MUJOCO_VENV_SITE}/.dylibs/libmujoco*.dylib"
    )
    if(MUJOCO_DYLIB_CANDIDATES)
        list(GET MUJOCO_DYLIB_CANDIDATES 0 MUJOCO_LIBRARY)
    endif()
endif()

# ------------------------------------------------------------------------------
# Core Control & Telemetry Library (Zero external sim/GPU deps — fast unit tests)
# ------------------------------------------------------------------------------
add_library(aegis_core STATIC
    src/control/kalman_tracker.cpp
    src/control/controller.cpp
    src/control/rover_controller.cpp
)
target_include_directories(aegis_core PUBLIC include)
target_link_libraries(aegis_core PUBLIC Threads::Threads)

# ------------------------------------------------------------------------------
# Optional CUDA & TensorRT Acceleration
# ------------------------------------------------------------------------------
set(AEGIS_GPU_SOURCES "")
set(AEGIS_GPU_LIBS "")

if(AEGIS_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 86=RTX 3090/A10, 87=Jetson Orin Nano, 89=RTX 4090/L4
        set(CMAKE_CUDA_ARCHITECTURES "86;87;89")
    endif()
    list(APPEND AEGIS_GPU_SOURCES src/kernels/preprocess.cu)
    list(APPEND AEGIS_GPU_LIBS CUDA::cudart CUDA::nvtx3)
    add_compile_definitions(AEGIS_ENABLE_CUDA AEGIS_ENABLE_NVTX)
endif()

if(AEGIS_ENABLE_TENSORRT)
    if(NOT AEGIS_ENABLE_CUDA)
        message(FATAL_ERROR "AEGIS_ENABLE_TENSORRT requires AEGIS_ENABLE_CUDA=ON")
    endif()
    find_library(NVINFER_LIB nvinfer HINTS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/tensorrt/lib REQUIRED)
    find_library(NVONNXPARSER_LIB nvonnxparser HINTS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/tensorrt/lib REQUIRED)
    list(APPEND AEGIS_GPU_SOURCES src/backends/trt_backend.cpp)
    list(APPEND AEGIS_GPU_LIBS ${NVINFER_LIB} ${NVONNXPARSER_LIB})
    add_compile_definitions(AEGIS_ENABLE_TENSORRT)
endif()

# ------------------------------------------------------------------------------
# Full Runtime Executable (`aegis_rt`) when MuJoCo is present
# ------------------------------------------------------------------------------
if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIBRARY)
    message(STATUS "Found MuJoCo: ${MUJOCO_LIBRARY} (include: ${MUJOCO_INCLUDE_DIR})")
    add_executable(aegis_rt
        src/main.cpp
        src/sim/mujoco_bridge.cpp
        src/backends/ort_backend.cpp
        ${AEGIS_GPU_SOURCES}
    )
    target_include_directories(aegis_rt PRIVATE
        include
        ${MUJOCO_INCLUDE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(aegis_rt PRIVATE
        aegis_core
        glfw
        ${MUJOCO_LIBRARY}
        ${ONNXRUNTIME_LIBRARY}
        ${AEGIS_GPU_LIBS}
    )
    if(APPLE)
        get_filename_component(MUJOCO_LIB_DIR "${MUJOCO_LIBRARY}" DIRECTORY)
        get_filename_component(MUJOCO_LIB_NAME "${MUJOCO_LIBRARY}" NAME)
        set_target_properties(aegis_rt PROPERTIES
            BUILD_RPATH "${MUJOCO_LIB_DIR};/opt/homebrew/lib"
        )
        add_custom_command(TARGET aegis_rt POST_BUILD
            COMMAND install_name_tool -change
                "@rpath/mujoco.framework/Versions/A/${MUJOCO_LIB_NAME}"
                "${MUJOCO_LIBRARY}"
                "$<TARGET_FILE:aegis_rt>"
            COMMENT "Patching macOS MuJoCo wheel install_name on aegis_rt"
        )
    endif()
else()
    message(WARNING "MuJoCo C library not yet detected; building core + unit tests only.")
endif()

# ------------------------------------------------------------------------------
# Automated Unit & Numerical Parity Tests (`ctest`)
# ------------------------------------------------------------------------------
enable_testing()

add_executable(test_spsc_queue tests/test_spsc_queue.cpp)
target_link_libraries(test_spsc_queue PRIVATE aegis_core)
add_test(NAME SpscQueueLockFreeTest COMMAND test_spsc_queue)

add_executable(test_kalman_control tests/test_kalman_control.cpp)
target_link_libraries(test_kalman_control PRIVATE aegis_core)
add_test(NAME KalmanAndSe3ControllerTest COMMAND test_kalman_control)

add_executable(test_letterbox_decode tests/test_letterbox_decode.cpp)
target_link_libraries(test_letterbox_decode PRIVATE aegis_core)
add_test(NAME LetterboxAndDecodeParityTest COMMAND test_letterbox_decode)

if(AEGIS_ENABLE_CUDA)
    add_executable(test_cuda_preprocess tests/test_cuda_preprocess.cu src/kernels/preprocess.cu)
    target_include_directories(test_cuda_preprocess PRIVATE include)
    target_link_libraries(test_cuda_preprocess PRIVATE ${AEGIS_GPU_LIBS})
    add_test(NAME CudaPreprocessParityTest COMMAND test_cuda_preprocess)
endif()
