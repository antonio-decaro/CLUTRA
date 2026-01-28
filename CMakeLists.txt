cmake_minimum_required(VERSION 3.25)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set explicit CUDA standard (avoid CMake inferring unsupported CUDA20)
# Use a CUDA-compatible host/device standard (choose 14 or 17 depending on your CUDA toolkit).
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(ENABLE_PROFILING "Enable CUDA kernel profiling instrumentation" OFF)

# Define the project
project(CLUTRA
    VERSION 0.1.0
    DESCRIPTION "CLUTRA: A CUDA Library for Graph Traversal"
    LANGUAGES CXX CUDA
)

# Fetch header-only dependencies.
include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fmt)

# Find CUDA toolkit (provides Thrust headers and CUDA runtime target)
find_package(CUDAToolkit REQUIRED)

# Add subdirectories: library will expose the headers/target, src builds the executable
add_subdirectory(library)
add_subdirectory(examples)
