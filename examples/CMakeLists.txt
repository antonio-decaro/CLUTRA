# src/CMakeLists.txt
# Build the clutra_test executable and link against the 'clutra' interface target

# Source file for the test binary
include(FetchContent)

set(TEST_SRCS
  ${PROJECT_SOURCE_DIR}/examples/src/test.cu
)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.1
  GIT_SHALLOW TRUE
)

FetchContent_GetProperties(cli11)
if(NOT cli11_POPULATED)
  FetchContent_Populate(cli11)
  add_subdirectory(${cli11_SOURCE_DIR} ${cli11_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()


include_directories(${PROJECT_SOURCE_DIR}/examples/include)

add_executable(clutra_test ${TEST_SRCS})

# Ensure the target uses compatible language standards
set_target_properties(clutra_test PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

# Link against the library target and CUDA runtime
target_link_libraries(clutra_test PRIVATE clutra CLI11::CLI11)
if (TARGET CUDA::cudart)
  target_link_libraries(clutra_test PRIVATE CUDA::cudart)
endif()

if(ENABLE_PROFILING)
  target_compile_definitions(clutra_test PUBLIC ENABLE_PROFILING)
endif()

# Apply nvcc-specific flags: enable extended lambdas in device code
target_compile_options(clutra_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

