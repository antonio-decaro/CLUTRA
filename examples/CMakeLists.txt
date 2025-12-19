# src/CMakeLists.txt
# Build the clutra_test executable and link against the 'clutra' interface target

# Source file for the test binary
include(FetchContent)

# Gather every CUDA example source in the src folder
file(GLOB EXAMPLE_SRCS
  ${PROJECT_SOURCE_DIR}/examples/src/*.cu
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

# Create a binary per CUDA source file so each example can be run independently
foreach(example_src ${EXAMPLE_SRCS})
  get_filename_component(example_name ${example_src} NAME_WE)
  set(target_name clutra_${example_name})

  add_executable(${target_name} ${example_src})

  # Ensure each target uses compatible language standards
  set_target_properties(${target_name} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )

  target_compile_features(${target_name} PRIVATE cxx_std_20)

  # Link against the library target and CUDA runtime
  target_link_libraries(${target_name} PRIVATE clutra CLI11::CLI11)
  if (TARGET CUDA::cudart)
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
  endif()

  if(ENABLE_PROFILING)
    target_compile_definitions(${target_name} PUBLIC ENABLE_PROFILING)
  endif()

  # Apply nvcc-specific flags: enable extended lambdas in device code
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  )
endforeach()
