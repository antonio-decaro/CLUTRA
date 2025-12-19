# library/CMakeLists.txt
# Build and export the 'clutra' target (interface/header + CUDA runtime)

# Build the library target from sources under library/src and expose headers in include/
file(GLOB_RECURSE CLUTRA_SOURCES
  ${PROJECT_SOURCE_DIR}/library/src/*.cu
  ${PROJECT_SOURCE_DIR}/library/src/*.cpp
  ${PROJECT_SOURCE_DIR}/library/src/*.c
  ${PROJECT_SOURCE_DIR}/library/src/io/*.cpp
)

if(CLUTRA_SOURCES)
  add_library(clutra STATIC ${CLUTRA_SOURCES})
else()
  # Fallback to an object library if there are no sources (keeps interface stable)
  add_library(clutra INTERFACE)
endif()

if (ENABLE_PROFILING)
  target_compile_definitions(clutra PUBLIC ENABLE_PROFILING)
endif()

# Public include directory
target_compile_features(clutra PUBLIC cxx_std_20)
target_include_directories(clutra PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  $<INSTALL_INTERFACE:include>
)

 # If the library was created as a compiled target, set properties and link CUDA runtime
get_target_property(_clutra_type clutra TYPE)
if(NOT _clutra_type STREQUAL "INTERFACE_LIBRARY")
  set_target_properties(clutra PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )

  # Pass nvcc-specific flags for device lambdas
  target_compile_options(clutra PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  )

  if (TARGET CUDA::cudart)
    target_link_libraries(clutra PRIVATE CUDA::cudart)
  endif()

  if (DEFINED CUDAToolkit_INCLUDE_DIRS)
    target_include_directories(clutra PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endif()
endif()
