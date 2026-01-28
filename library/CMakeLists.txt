# Build and export the 'clutra' target (interface/header + CUDA runtime)
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

# Device-only CUDA sources that must be linked into final executables.
file(GLOB_RECURSE CLUTRA_DEVICE_SOURCES
  ${PROJECT_SOURCE_DIR}/library/device/*.cu
)
if(CLUTRA_DEVICE_SOURCES)
  add_library(clutra_device_objs OBJECT ${CLUTRA_DEVICE_SOURCES})
  set_target_properties(clutra_device_objs PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_include_directories(clutra_device_objs PRIVATE
    ${PROJECT_SOURCE_DIR}/library/include
  )
endif()

if (ENABLE_PROFILING)
  target_compile_definitions(clutra PUBLIC ENABLE_PROFILING)
endif()

target_compile_features(clutra PUBLIC cxx_std_20)
target_include_directories(clutra PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(clutra PUBLIC fmt::fmt-header-only)
target_compile_definitions(clutra PUBLIC
  $<$<CONFIG:Debug>:CLUTRA_DEBUG_LOG=1>
  $<$<NOT:$<CONFIG:Debug>>:CLUTRA_DEBUG_LOG=0>
)

# If the library was created as a compiled target, set properties and link CUDA runtime
get_target_property(_clutra_type clutra TYPE)
if(NOT _clutra_type STREQUAL "INTERFACE_LIBRARY")
  set_target_properties(clutra PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
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
