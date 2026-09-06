cmake_minimum_required(VERSION 3.22.1)
project(hymt2_jni)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
# Target the Snapdragon 8 Gen 2-class ARMv9 CPUs used by the device.
# This deliberately drops generic ARM64 compatibility to use DotProd and I8MM.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv8.6-a+dotprod+i8mm" CACHE STRING "" FORCE)

add_subdirectory(third_party/llama.cpp)

add_library(hymt2_jni SHARED hymt2_jni.cpp)

target_compile_features(hymt2_jni PRIVATE cxx_std_17)
target_link_libraries(hymt2_jni PRIVATE llama log android)
