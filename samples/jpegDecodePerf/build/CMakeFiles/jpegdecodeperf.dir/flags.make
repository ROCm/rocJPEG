# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# compile CXX with /opt/rocm/llvm/bin/clang++
CXX_DEFINES = -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1

CXX_INCLUDES = -I/opt/rocm/include/rocjpeg -isystem /usr/include/libdrm -isystem /opt/rocm-5.6.0/include

CXX_FLAGS =  -std=gnu++17 -std=gnu++17 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -x hip --offload-arch=gfx803 --offload-arch=gfx900 --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx90a --offload-arch=gfx940 --offload-arch=gfx1030 --offload-arch=gfx1031 --offload-arch=gfx1032

