/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rocjpeg_hip_kernels.h"

__device__ __forceinline__ uint32_t hipPack(float4 src) {
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

__device__ __forceinline__ float hipUnpack0(uint32_t src) {
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float hipUnpack1(uint32_t src) {
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float hipUnpack2(uint32_t src) {
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float hipUnpack3(uint32_t src) {
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 hipUnpack(uint32_t src) {
    return make_float4(hipUnpack0(src), hipUnpack1(src), hipUnpack2(src), hipUnpack3(src));
}

__global__ void ColorConvertYUV444ToRGBKernel(uint32_t dst_width, uint32_t dst_height, uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    uint32_t dst_image_stride_in_bytes_comp, const uint8_t *src_y_image, const uint8_t *src_u_image, const uint8_t *src_v_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_yuv_image_stride_in_bytes_comp) {

    int32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_yuv_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_yuv_image_stride_in_bytes;


        uint2 y0 = *((uint2 *)(&src_y_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_y_image[src_y1_idx]));

        uint2 u0 = *((uint2 *)(&src_u_image[src_y0_idx]));
        uint2 u1 = *((uint2 *)(&src_u_image[src_y1_idx]));

        uint2 v0 = *((uint2 *)(&src_v_image[src_y0_idx]));
        uint2 v1 = *((uint2 *)(&src_v_image[src_y1_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 rgb0, rgb1;
        float4 f;

        yuv.x = hipUnpack0(y0.x);
        yuv.y = hipUnpack0(u0.x);
        yuv.z = hipUnpack0(v0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(y0.x);
        yuv.y = hipUnpack1(u0.x);
        yuv.z = hipUnpack1(v0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(y0.x);
        yuv.y = hipUnpack2(u0.x);
        yuv.z = hipUnpack2(v0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(y0.x);
        yuv.y = hipUnpack3(u0.x);
        yuv.z = hipUnpack3(v0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[2] = hipPack(f);

        yuv.x = hipUnpack0(y0.y);
        yuv.y = hipUnpack0(u0.y);
        yuv.z = hipUnpack0(v0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(y0.y);
        yuv.y = hipUnpack1(u0.y);
        yuv.z = hipUnpack1(v0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(y0.y);
        yuv.y = hipUnpack2(u0.y);
        yuv.z = hipUnpack2(v0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(y0.y);
        yuv.y = hipUnpack3(u0.y);
        yuv.z = hipUnpack3(v0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[5] = hipPack(f);

        yuv.x = hipUnpack0(y1.x);
        yuv.y = hipUnpack0(u1.x);
        yuv.z = hipUnpack0(v1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(y1.x);
        yuv.y = hipUnpack1(u1.x);
        yuv.z = hipUnpack1(v1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(y1.x);
        yuv.y = hipUnpack2(u1.x);
        yuv.z = hipUnpack2(v1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(y1.x);
        yuv.y = hipUnpack3(u1.x);
        yuv.z = hipUnpack3(v1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[2] = hipPack(f);

        yuv.x = hipUnpack0(y1.y);
        yuv.y = hipUnpack0(u1.y);
        yuv.z = hipUnpack0(v1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(y1.y);
        yuv.y = hipUnpack1(u1.y);
        yuv.z = hipUnpack1(v1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(y1.y);
        yuv.y = hipUnpack2(u1.y);
        yuv.z = hipUnpack2(v1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(y1.y);
        yuv.y = hipUnpack3(u1.y);
        yuv.z = hipUnpack3(v1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[5] = hipPack(f);

        *((DUINT6 *)(&dst_image[rgb0_idx])) = rgb0;
        *((DUINT6 *)(&dst_image[rgb1_idx])) = rgb1;
    }
}

void ColorConvertYUV444ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_yuv_image_stride_in_bytes_comp = src_yuv_image_stride_in_bytes * 2;

    ColorConvertYUV444ToRGBKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_width, dst_height, (uint8_t *)dst_image,
                        dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp, src_yuv_image, src_yuv_image + src_u_image_offset,
                        src_yuv_image + (src_u_image_offset * 2), src_yuv_image_stride_in_bytes,
                        dst_width_comp, dst_height_comp, src_yuv_image_stride_in_bytes_comp);
}

__global__ void ColorConvertYUYVToRGBKernel(uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, uint32_t dst_image_stride_in_bytes_comp,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes, uint32_t src_image_stride_in_bytes_comp,
    uint32_t dst_width_comp, uint32_t dst_height_comp) {

    int32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t l0_idx = y * src_image_stride_in_bytes_comp + (x << 4);
        uint32_t l1_idx = l0_idx + src_image_stride_in_bytes;
        uint4 l0 = *((uint4 *)(&src_image[l0_idx]));
        uint4 l1 = *((uint4 *)(&src_image[l1_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float4 f;

        uint2 py0, py1;
        uint2 pu0, pu1;
        uint2 pv0, pv1;

        py0.x = hipPack(make_float4(hipUnpack0(l0.x), hipUnpack2(l0.x), hipUnpack0(l0.y), hipUnpack2(l0.y)));
        py0.y = hipPack(make_float4(hipUnpack0(l0.z), hipUnpack2(l0.z), hipUnpack0(l0.w), hipUnpack2(l0.w)));
        py1.x = hipPack(make_float4(hipUnpack0(l1.x), hipUnpack2(l1.x), hipUnpack0(l1.y), hipUnpack2(l1.y)));
        py1.y = hipPack(make_float4(hipUnpack0(l1.z), hipUnpack2(l1.z), hipUnpack0(l1.w), hipUnpack2(l1.w)));
        pu0.x = hipPack(make_float4(hipUnpack1(l0.x), hipUnpack1(l0.x), hipUnpack1(l0.y), hipUnpack1(l0.y)));
        pu0.y = hipPack(make_float4(hipUnpack1(l0.z), hipUnpack1(l0.z), hipUnpack1(l0.w), hipUnpack1(l0.w)));
        pu1.x = hipPack(make_float4(hipUnpack1(l1.x), hipUnpack1(l1.x), hipUnpack1(l1.y), hipUnpack1(l1.y)));
        pu1.y = hipPack(make_float4(hipUnpack1(l1.z), hipUnpack1(l1.z), hipUnpack1(l1.w), hipUnpack1(l1.w)));
        pv0.x = hipPack(make_float4(hipUnpack3(l0.x), hipUnpack3(l0.x), hipUnpack3(l0.y), hipUnpack3(l0.y)));
        pv0.y = hipPack(make_float4(hipUnpack3(l0.z), hipUnpack3(l0.z), hipUnpack3(l0.w), hipUnpack3(l0.w)));
        pv1.x = hipPack(make_float4(hipUnpack3(l1.x), hipUnpack3(l1.x), hipUnpack3(l1.y), hipUnpack3(l1.y)));
        pv1.y = hipPack(make_float4(hipUnpack3(l1.z), hipUnpack3(l1.z), hipUnpack3(l1.w), hipUnpack3(l1.w)));

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 prgb0, prgb1;

        yuv.x = hipUnpack0(py0.x);
        yuv.y = hipUnpack0(pu0.x);
        yuv.z = hipUnpack0(pv0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(py0.x);
        yuv.y = hipUnpack1(pu0.x);
        yuv.z = hipUnpack1(pv0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(py0.x);
        yuv.y = hipUnpack2(pu0.x);
        yuv.z = hipUnpack2(pv0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(py0.x);
        yuv.y = hipUnpack3(pu0.x);
        yuv.z = hipUnpack3(pv0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb0.data[2] = hipPack(f);

        yuv.x = hipUnpack0(py0.y);
        yuv.y = hipUnpack0(pu0.y);
        yuv.z = hipUnpack0(pv0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(py0.y);
        yuv.y = hipUnpack1(pu0.y);
        yuv.z = hipUnpack1(pv0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(py0.y);
        yuv.y = hipUnpack2(pu0.y);
        yuv.z = hipUnpack2(pv0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(py0.y);
        yuv.y = hipUnpack3(pu0.y);
        yuv.z = hipUnpack3(pv0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb0.data[5] = hipPack(f);

        yuv.x = hipUnpack0(py1.x);
        yuv.y = hipUnpack0(pu1.x);
        yuv.z = hipUnpack0(pv1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(py1.x);
        yuv.y = hipUnpack1(pu1.x);
        yuv.z = hipUnpack1(pv1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(py1.x);
        yuv.y = hipUnpack2(pu1.x);
        yuv.z = hipUnpack2(pv1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(py1.x);
        yuv.y = hipUnpack3(pu1.x);
        yuv.z = hipUnpack3(pv1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb1.data[2] = hipPack(f);

        yuv.x = hipUnpack0(py1.y);
        yuv.y = hipUnpack0(pu1.y);
        yuv.z = hipUnpack0(pv1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(py1.y);
        yuv.y = hipUnpack1(pu1.y);
        yuv.z = hipUnpack1(pv1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        prgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(py1.y);
        yuv.y = hipUnpack2(pu1.y);
        yuv.z = hipUnpack2(pv1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        prgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(py1.y);
        yuv.y = hipUnpack3(pu1.y);
        yuv.z = hipUnpack3(pv1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        prgb1.data[5] = hipPack(f);

        *((DUINT6 *)(&dst_image[rgb0_idx])) = prgb0;
        *((DUINT6 *)(&dst_image[rgb1_idx])) = prgb1;
    }
}

void ColorConvertYUYVToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {
    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_image_stride_in_bytes_comp = src_image_stride_in_bytes * 2;

    ColorConvertYUYVToRGBKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                                   dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_width, dst_height, (uint8_t *)dst_image,
                                   dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp, src_image, src_image_stride_in_bytes,
                                   src_image_stride_in_bytes_comp, dst_width_comp, dst_height_comp);
}

__global__ void ColorConvertNV12ToRGBKernel(uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, uint32_t dst_image_stride_in_bytes_comp,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes,
    uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_luma_image_stride_in_bytes_comp) {

    int32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_luma_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_luma_image_stride_in_bytes;
        uint32_t src_uv_idx = y * src_chroma_image_stride_in_bytes + (x << 3);
        uint2 y0 = *((uint2 *)(&src_luma_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_luma_image[src_y1_idx]));
        uint2 uv = *((uint2 *)(&src_chroma_image[src_uv_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float4 f;
        uint2 u0, u1;
        uint2 v0, v1;

        f.x = hipUnpack0(uv.x);
        f.y = f.x;
        f.z = hipUnpack2(uv.x);
        f.w = f.z;
        u0.x = hipPack(f);

        f.x = hipUnpack0(uv.y);
        f.y = f.x;
        f.z = hipUnpack2(uv.y);
        f.w = f.z;
        u0.y = hipPack(f);

        u1.x = u0.x;
        u1.y = u0.y;

        f.x = hipUnpack1(uv.x);
        f.y = f.x;
        f.z = hipUnpack3(uv.x);
        f.w = f.z;
        v0.x = hipPack(f);

        f.x = hipUnpack1(uv.y);
        f.y = f.x;
        f.z = hipUnpack3(uv.y);
        f.w = f.z;
        v0.y = hipPack(f);

        v1.x = v0.x;
        v1.y = v0.y;

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        DUINT6 rgb0, rgb1;

        yuv.x = hipUnpack0(y0.x);
        yuv.y = hipUnpack0(u0.x);
        yuv.z = hipUnpack0(v0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(y0.x);
        yuv.y = hipUnpack1(u0.x);
        yuv.z = hipUnpack1(v0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(y0.x);
        yuv.y = hipUnpack2(u0.x);
        yuv.z = hipUnpack2(v0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(y0.x);
        yuv.y = hipUnpack3(u0.x);
        yuv.z = hipUnpack3(v0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[2] = hipPack(f);

        yuv.x = hipUnpack0(y0.y);
        yuv.y = hipUnpack0(u0.y);
        yuv.z = hipUnpack0(v0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(y0.y);
        yuv.y = hipUnpack1(u0.y);
        yuv.z = hipUnpack1(v0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb0.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(y0.y);
        yuv.y = hipUnpack2(u0.y);
        yuv.z = hipUnpack2(v0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb0.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(y0.y);
        yuv.y = hipUnpack3(u0.y);
        yuv.z = hipUnpack3(v0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb0.data[5] = hipPack(f);

        yuv.x = hipUnpack0(y1.x);
        yuv.y = hipUnpack0(u1.x);
        yuv.z = hipUnpack0(v1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(y1.x);
        yuv.y = hipUnpack1(u1.x);
        yuv.z = hipUnpack1(v1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[0] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(y1.x);
        yuv.y = hipUnpack2(u1.x);
        yuv.z = hipUnpack2(v1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[1] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(y1.x);
        yuv.y = hipUnpack3(u1.x);
        yuv.z = hipUnpack3(v1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[2] = hipPack(f);

        yuv.x = hipUnpack0(y1.y);
        yuv.y = hipUnpack0(u1.y);
        yuv.z = hipUnpack0(v1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cr.y, yuv.z, yuv.x);
        f.y = fmaf(cg.x, yuv.y, yuv.x);
        f.y = fmaf(cg.y, yuv.z, f.y);
        f.z = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack1(y1.y);
        yuv.y = hipUnpack1(u1.y);
        yuv.z = hipUnpack1(v1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cr.y, yuv.z, yuv.x);
        rgb1.data[3] = hipPack(f);

        f.x = fmaf(cg.x, yuv.y, yuv.x);
        f.x = fmaf(cg.y, yuv.z, f.x);
        f.y = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack2(y1.y);
        yuv.y = hipUnpack2(u1.y);
        yuv.z = hipUnpack2(v1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cr.y, yuv.z, yuv.x);
        f.w = fmaf(cg.x, yuv.y, yuv.x);
        f.w = fmaf(cg.y, yuv.z, f.w);
        rgb1.data[4] = hipPack(f);

        f.x = fmaf(cb.x, yuv.y, yuv.x);
        yuv.x = hipUnpack3(y1.y);
        yuv.y = hipUnpack3(u1.y);
        yuv.z = hipUnpack3(v1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cr.y, yuv.z, yuv.x);
        f.z = fmaf(cg.x, yuv.y, yuv.x);
        f.z = fmaf(cg.y, yuv.z, f.z);
        f.w = fmaf(cb.x, yuv.y, yuv.x);
        rgb1.data[5] = hipPack(f);

        *((DUINT6 *)(&dst_image[rgb0_idx])) = rgb0;
        *((DUINT6 *)(&dst_image[rgb1_idx])) = rgb1;
    }
}

void ColorConvertNV12ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes) {
    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_luma_image_stride_in_bytes_comp = src_luma_image_stride_in_bytes * 2;

    ColorConvertNV12ToRGBKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_width, dst_height, dst_image, dst_image_stride_in_bytes,
                        dst_image_stride_in_bytes_comp, src_luma_image, src_luma_image_stride_in_bytes, src_chroma_image,
                        src_chroma_image_stride_in_bytes, dst_width_comp, dst_height_comp, src_luma_image_stride_in_bytes_comp);
}

__global__ void ColorConvertYUV400ToRGBKernel(uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, uint32_t dst_image_stride_in_bytes_comp,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    uint32_t dst_width_comp, uint32_t dst_height_comp, uint32_t src_luma_image_stride_in_bytes_comp) {

    int32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint32_t src_y0_idx = y * src_luma_image_stride_in_bytes_comp + (x << 3);
        uint32_t src_y1_idx = src_y0_idx + src_luma_image_stride_in_bytes;

        uint2 y0 = *((uint2 *)(&src_luma_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_luma_image[src_y1_idx]));

        uint32_t rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint32_t rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        DUINT6 rgb0, rgb1;

        uint8_t y0_b0, y0_b1, y0_b2, y0_b3, y0_b4, y0_b5, y0_b6, y0_b7;
        uint8_t y1_b0, y1_b1, y1_b2, y1_b3, y1_b4, y1_b5, y1_b6, y1_b7;

        y0_b0 = hipUnpack0(y0.x);
        y0_b1 = hipUnpack1(y0.x);
        y0_b2 = hipUnpack2(y0.x);
        y0_b3 = hipUnpack3(y0.x);
        y0_b4 = hipUnpack0(y0.y);
        y0_b5 = hipUnpack1(y0.y);
        y0_b6 = hipUnpack2(y0.y);
        y0_b7 = hipUnpack3(y0.y);

        y1_b0 = hipUnpack0(y1.x);
        y1_b1 = hipUnpack1(y1.x);
        y1_b2 = hipUnpack2(y1.x);
        y1_b3 = hipUnpack3(y1.x);
        y1_b4 = hipUnpack0(y1.y);
        y1_b5 = hipUnpack1(y1.y);
        y1_b6 = hipUnpack2(y1.y);
        y1_b7 = hipUnpack3(y1.y);

        rgb0.data[0] = hipPack(make_float4(y0_b0, y0_b0, y0_b0, y0_b1));
        rgb0.data[1] = hipPack(make_float4(y0_b1, y0_b1, y0_b2, y0_b2));
        rgb0.data[2] = hipPack(make_float4(y0_b2, y0_b3, y0_b3, y0_b3));
        rgb0.data[3] = hipPack(make_float4(y0_b4, y0_b4, y0_b4, y0_b5));
        rgb0.data[4] = hipPack(make_float4(y0_b5, y0_b5, y0_b6, y0_b6));
        rgb0.data[5] = hipPack(make_float4(y0_b6, y0_b7, y0_b7, y0_b7));

        rgb1.data[0] = hipPack(make_float4(y1_b0, y1_b0, y1_b0, y1_b1));
        rgb1.data[1] = hipPack(make_float4(y1_b1, y1_b1, y1_b2, y1_b2));
        rgb1.data[2] = hipPack(make_float4(y1_b2, y1_b3, y1_b3, y1_b3));
        rgb1.data[3] = hipPack(make_float4(y1_b4, y1_b4, y1_b4, y1_b5));
        rgb1.data[4] = hipPack(make_float4(y1_b5, y1_b5, y1_b6, y1_b6));
        rgb1.data[5] = hipPack(make_float4(y1_b6, y1_b7, y1_b7, y1_b7));

        *((DUINT6 *)(&dst_image[rgb0_idx])) = rgb0;
        *((DUINT6 *)(&dst_image[rgb1_idx])) = rgb1;
    }
}

void ColorConvertYUV400ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes){

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_luma_image_stride_in_bytes_comp = src_luma_image_stride_in_bytes * 2;

    ColorConvertYUV400ToRGBKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                        dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_width, dst_height, dst_image, dst_image_stride_in_bytes,
                        dst_image_stride_in_bytes_comp, src_luma_image, src_luma_image_stride_in_bytes, dst_width_comp, dst_height_comp,
                        src_luma_image_stride_in_bytes_comp);

}


__global__ void ConvertInterleavedUVToPlanarUVKernel(uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image1, uint8_t *dst_image2, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {

    int32_t x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dst_width || y >= dst_height) {
        return;
    }

    uint32_t src_idx = y * src_image_stride_in_bytes + x + x;
    uint32_t dst_idx = y * dst_image_stride_in_bytes + x;

    uint4 src = *((uint4 *)(&src_image[src_idx]));
    uint2 dst1, dst2;

    dst1.x = hipPack(make_float4(hipUnpack0(src.x), hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y)));
    dst1.y = hipPack(make_float4(hipUnpack0(src.z), hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w)));
    dst2.x = hipPack(make_float4(hipUnpack1(src.x), hipUnpack3(src.x), hipUnpack1(src.y), hipUnpack3(src.y)));
    dst2.y = hipPack(make_float4(hipUnpack1(src.z), hipUnpack3(src.z), hipUnpack1(src.w), hipUnpack3(src.w)));

    *((uint2 *)(&dst_image1[dst_idx])) = dst1;
    *((uint2 *)(&dst_image2[dst_idx])) = dst2;

}
void ConvertInterleavedUVToPlanarUV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image1, uint8_t *dst_image2, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes) {
    int32_t local_threads_x = 16, local_threads_y = 16;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = dst_height;

    ConvertInterleavedUVToPlanarUVKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                                    dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_width, dst_height, dst_image1, dst_image2,
                                    dst_image_stride_in_bytes, src_image1, src_image1_stride_in_bytes);

}

__global__ void ExtractYFromPackedYUYVKernel(uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint32_t dst_luma_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes,
    uint32_t dst_width_comp) {

    int32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x < dst_width_comp && y < dst_height) {
        uint32_t src_idx = y * src_image_stride_in_bytes + (x << 4);
        uint32_t dst_idx = y * dst_luma_stride_in_bytes + (x << 3);

        uint4 src = *((uint4 *)(&src_image[src_idx]));
        uint2 dst_y;
        dst_y.x = hipPack(make_float4(hipUnpack0(src.x), hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y)));
        dst_y.y = hipPack(make_float4(hipUnpack0(src.z), hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w)));

        *((uint2 *)(&destination_y[dst_idx])) = dst_y;
    }
}

void ExtractYFromPackedYUYV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint32_t dst_luma_stride_in_bytes, const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {
    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = dst_height;

    uint32_t dst_width_comp = (dst_width + 7) / 8;

    ExtractYFromPackedYUYVKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                                  dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_width, dst_height, destination_y,
                                  dst_luma_stride_in_bytes, src_image, src_image_stride_in_bytes, dst_width_comp);
}

__global__ void ConvertPackedYUYVToPlanarYUVKernel(uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint8_t *destination_u, uint8_t *destination_v, uint32_t dst_luma_stride_in_bytes, uint32_t dst_chroma_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes,
    uint32_t dst_width_comp) {

    int32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp && y < dst_height)) {
        uint32_t src_idx = y * src_image_stride_in_bytes + (x << 4);
        uint32_t dst_y_idx = y * dst_luma_stride_in_bytes + (x << 3);
        uint32_t dst_uv_idx = y * dst_chroma_stride_in_bytes + (x << 2);

        uint4 src = *((uint4 *)(&src_image[src_idx]));
        uint2 dst_y;
        uint32_t dst_u, dst_v;

        dst_y.x = hipPack(make_float4(hipUnpack0(src.x), hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y)));
        dst_y.y = hipPack(make_float4(hipUnpack0(src.z), hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w)));
        dst_u = hipPack(make_float4(hipUnpack1(src.x), hipUnpack1(src.y), hipUnpack1(src.z), hipUnpack1(src.w)));
        dst_v = hipPack(make_float4(hipUnpack3(src.x), hipUnpack3(src.y), hipUnpack3(src.z), hipUnpack3(src.w)));

        *((uint2 *)(&destination_y[dst_y_idx])) = dst_y;
        *((uint32_t *)(&destination_u[dst_uv_idx])) = dst_u;
        *((uint32_t *)(&destination_v[dst_uv_idx])) = dst_v;
    }
}

void ConvertPackedYUYVToPlanarYUV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint8_t *destination_u, uint8_t *destination_v, uint32_t dst_luma_stride_in_bytes, uint32_t dst_chroma_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {

    int32_t local_threads_x = 16;
    int32_t local_threads_y = 4;
    int32_t global_threads_x = (dst_width + 7) >> 3;
    int32_t global_threads_y = dst_height;
    uint32_t dst_width_comp = (dst_width + 7) / 8;

    ConvertPackedYUYVToPlanarYUVKernel<<<dim3(ceil(static_cast<float>(global_threads_x) / local_threads_x), ceil(static_cast<float>(global_threads_y) / local_threads_y)),
                                    dim3(local_threads_x, local_threads_y), 0, stream>>>(dst_width, dst_height, destination_y, destination_u,
                                    destination_v, dst_luma_stride_in_bytes, dst_chroma_stride_in_bytes, src_image, src_image_stride_in_bytes, dst_width_comp);
}