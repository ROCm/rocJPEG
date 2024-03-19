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

#include <hip/hip_runtime.h>

extern "C" {
void HipExecColorConvertNV12ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes);

void HipExecColorConvertYUV444ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset);

void HipExecScaleImageNV12Nearest(hipStream_t stream, uint32_t scaled_y_width, uint32_t scaled_y_height,
    uint8_t *scaled_y_image, uint32_t scaled_y_image_stride_in_bytes, uint32_t src_y_width, uint32_t src_y_height,
    const uint8_t *src_y_image, uint32_t src_y_image_stride_in_bytes, uint8_t *scaled_u_image, uint8_t *scaled_v_image,
    const uint8_t *src_u_image, const uint8_t *src_v_image);

void HipExecChannelExtractU16ToU8U8(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image1, uint8_t *dst_image2, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes);

void HipExecChannelCombineU16U8U8(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes,
    const uint8_t *src_image2, uint32_t src_image2_stride_in_bytes);

void HipExecScaleImageU8U8Nearest(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, uint32_t src_width, uint32_t src_height,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

void HipExecScaleImageYUV444Nearest(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_yuv_image, uint32_t dst_image_stride_in_bytes, uint32_t dst_u_image_offset,
    uint32_t src_width, uint32_t src_height, const uint8_t *src_yuv_image,
    uint32_t src_image_stride_in_bytes, uint32_t src_u_image_offset);
}

typedef struct d_uint6 {
  uint data[6];
} d_uint6;

__device__ __forceinline__ uint hipPack(float4 src) {
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

__device__ __forceinline__ float hipUnpack0(uint src) {
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float hipUnpack1(uint src) {
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float hipUnpack2(uint src) {
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float hipUnpack3(uint src) {
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 hipUnpack(uint src) {
    return make_float4(hipUnpack0(src), hipUnpack1(src), hipUnpack2(src), hipUnpack3(src));
}

__global__ void __attribute__((visibility("default")))
HipColorConvertNV12ToRGB(uint dst_width, uint dst_height,
    uchar *dst_image, uint dst_image_stride_in_bytes, uint dst_image_stride_in_bytes_comp,
    const uchar *src_luma_image, uint src_luma_image_stride_in_bytes,
    const uchar *src_chroma_image, uint src_chroma_image_stride_in_bytes,
    uint dst_width_comp, uint dst_height_comp, uint src_luma_image_stride_in_bytes_comp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint src_y0_idx = y * src_luma_image_stride_in_bytes_comp + (x << 3);
        uint src_y1_idx = src_y0_idx + src_luma_image_stride_in_bytes;
        uint src_uv_idx = y * src_chroma_image_stride_in_bytes + (x << 3);
        uint2 y0 = *((uint2 *)(&src_luma_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_luma_image[src_y1_idx]));
        uint2 uv = *((uint2 *)(&src_chroma_image[src_uv_idx]));

        uint rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

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
        d_uint6 rgb0, rgb1;

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

        *((d_uint6 *)(&dst_image[rgb0_idx])) = rgb0;
        *((d_uint6 *)(&dst_image[rgb1_idx])) = rgb1;
    }
}

void HipExecColorConvertNV12ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dst_width + 7) >> 3;
    int globalThreads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_luma_image_stride_in_bytes_comp = src_luma_image_stride_in_bytes * 2;

    hipLaunchKernelGGL(HipColorConvertNV12ToRGB, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dst_width, dst_height, (uchar *)dst_image, dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp,
                        (const uchar *)src_luma_image, src_luma_image_stride_in_bytes, (const uchar *)src_chroma_image, src_chroma_image_stride_in_bytes,
                        dst_width_comp, dst_height_comp, src_luma_image_stride_in_bytes_comp);
}

__global__ void __attribute__((visibility("default")))
HipColorConvertYUV444ToRGB(uint dst_width, uint dst_height,
    uchar *dst_image, uint dst_image_stride_in_bytes, uint dst_image_stride_in_bytes_comp,
    const uchar *src_y_image, const uchar *src_u_image, const uchar *src_v_image, uint src_yuv_image_stride_in_bytes,
    uint dst_width_comp, uint dst_height_comp, uint src_yuv_image_stride_in_bytes_comp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dst_width_comp) && (y < dst_height_comp)) {
        uint src_y0_idx = y * src_yuv_image_stride_in_bytes_comp + (x << 3);
        uint src_y1_idx = src_y0_idx + src_yuv_image_stride_in_bytes;


        uint2 y0 = *((uint2 *)(&src_y_image[src_y0_idx]));
        uint2 y1 = *((uint2 *)(&src_y_image[src_y1_idx]));

        uint2 u0 = *((uint2 *)(&src_u_image[src_y0_idx]));
        uint2 u1 = *((uint2 *)(&src_u_image[src_y1_idx]));

        uint2 v0 = *((uint2 *)(&src_v_image[src_y0_idx]));
        uint2 v1 = *((uint2 *)(&src_v_image[src_y1_idx]));

        uint rgb0_idx = y * dst_image_stride_in_bytes_comp + (x * 24);
        uint rgb1_idx = rgb0_idx + dst_image_stride_in_bytes;

        float2 cr = make_float2( 0.0000f,  1.5748f);
        float2 cg = make_float2(-0.1873f, -0.4681f);
        float2 cb = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        d_uint6 rgb0, rgb1;
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

        *((d_uint6 *)(&dst_image[rgb0_idx])) = rgb0;
        *((d_uint6 *)(&dst_image[rgb1_idx])) = rgb1;
    }
}

void HipExecColorConvertYUV444ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset) {

    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dst_width + 7) >> 3;
    int globalThreads_y = (dst_height + 1) >> 1;

    uint32_t dst_width_comp = (dst_width + 7) / 8;
    uint32_t dst_height_comp = (dst_height + 1) / 2;
    uint32_t dst_image_stride_in_bytes_comp = dst_image_stride_in_bytes * 2;
    uint32_t src_yuv_image_stride_in_bytes_comp = src_yuv_image_stride_in_bytes * 2;

    hipLaunchKernelGGL(HipColorConvertYUV444ToRGB, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dst_width, dst_height, (uchar *)dst_image, dst_image_stride_in_bytes, dst_image_stride_in_bytes_comp,
                        (const uchar *)src_yuv_image,
                        (const uchar *)src_yuv_image + src_u_image_offset,
                        (const uchar *)src_yuv_image + (src_u_image_offset * 2),
                        src_yuv_image_stride_in_bytes,
                        dst_width_comp, dst_height_comp, src_yuv_image_stride_in_bytes_comp);

}

__global__ void __attribute__((visibility("default")))
HipScaleImageNV12Nearest(uint scaled_y_width, uint scaled_y_height, uchar *scaled_y_image, uint scaled_y_image_stride_in_bytes,
    const uchar *src_y_image, uint src_y_image_stride_in_bytes, float xscale_y, float yscale_y, float xoffset_y, float yoffset_y,
    uint scaled_uv_width, uint scaled_uv_height, uchar *scaled_u_image, uchar *scaled_v_image, uint scaled_uv_image_stride_in_bytes,
    const uchar *src_u_image, const uchar *src_v_image, uint src_uv_image_stride_in_bytes,
    float x_scale_uv, float y_scale_uv, float x_offset_uv, float y_offset_uv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= scaled_y_width || y >= scaled_y_height) {
        return;
    }

    uint scaled_y_idx = y * scaled_y_image_stride_in_bytes + x;

    float4 scale_info = make_float4(xscale_y, yscale_y, xoffset_y, yoffset_y);

    uint2 scaled_y_dst;
    src_y_image += src_y_image_stride_in_bytes * (uint)fmaf((float)y, scale_info.y, scale_info.w);
    float fx = fmaf((float)x, scale_info.x, scale_info.z);

    scaled_y_dst.x  = src_y_image[(int)fx];
    fx += scale_info.x;
    scaled_y_dst.x |= src_y_image[(int)fx] << 8;
    fx += scale_info.x;
    scaled_y_dst.x |= src_y_image[(int)fx] << 16;
    fx += scale_info.x;
    scaled_y_dst.x |= src_y_image[(int)fx] << 24;

    fx += scale_info.x;

    scaled_y_dst.y  = src_y_image[(int)fx];
    fx += scale_info.x;
    scaled_y_dst.y |= src_y_image[(int)fx] << 8;
    fx += scale_info.x;
    scaled_y_dst.y |= src_y_image[(int)fx] << 16;
    fx += scale_info.x;
    scaled_y_dst.y |= src_y_image[(int)fx] << 24;

    *((uint2 *)(&scaled_y_image[scaled_y_idx])) = scaled_y_dst;

    //scale the U and V components here
    if (x >= scaled_uv_width || y >= scaled_uv_height) {
        return;
    }

    uint scaled_uv_idx = y * scaled_uv_image_stride_in_bytes + x;

    scale_info = make_float4(x_scale_uv, y_scale_uv, x_offset_uv, y_offset_uv);

    uint2 scaled_u_dst, scaled_v_dst;
    src_u_image += src_uv_image_stride_in_bytes * (uint)fmaf((float)y, scale_info.y, scale_info.w);
    src_v_image += src_uv_image_stride_in_bytes * (uint)fmaf((float)y, scale_info.y, scale_info.w);
    fx = fmaf((float)x, scale_info.x, scale_info.z);

    scaled_u_dst.x  = src_u_image[(int)fx];
    scaled_v_dst.x  = src_v_image[(int)fx];
    fx += scale_info.x;
    scaled_u_dst.x |= src_u_image[(int)fx] << 8;
    scaled_v_dst.x |= src_v_image[(int)fx] << 8;
    fx += scale_info.x;
    scaled_u_dst.x |= src_u_image[(int)fx] << 16;
    scaled_v_dst.x |= src_v_image[(int)fx] << 16;
    fx += scale_info.x;
    scaled_u_dst.x |= src_u_image[(int)fx] << 24;
    scaled_v_dst.x |= src_v_image[(int)fx] << 24;

    fx += scale_info.x;

    scaled_u_dst.y  = src_u_image[(int)fx];
    scaled_v_dst.y  = src_v_image[(int)fx];
    fx += scale_info.x;
    scaled_u_dst.y |= src_u_image[(int)fx] << 8;
    scaled_v_dst.y |= src_v_image[(int)fx] << 8;
    fx += scale_info.x;
    scaled_u_dst.y |= src_u_image[(int)fx] << 16;
    scaled_v_dst.y |= src_v_image[(int)fx] << 16;
    fx += scale_info.x;
    scaled_u_dst.y |= src_u_image[(int)fx] << 24;
    scaled_v_dst.y |= src_v_image[(int)fx] << 24;

    *((uint2 *)(&scaled_u_image[scaled_uv_idx])) = scaled_u_dst;
    *((uint2 *)(&scaled_v_image[scaled_uv_idx])) = scaled_v_dst;

}

void HipExecScaleImageNV12Nearest(hipStream_t stream, uint32_t scaled_y_width, uint32_t scaled_y_height,
    uint8_t *scaled_y_image, uint32_t scaled_y_image_stride_in_bytes, uint32_t src_y_width, uint32_t src_y_height,
    const uint8_t *src_y_image, uint32_t src_y_image_stride_in_bytes, uint8_t *scaled_u_image, uint8_t *scaled_v_image,
    const uint8_t *src_u_image, const uint8_t *src_v_image) {

    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (scaled_y_width + 7) >> 3;
    int globalThreads_y = scaled_y_height;

    uint32_t src_uv_width = src_y_width / 2;
    uint32_t src_uv_height = src_y_height / 2;
    uint32_t src_uv_image_stride_in_bytes = src_y_image_stride_in_bytes / 2;

    uint32_t scaled_uv_width = scaled_y_width / 2;
    uint32_t scaled_uv_height = scaled_y_height / 2;
    uint32_t scaled_uv_image_stride_in_bytes = scaled_y_image_stride_in_bytes / 2;

    float xscale_y = (float)((double)src_y_width / (double)scaled_y_width);
    float yscale_y = (float)((double)src_y_height / (double)scaled_y_height);
    float xoffset_y = (float)((double)src_y_width / (double)scaled_y_width * 0.5);
    float yoffset_y = (float)((double)src_y_height / (double)scaled_y_height * 0.5);

    float x_scale_uv = (float)((double)src_uv_width / (double)scaled_uv_width);
    float y_scale_uv = (float)((double)src_uv_height / (double)scaled_uv_height);
    float x_offset_uv = (float)((double)src_uv_width / (double)scaled_uv_width * 0.5);
    float y_offset_uv = (float)((double)src_uv_height / (double)scaled_uv_height * 0.5);

    hipLaunchKernelGGL(HipScaleImageNV12Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, scaled_y_width, scaled_y_height, (uchar *)scaled_y_image , scaled_y_image_stride_in_bytes,
                        (const uchar *)src_y_image, src_y_image_stride_in_bytes, xscale_y, yscale_y, xoffset_y, yoffset_y,
                        scaled_uv_width, scaled_uv_height, (uchar *)scaled_u_image, (uchar *)scaled_v_image, scaled_uv_image_stride_in_bytes,
                        (const uchar *)src_u_image, (const uchar *)src_v_image, src_uv_image_stride_in_bytes, x_scale_uv, y_scale_uv, x_offset_uv, y_offset_uv);
}

__global__ void __attribute__((visibility("default")))
HipChannelExtractU16ToU8U8(uint dst_width, uint dst_height,
    uchar *dst_image1, uchar *dst_image2, uint dst_image_stride_in_bytes,
    const uchar *src_image, uint src_image_stride_in_bytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dst_width || y >= dst_height) {
        return;
    }

    unsigned int src_idx = y * src_image_stride_in_bytes + x + x;
    unsigned int dst_idx = y * dst_image_stride_in_bytes + x;

    uint4 src = *((uint4 *)(&src_image[src_idx]));
    uint2 dst1, dst2;

    dst1.x = hipPack(make_float4(hipUnpack0(src.x), hipUnpack2(src.x), hipUnpack0(src.y), hipUnpack2(src.y)));
    dst1.y = hipPack(make_float4(hipUnpack0(src.z), hipUnpack2(src.z), hipUnpack0(src.w), hipUnpack2(src.w)));
    dst2.x = hipPack(make_float4(hipUnpack1(src.x), hipUnpack3(src.x), hipUnpack1(src.y), hipUnpack3(src.y)));
    dst2.y = hipPack(make_float4(hipUnpack1(src.z), hipUnpack3(src.z), hipUnpack1(src.w), hipUnpack3(src.w)));

    *((uint2 *)(&dst_image1[dst_idx])) = dst1;
    *((uint2 *)(&dst_image2[dst_idx])) = dst2;

}
void HipExecChannelExtractU16ToU8U8(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image1, uint8_t *dst_image2, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dst_width + 7) >> 3;
    int globalThreads_y = dst_height;

    hipLaunchKernelGGL(HipChannelExtractU16ToU8U8, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dst_width, dst_height, (uchar *)dst_image1, (uchar *)dst_image2, dst_image_stride_in_bytes,
                        (const uchar *)src_image1, src_image1_stride_in_bytes);

}

__global__ void __attribute__((visibility("default")))
HipChannelCombineU8U8ToU16(uint dst_width, uint dst_height,
    uchar *dst_image, uint dst_image_stride_in_bytes,
    const uchar *src_image1, uint src_image1_stride_in_bytes,
    const uchar *src_image2, uint src_image2_stride_in_bytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dst_width || y >= dst_height) {
        return;
    }

    uint src1_idx = y * src_image1_stride_in_bytes + x;
    uint src2_idx = y * src_image2_stride_in_bytes + x;
    uint dst_idx =  y * dst_image_stride_in_bytes + x + x;

    uint2 src1 = *((uint2 *)(&src_image1[src1_idx]));
    uint2 src2 = *((uint2 *)(&src_image2[src2_idx]));
    uint4 dst;

    dst.x = hipPack(make_float4(hipUnpack0(src1.x), hipUnpack0(src2.x), hipUnpack1(src1.x), hipUnpack1(src2.x)));
    dst.y = hipPack(make_float4(hipUnpack2(src1.x), hipUnpack2(src2.x), hipUnpack3(src1.x), hipUnpack3(src2.x)));
    dst.z = hipPack(make_float4(hipUnpack0(src1.y), hipUnpack0(src2.y), hipUnpack1(src1.y), hipUnpack1(src2.y)));
    dst.w = hipPack(make_float4(hipUnpack2(src1.y), hipUnpack2(src2.y), hipUnpack3(src1.y), hipUnpack3(src2.y)));

    *((uint4 *)(&dst_image[dst_idx])) = dst;
}
void HipExecChannelCombineU16U8U8(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes,
    const uint8_t *src_image2, uint32_t src_image2_stride_in_bytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dst_width + 7) >> 3;
    int globalThreads_y = dst_height;

    hipLaunchKernelGGL(HipChannelCombineU8U8ToU16, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dst_width, dst_height, (uchar *)dst_image , dst_image_stride_in_bytes,
                        (const uchar *)src_image1, src_image1_stride_in_bytes, (const uchar *)src_image2, src_image2_stride_in_bytes);

}

__global__ void __attribute__((visibility("default")))
HipScaleImageU8U8Nearest(uint dst_width, uint dst_height,
    uchar *dst_image, uint dst_image_stride_in_bytes,
    const uchar *src_image, uint src_image_stride_in_bytes,
    float xscale, float yscale, float xoffset, float yoffset) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dst_width || y >= dst_height) {
        return;
    }

    uint dst_idx = y * dst_image_stride_in_bytes + x;

    float4 scale_info = make_float4(xscale, yscale, xoffset, yoffset);

    uint2 dst;
    src_image += src_image_stride_in_bytes * (uint)fmaf((float)y, scale_info.y, scale_info.w);
    float fx = fmaf((float)x, scale_info.x, scale_info.z);

    dst.x  = src_image[(int)fx];
    fx += scale_info.x;
    dst.x |= src_image[(int)fx] << 8;
    fx += scale_info.x;
    dst.x |= src_image[(int)fx] << 16;
    fx += scale_info.x;
    dst.x |= src_image[(int)fx] << 24;

    fx += scale_info.x;

    dst.y  = src_image[(int)fx];
    fx += scale_info.x;
    dst.y |= src_image[(int)fx] << 8;
    fx += scale_info.x;
    dst.y |= src_image[(int)fx] << 16;
    fx += scale_info.x;
    dst.y |= src_image[(int)fx] << 24;

    *((uint2 *)(&dst_image[dst_idx])) = dst;
}

void HipExecScaleImageU8U8Nearest(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    uint32_t src_width, uint32_t src_height,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dst_width + 7) >> 3;
    int globalThreads_y = dst_height;

    float xscale = (float)((double)src_width / (double)dst_width);
    float yscale = (float)((double)src_height / (double)dst_height);
    float xoffset = (float)((double)src_width / (double)dst_width * 0.5);
    float yoffset = (float)((double)src_height / (double)dst_height * 0.5);

    hipLaunchKernelGGL(HipScaleImageU8U8Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dst_width, dst_height, (uchar *)dst_image , dst_image_stride_in_bytes,
                        (const uchar *)src_image, src_image_stride_in_bytes,
                        xscale, yscale, xoffset, yoffset);

}

__global__ void __attribute__((visibility("default")))
HipScaleImageYUV444Nearest(uint dst_width, uint dst_height, uchar *dst_y_image, uchar *dst_u_image, uchar *dst_v_image, uint dst_image_stride_in_bytes,
    const uchar *src_y_image, const uchar *src_u_image, const uchar *src_v_image, uint src_image_stride_in_bytes,
    float xscale, float yscale, float xoffset, float yoffset) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dst_width || y >= dst_height) {
        return;
    }

    uint dst_idx = y * dst_image_stride_in_bytes + x;

    float4 scale_info = make_float4(xscale, yscale, xoffset, yoffset);

    uint2 y_dst, u_dst, v_dst;
    src_y_image += src_image_stride_in_bytes * (uint)fmaf((float)y, scale_info.y, scale_info.w);
    src_u_image += src_image_stride_in_bytes * (uint)fmaf((float)y, scale_info.y, scale_info.w);
    src_v_image += src_image_stride_in_bytes * (uint)fmaf((float)y, scale_info.y, scale_info.w);

    float fx = fmaf((float)x, scale_info.x, scale_info.z);

    y_dst.x  = src_y_image[(int)fx];
    u_dst.x  = src_u_image[(int)fx];
    v_dst.x  = src_v_image[(int)fx];
    fx += scale_info.x;
    y_dst.x |= src_y_image[(int)fx] << 8;
    u_dst.x |= src_u_image[(int)fx] << 8;
    v_dst.x |= src_v_image[(int)fx] << 8;
    fx += scale_info.x;
    y_dst.x |= src_y_image[(int)fx] << 16;
    u_dst.x |= src_u_image[(int)fx] << 16;
    v_dst.x |= src_v_image[(int)fx] << 16;
    fx += scale_info.x;
    y_dst.x |= src_y_image[(int)fx] << 24;
    u_dst.x |= src_u_image[(int)fx] << 24;
    v_dst.x |= src_v_image[(int)fx] << 24;
    fx += scale_info.x;

    y_dst.y  = src_y_image[(int)fx];
    u_dst.y  = src_u_image[(int)fx];
    v_dst.y  = src_v_image[(int)fx];
    fx += scale_info.x;
    y_dst.y |= src_y_image[(int)fx] << 8;
    u_dst.y |= src_u_image[(int)fx] << 8;
    v_dst.y |= src_v_image[(int)fx] << 8;
    fx += scale_info.x;
    y_dst.y |= src_y_image[(int)fx] << 16;
    u_dst.y |= src_u_image[(int)fx] << 16;
    v_dst.y |= src_v_image[(int)fx] << 16;
    fx += scale_info.x;
    y_dst.y |= src_y_image[(int)fx] << 24;
    u_dst.y |= src_u_image[(int)fx] << 24;
    v_dst.y |= src_v_image[(int)fx] << 24;

    *((uint2 *)(&dst_y_image[dst_idx])) = y_dst;
    *((uint2 *)(&dst_u_image[dst_idx])) = u_dst;
    *((uint2 *)(&dst_v_image[dst_idx])) = v_dst;
}

void HipExecScaleImageYUV444Nearest(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_yuv_image, uint32_t dst_image_stride_in_bytes, uint32_t dst_u_image_offset,
    uint32_t src_width, uint32_t src_height, const uint8_t *src_yuv_image, uint32_t src_image_stride_in_bytes, uint32_t src_u_image_offset) {

    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dst_width + 7) >> 3;
    int globalThreads_y = dst_height;

    float xscale = (float)((double)src_width / (double)dst_width);
    float yscale = (float)((double)src_height / (double)dst_height);
    float xoffset = (float)((double)src_width / (double)dst_width * 0.5);
    float yoffset = (float)((double)src_height / (double)dst_height * 0.5);

    hipLaunchKernelGGL(HipScaleImageYUV444Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dst_width, dst_height,
                        (uchar *)dst_yuv_image,
                        (uchar *)dst_yuv_image + dst_u_image_offset,
                        (uchar *)dst_yuv_image + (dst_u_image_offset * 2), dst_image_stride_in_bytes,
                        (const uchar *)src_yuv_image,
                        (const uchar *)src_yuv_image + src_u_image_offset,
                        (const uchar *)src_yuv_image + (src_u_image_offset * 2),
                        src_image_stride_in_bytes, xscale, yscale, xoffset, yoffset);

}