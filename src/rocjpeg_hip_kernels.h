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

#ifndef ROC_JPEG_HIP_KERNELS_H_
#define ROC_JPEG_HIP_KERNELS_H_

#pragma once

#include <hip/hip_runtime.h>

void ColorConvertYUV444ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset);

void ColorConvertYUYVToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

void ColorConvertNV12ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes);

void ColorConvertYUV400ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes);

void ColorConvertRGBAToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

void ColorConvertYUV444ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset);

void ColorConvertYUYVToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

void ColorConvertNV12ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes);

void ColorConvertYUV400ToRGBPlanar(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image_r, uint8_t *dst_image_g, uint8_t *dst_image_b, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes);

void ConvertInterleavedUVToPlanarUV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image1, uint8_t *dst_image2, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes);

void ExtractYFromPackedYUYV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint32_t dst_luma_stride_in_bytes, const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

void ConvertPackedYUYVToPlanarYUV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint8_t *destination_u, uint8_t *destination_v, uint32_t dst_luma_stride_in_bytes,
    uint32_t dst_chroma_stride_in_bytes, const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

typedef struct UINT6TYPE {
  uint data[6];
} DUINT6;

typedef struct UINT8TYPE {
  uint data[8];
} DUINT8;

#endif //ROC_JPEG_HIP_KERNELS_H_