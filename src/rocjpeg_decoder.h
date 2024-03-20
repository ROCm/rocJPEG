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

#include <unistd.h>
#include <vector>
#include <hip/hip_runtime.h>
#include <mutex>
#include <queue>
#include "../api/rocjpeg.h"
#include "rocjpeg_parser.h"
#include "commons.h"
#include "rocjpeg_vaapi_decoder.h"

extern "C" {
void HipExecColorConvertYUV444ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset);

void HipExecColorConvertYUYVToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

void HipExecColorConvertNV12ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes);

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

void HipExecChannelExtractUYVYToY(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_y, uint32_t dst_luma_stride_in_bytes, const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

void HipExecChannelExtractYUYVtoUV(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *destination_u, uint8_t *destination_v, uint32_t dst_chroma_stride_in_bytes,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

}


struct HipInteropDeviceMem {
    hipExternalMemory_t hip_ext_mem; // Interface to the vaapi-hip interop
    uint8_t* hip_mapped_device_mem; // Mapped device memory for the YUV plane
    uint32_t surface_format; // Pixel format fourcc of the whole surface
    uint32_t width; // Width of the surface in pixels.
    uint32_t height; // Height of the surface in pixels.
    uint32_t offset[3]; // Offset of each plane
    uint32_t pitch[3]; // Pitch of each plane
    uint32_t num_layers; // Number of layers making up the surface
};

/// @brief
class ROCJpegDecoder {
    public:
       ROCJpegDecoder(RocJpegBackend backend = ROCJPEG_BACKEND_HARDWARE, int device_id = 0);
       ~ROCJpegDecoder();
       RocJpegStatus InitializeDecoder();
       RocJpegStatus GetImageInfo(const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights);
       RocJpegStatus Decode(const uint8_t *data, size_t length, RocJpegOutputFormat output_format, RocJpegImage *destination);
    private:
       RocJpegStatus InitHIP(int device_id);
       RocJpegStatus ConvertYUVtoRGB(const void *yuv_dev_mem, uint32_t width, uint32_t height, uint32_t yuv_image_stride, RocJpegChromaSubsampling subsampling,
            void *rgb_dev_mem, const size_t rgb_image_stride);
       RocJpegStatus GetHipInteropMem(VADRMPRIMESurfaceDescriptor &va_drm_prime_surface_desc);
       RocJpegStatus ReleaseHipInteropMem();
       RocJpegStatus GetChromaHeight(uint16_t picture_height, uint16_t &chroma_height);
       RocJpegStatus CopyLuma(RocJpegImage *destination, uint16_t picture_height);
       RocJpegStatus CopyChroma(RocJpegImage *destination, uint16_t chroma_height);
       int num_devices_;
       int device_id_;
       hipDeviceProp_t hip_dev_prop_;
       hipStream_t hip_stream_;
       std::mutex mutex_;
       JpegParser jpeg_parser_;
       RocJpegBackend backend_;
       RocJpegVappiDecoder jpeg_vaapi_decoder_;
       HipInteropDeviceMem hip_interop_;
};