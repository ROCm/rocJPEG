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
int HipExecColorConvertNV12ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_luma_image, uint32_t src_luma_image_stride_in_bytes,
    const uint8_t *src_chroma_image, uint32_t src_chroma_image_stride_in_bytes);

int HipExecColorConvertYUV444ToRGB(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, const uint8_t *src_yuv_image,
    uint32_t src_yuv_image_stride_in_bytes, uint32_t src_u_image_offset);

int HipExecScaleImageNV12Nearest(hipStream_t stream, uint32_t scaled_y_width, uint32_t scaled_y_height,
    uint8_t *scaled_y_image, uint32_t scaled_y_image_stride_in_bytes, uint32_t src_y_width, uint32_t src_y_height,
    const uint8_t *src_y_image, uint32_t src_y_image_stride_in_bytes, uint8_t *scaled_u_image, uint8_t *scaled_v_image,
    const uint8_t *src_u_image, const uint8_t *src_v_image);

int HipExecChannelExtractU16ToU8U8(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image1, uint8_t *dst_image2, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes);

int HipExecChannelCombineU16U8U8(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes,
    const uint8_t *src_image1, uint32_t src_image1_stride_in_bytes,
    const uint8_t *src_image2, uint32_t src_image2_stride_in_bytes);

int HipExecScaleImageU8U8Nearest(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_image, uint32_t dst_image_stride_in_bytes, uint32_t src_width, uint32_t src_height,
    const uint8_t *src_image, uint32_t src_image_stride_in_bytes);

int HipExecScaleImageYUV444Nearest(hipStream_t stream, uint32_t dst_width, uint32_t dst_height,
    uint8_t *dst_yuv_image, uint32_t dst_image_stride_in_bytes, uint32_t dst_u_image_offset,
    uint32_t src_width, uint32_t src_height, const uint8_t *src_yuv_image,
    uint32_t src_image_stride_in_bytes, uint32_t src_u_image_offset);
}

/// @brief
class ROCJpegDecode {
    public:
       ROCJpegDecode(RocJpegBackend backend = ROCJPEG_BACKEND_HARDWARE, int device_id = 0);
       ~ROCJpegDecode();
       RocJpegStatus InitializeDecoder();
       RocJpegStatus GetImageInfo(const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights);
       RocJpegStatus Decode(const uint8_t *data, size_t length, RocJpegOutputFormat output_format, RocJpegImage *destination);
    private:
       RocJpegStatus InitHIP(int device_id);
       bool ConvertYUVtoRGB(const void *yuv_dev_mem, const size_t yuv_image_size, uint32_t width, uint32_t height, uint32_t yuv_image_stride, RocJpegChromaSubsampling subsampling,
        void *rgb_dev_mem, const size_t rgb_dev_mem_size, const size_t rgb_image_stride);
       int num_devices_;
       int device_id_;
       hipDeviceProp_t hip_dev_prop_;
       hipStream_t hip_stream_;
       hipExternalMemory_t hip_ext_mem_;
       hipExternalMemoryHandleDesc external_mem_handle_desc_;
       hipExternalMemoryBufferDesc external_mem_buffer_desc_;
       uint8_t *yuv_dev_mem_;
       std::mutex mutex_;
       JpegParser jpeg_parser_;
       RocJpegBackend backend_;
       RocJpegVappiDecoder jpeg_vaapi_decoder_;
};