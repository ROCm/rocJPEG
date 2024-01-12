/*
Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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
    int HipExec_ColorConvert_NV12_to_RGB(hipStream_t stream, uint32_t dstWidth, uint32_t dstHeight,
        uint8_t *pHipDstImage, uint32_t dstImageStrideInBytes,
        const uint8_t *pHipSrcLumaImage, uint32_t srcLumaImageStrideInBytes,
        const uint8_t *pHipSrcChromaImage, uint32_t srcChromaImageStrideInBytes);

    int HipExec_ColorConvert_YUV444_to_RGB(hipStream_t stream, uint32_t dstWidth, uint32_t dstHeight,
        uint8_t *pHipDstImage, uint32_t dstImageStrideInBytes, const uint8_t *pHipSrcYUVImage,
        uint32_t srcYUVImageStrideInBytes, uint32_t srcUImageOffset);

    int HipExec_ScaleImage_NV12_Nearest(hipStream_t stream, uint32_t scaledYWidth, uint32_t scaledYHeight,
        uint8_t *pHipScaledYImage, uint32_t scaledYImageStrideInBytes, uint32_t srcYWidth, uint32_t srcYHeight,
        const uint8_t *pHipSrcYImage, uint32_t srcYImageStrideInBytes, uint8_t *pHipScaledUImage, uint8_t *pHipScaledVImage,
        const uint8_t *pHipSrcUImage, const uint8_t *pHipSrcVImage);

    int HipExec_ChannelExtract_U8U8_U16(hipStream_t stream, uint32_t dstWidth, uint32_t dstHeight,
        uint8_t *pHipDstImage1, uint8_t *pHipDstImage2, uint32_t dstImageStrideInBytes,
        const uint8_t *pHipSrcImage1, uint32_t srcImage1StrideInBytes);

    int HipExec_ChannelCombine_U16_U8U8(hipStream_t stream, uint32_t dstWidth, uint32_t dstHeight,
        uint8_t *pHipDstImage, uint32_t dstImageStrideInBytes,
        const uint8_t *pHipSrcImage1, uint32_t srcImage1StrideInBytes,
        const uint8_t *pHipSrcImage2, uint32_t srcImage2StrideInBytes);

    int HipExec_ScaleImage_U8_U8_Nearest(hipStream_t stream, uint32_t dstWidth, uint32_t dstHeight,
        uint8_t *pHipDstImage, uint32_t dstImageStrideInBytes, uint32_t srcWidth, uint32_t srcHeight,
        const uint8_t *pHipSrcImage, uint32_t srcImageStrideInBytes);

    int HipExec_ScaleImage_YUV444_Nearest(hipStream_t stream, uint32_t dstWidth, uint32_t dstHeight,
        uint8_t *pHipDstYUVImage, uint32_t dstImageStrideInBytes, uint32_t dstUImageOffset,
        uint32_t srcWidth, uint32_t srcHeight, const uint8_t *pHipSrcYUVImage,
        uint32_t srcImageStrideInBytes, uint32_t srcUImageOffset);
}

/// @brief
class ROCJpegDecode {
    public:
       ROCJpegDecode(RocJpegBackend backend = ROCJPEG_BACKEND_HARDWARE, int device_id = 0);
       ~ROCJpegDecode();
       RocJpegStatus InitializeDecoder();
       RocJpegStatus GetImageInfo(const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *width, uint32_t *height);
       RocJpegStatus Decode(const uint8_t *data, size_t length, RocJpegOutputFormat output_format, RocJpegImage *destination, hipStream_t stream);
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
       void *yuv_dev_mem_;
       std::mutex mutex_;
       JpegParser jpeg_parser_;
       RocJpegBackend backend_;
       RocJpegVappiDecoder jpeg_vaapi_decoder_;
};