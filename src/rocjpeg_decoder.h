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

#ifndef ROC_JPEG_DECODER_H_
#define ROC_JPEG_DECODER_H_

#include <unistd.h>
#include <vector>
#include <hip/hip_runtime.h>
#include <mutex>
#include <queue>
#include "../api/rocjpeg.h"
#include "rocjpeg_parser.h"
#include "rocjpeg_commons.h"
#include "rocjpeg_vaapi_decoder.h"
#include "rocjpeg_hip_kernels.h"

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

class ROCJpegDecoder {
    public:
       ROCJpegDecoder(RocJpegBackend backend = ROCJPEG_BACKEND_HARDWARE, int device_id = 0);
       ~ROCJpegDecoder();
       RocJpegStatus InitializeDecoder();
       RocJpegStatus GetImageInfo(const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights);
       RocJpegStatus Decode(const uint8_t *data, size_t length, RocJpegOutputFormat output_format, RocJpegImage *destination);
    private:
       RocJpegStatus InitHIP(int device_id);
       RocJpegStatus GetHipInteropMem(VADRMPRIMESurfaceDescriptor &va_drm_prime_surface_desc);
       RocJpegStatus ReleaseHipInteropMem(VASurfaceID current_surface_id);
       RocJpegStatus GetChromaHeight(uint16_t picture_height, uint16_t &chroma_height);
       RocJpegStatus CopyLuma(RocJpegImage *destination, uint16_t picture_height);
       RocJpegStatus CopyChroma(RocJpegImage *destination, uint16_t chroma_height);
       RocJpegStatus ColorConvertToRGB(uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination);
       RocJpegStatus GetPlanarYUVOutputFormat(uint32_t picture_width, uint32_t picture_height, uint16_t chroma_height, RocJpegImage *destination);
       RocJpegStatus GetYOutputFormat(uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination);
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

#endif //ROC_JPEG_DECODER_H_