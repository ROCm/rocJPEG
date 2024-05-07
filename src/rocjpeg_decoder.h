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
#include <mutex>
#include <queue>
#include "../api/rocjpeg.h"
#include "rocjpeg_parser.h"
#include "rocjpeg_commons.h"
#include "rocjpeg_vaapi_decoder.h"
#include "rocjpeg_hip_kernels.h"

class ROCJpegDecoder {
    public:
       ROCJpegDecoder(RocJpegBackend backend = ROCJPEG_BACKEND_HARDWARE, int device_id = 0);
       ~ROCJpegDecoder();
       RocJpegStatus InitializeDecoder();
       RocJpegStatus GetImageInfo(const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights);
       RocJpegStatus Decode(const uint8_t *data, size_t length, const RocJpegDecodeParams *decode_params, RocJpegImage *destination);
    private:
       RocJpegStatus InitHIP(int device_id);
       RocJpegStatus GetChromaHeight(uint32_t surface_format, uint16_t picture_height, uint16_t &chroma_height);
       RocJpegStatus CopyChannel(HipInteropDeviceMem& hip_interop, uint16_t channel_height, uint8_t channel_index, RocJpegImage *destination);
       RocJpegStatus ColorConvertToRGB(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination);
       RocJpegStatus ColorConvertToRGBPlanar(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination);
       RocJpegStatus GetPlanarYUVOutputFormat(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, uint16_t chroma_height, RocJpegImage *destination);
       RocJpegStatus GetYOutputFormat(HipInteropDeviceMem& hip_interop, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination);
       int num_devices_;
       int device_id_;
       hipDeviceProp_t hip_dev_prop_;
       hipStream_t hip_stream_;
       std::mutex mutex_;
       JpegParser jpeg_parser_;
       RocJpegBackend backend_;
       RocJpegVappiDecoder jpeg_vaapi_decoder_;
};

#endif //ROC_JPEG_DECODER_H_