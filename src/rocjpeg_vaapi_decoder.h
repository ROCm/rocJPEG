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

#ifndef ROC_JPEG_VAAPI_DECODER_H_
#define ROC_JPEG_VAAPI_DECODER_H_

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
#include "rocjpeg_commons.h"
#include "rocjpeg_parser.h"
#include "../api/rocjpeg.h"

/*Note: va.h doesn't have VA_FOURCC_YUYV defined but vaExportSurfaceHandle returns 0x56595559 for packed YUYV for YUV 4:2:2*/
#define ROCJPEG_FOURCC_YUYV 0x56595559

typedef enum {
    kSpx = 0, // Single Partition Accelerator
    kDpx = 1, // Dual Partition Accelerator
    kTpx = 2, // Triple Partition Accelerator
    kQpx = 3, // Quad Partition Accelerator
    kCpx = 4, // Core Partition Accelerator
} ComputePartition;

class RocJpegVappiDecoder {
public:
    RocJpegVappiDecoder(int device_id = 0);
    ~RocJpegVappiDecoder();
    RocJpegStatus InitializeDecoder(std::string device_name, std::string gcn_arch_name, int device_id);
    RocJpegStatus SubmitDecode(const JpegStreamParameters *jpeg_stream_params, uint32_t &surface_id);
    RocJpegStatus ExportSurface(VASurfaceID surface_id, VADRMPRIMESurfaceDescriptor &va_drm_prime_surface_desc);
    RocJpegStatus SyncSurface(VASurfaceID surface_id);
    RocJpegStatus ReleaseSurface(VASurfaceID surface_id);
private:
    int device_id_;
    int drm_fd_;
    uint32_t min_picture_width_;
    uint32_t min_picture_height_;
    uint32_t max_picture_width_;
    uint32_t max_picture_height_;
    VADisplay va_display_;
    std::vector<VAConfigAttrib> va_config_attrib_;
    VAConfigID va_config_id_;
    VAProfile va_profile_;
    VAContextID va_context_id_;
    std::vector<VASurfaceID> va_surface_ids_;
    VABufferID va_picture_parameter_buf_id_;
    VABufferID va_quantization_matrix_buf_id_;
    VABufferID va_huffmantable_buf_id_;
    VABufferID va_slice_param_buf_id_;
    VABufferID va_slice_data_buf_id_;
    RocJpegStatus InitVAAPI(std::string drm_node);
    RocJpegStatus CreateDecoderConfig();
    RocJpegStatus DestroyDataBuffers();
    void GetVisibleDevices(std::vector<int>& visible_devices);
    void GetCurrentComputePartition(std::vector<ComputePartition> &currnet_compute_partitions);
    void GetNumRenderCardsPerDevice(std::string device_name, uint8_t device_id, std::vector<int>& visible_devices,
                                    std::vector<ComputePartition> &current_compute_partitions,
                                    int &num_render_cards_per_socket, int &offset);
};

#endif // ROC_JPEG_VAAPI_DECODER_H_