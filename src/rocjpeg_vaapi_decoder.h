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
#include <unordered_map>
#include <memory>
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

typedef struct {
    uint32_t num_jpeg_cores;
    bool can_convert_to_rgb;
    bool can_roi_decode;
} VcnJpegSpec;

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

struct RocJpegVappiMemPoolEntry {
    uint32_t image_width;
    uint32_t image_height;
    VASurfaceID va_surface_id;
    VAContextID va_context_id;
    HipInteropDeviceMem hip_interop;
};

class RocJpegVappiMemoryPool {
    public:
        RocJpegVappiMemoryPool();
        void ReleaseResources();
        void SetPoolSize(int32_t max_pool_size);
        void SetVaapiDisplay(const VADisplay& va_display);
        bool FindSurfaceId(VASurfaceID surface_id);
        RocJpegVappiMemPoolEntry GetEntry(uint32_t surface_format, uint32_t image_width, uint32_t image_height);
        RocJpegStatus AddPoolEntry(uint32_t surface_format, const RocJpegVappiMemPoolEntry& pool_entry);
        RocJpegStatus DeleteSurfaceId(VASurfaceID surface_id);
        RocJpegStatus GetHipInteropMem(VASurfaceID surface_id, HipInteropDeviceMem& hip_interop);
    private:
        VADisplay va_display_;
        std::unordered_map<uint32_t, std::vector<RocJpegVappiMemPoolEntry>> mem_pool_;
};

class RocJpegVappiDecoder {
public:
    RocJpegVappiDecoder(int device_id = 0);
    ~RocJpegVappiDecoder();
    RocJpegStatus InitializeDecoder(std::string device_name, std::string gcn_arch_name, int device_id);
    RocJpegStatus SubmitDecode(const JpegStreamParameters *jpeg_stream_params, uint32_t &surface_id, RocJpegOutputFormat output_format);
    RocJpegStatus SyncSurface(VASurfaceID surface_id);
    RocJpegStatus GetHipInteropMem(VASurfaceID surface_id, HipInteropDeviceMem& hip_interop);
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
    std::unordered_map<std::string, VcnJpegSpec> vcn_jpeg_spec_;
    std::unique_ptr<RocJpegVappiMemoryPool> vaapi_mem_pool_;
    VcnJpegSpec current_vcn_jpeg_spec_;
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
    void GetDrmNodeOffset(std::string device_name, uint8_t device_id, std::vector<int>& visible_devices,
                                    std::vector<ComputePartition> &current_compute_partitions,
                                    int &offset);
};

#endif // ROC_JPEG_VAAPI_DECODER_H_