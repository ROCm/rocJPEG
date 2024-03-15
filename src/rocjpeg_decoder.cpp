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

#include "rocjpeg_decoder.h"

ROCJpegDecoder::ROCJpegDecoder(RocJpegBackend backend, int device_id) :
    num_devices_{0}, device_id_ {device_id}, hip_stream_ {0}, external_mem_handle_desc_{{}}, external_mem_buffer_desc_{{}},
    yuv_dev_mem_{nullptr}, backend_{backend} {}

ROCJpegDecoder::~ROCJpegDecoder() {
    if (hip_stream_) {
        hipError_t hip_status = hipStreamDestroy(hip_stream_);
    }
}

RocJpegStatus ROCJpegDecoder::InitHIP(int device_id) {
    hipError_t hip_status = hipSuccess;
    CHECK_HIP(hipGetDeviceCount(&num_devices_));
    if (num_devices_ < 1) {
        ERR("ERROR: Failed to find any GPU!");
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    if (device_id >= num_devices_) {
        ERR("ERROR: the requested device_id is not found!");
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    CHECK_HIP(hipSetDevice(device_id));
    CHECK_HIP(hipGetDeviceProperties(&hip_dev_prop_, device_id));
    CHECK_HIP(hipStreamCreate(&hip_stream_));
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::InitializeDecoder() {
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;
    rocjpeg_status = InitHIP(device_id_);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        ERR("ERROR: Failed to initilize the HIP!" + TOSTR(rocjpeg_status));
        return rocjpeg_status;
    }
    if (backend_ == ROCJPEG_BACKEND_HARDWARE) {
        rocjpeg_status = jpeg_vaapi_decoder_.InitializeDecoder(hip_dev_prop_.gcnArchName);
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            ERR("ERROR: Failed to initilize the VAAPI JPEG decoder!" + TOSTR(rocjpeg_status));
            return rocjpeg_status;
        }
    } else if (backend_ == ROCJPEG_BACKEND_HYBRID) {
        return ROCJPEG_STATUS_NOT_IMPLEMENTED;
    }
    return rocjpeg_status;
}

RocJpegStatus ROCJpegDecoder::Decode(const uint8_t *data, size_t length, RocJpegOutputFormat output_format, RocJpegImage *destination) {
    std::lock_guard<std::mutex> lock(mutex_);
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;

    if (!jpeg_parser_.ParseJpegStream(data, length)) {
        ERR("ERROR: Failed to parse the jpeg stream!");
        return ROCJPEG_STATUS_BAD_JPEG;
    }

    const JpegStreamParameters *jpeg_stream_params = jpeg_parser_.GetJpegStreamParameters();
    VASurfaceID current_surface_id;
    rocjpeg_status = jpeg_vaapi_decoder_.SubmitDecode(jpeg_stream_params, current_surface_id);
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
        return rocjpeg_status;
    }

    if (destination != nullptr) {
        VADRMPRIMESurfaceDescriptor va_drm_prime_surface_desc = {};
        rocjpeg_status = jpeg_vaapi_decoder_.SyncSurface(current_surface_id);
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            return rocjpeg_status;
        }

        rocjpeg_status = jpeg_vaapi_decoder_.ExportSurface(current_surface_id, va_drm_prime_surface_desc);
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            return rocjpeg_status;
        }

        // import the decoded surface (DRM-PRIME FDs) into the HIP
        external_mem_handle_desc_.type = hipExternalMemoryHandleTypeOpaqueFd;
        external_mem_handle_desc_.handle.fd = va_drm_prime_surface_desc.objects[0].fd;
        external_mem_handle_desc_.size = va_drm_prime_surface_desc.objects[0].size;

        CHECK_HIP(hipImportExternalMemory(&hip_ext_mem_, &external_mem_handle_desc_));

        external_mem_buffer_desc_.size = va_drm_prime_surface_desc.objects[0].size;
        CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&yuv_dev_mem_, hip_ext_mem_, &external_mem_buffer_desc_));

        uint32_t chroma_height = 0;
        switch (va_drm_prime_surface_desc.fourcc) {
            case VA_FOURCC_NV12: /*NV12: two-plane 8-bit YUV 4:2:0*/
                chroma_height = jpeg_stream_params->picture_parameter_buffer.picture_height >> 1;
                break;
            case VA_FOURCC_444P: /*444P: three-plane 8-bit YUV 4:4:4*/
                chroma_height = jpeg_stream_params->picture_parameter_buffer.picture_height;
                break;
            case VA_FOURCC_Y800: /*Y800: one-plane 8-bit greyscale YUV 4:0:0*/
                chroma_height = 0;
                break;
            case 0x56595559: /*YUYV: one-plane packed 8-bit YUV 4:2:2. Four bytes per pair of pixels: Y, U, Y, V*/
                /*Note: va.h doesn't have VA_FOURCC_YUYV defined but vaExportSurfaceHandle returns 0x56595559 for packed YUYV for YUV 4:2:2*/
                chroma_height = jpeg_stream_params->picture_parameter_buffer.picture_height >> 1;
                break;
            default:
                return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
        }
        /*switch (output_format) {
            case ROCJPEG_OUTPUT_UNCHANGED:
                // just copy the interop mempry directly to the destination buffrs
                break;
            case ROCJPEG_OUTPUT_YUV:
                // Need to separate the UV channels in the case of NV12 or packed YUYV and then copy into three separate channels
                break;
            case ROCJPEG_OUTPUT_Y:
                // just copy the Y to the first-channel of the destination, for YUYV first extreact the Y then copy
                break;
            case ROCJPEG_OUTPUT_RGBI:
                    // Need to do color conversion and then copy to the first channel of the destination
                break;
            default:
                break;
        }*/
        // this is for ROCJPEG_OUTPUT_UNCHANGED
        // copy Y (luma) channel0
        if (va_drm_prime_surface_desc.layers[0].pitch[0] != 0 && destination->pitch[0] != 0 && destination->channel[0] != nullptr) {
            if (destination->pitch[0] == va_drm_prime_surface_desc.layers[0].pitch[0]) {
                uint32_t luma_size = destination->pitch[0] * jpeg_stream_params->picture_parameter_buffer.picture_height;
                CHECK_HIP(hipMemcpyDtoDAsync(destination->channel[0], yuv_dev_mem_, luma_size, hip_stream_));
            } else {
                CHECK_HIP(hipMemcpy2DAsync(destination->channel[0], destination->pitch[0], yuv_dev_mem_, va_drm_prime_surface_desc.layers[0].pitch[0],
                    destination->pitch[0], jpeg_stream_params->picture_parameter_buffer.picture_height, hipMemcpyDeviceToDevice, hip_stream_));
            }
        }
        // copy channel1
        if (va_drm_prime_surface_desc.layers[1].pitch[0] != 0 && destination->pitch[1] != 0 && destination->channel[1] != nullptr) {
            uint32_t chroma_size = destination->pitch[1] * chroma_height;
            uint8_t *layer1_mem = yuv_dev_mem_ + va_drm_prime_surface_desc.layers[1].offset[0];
            if (destination->pitch[1] == va_drm_prime_surface_desc.layers[1].pitch[0]) {
                CHECK_HIP(hipMemcpyDtoDAsync(destination->channel[1], layer1_mem, chroma_size, hip_stream_));
            } else {
                CHECK_HIP(hipMemcpy2DAsync(destination->channel[1], destination->pitch[1], layer1_mem, va_drm_prime_surface_desc.layers[1].pitch[0],
                    destination->pitch[1], chroma_height, hipMemcpyDeviceToDevice, hip_stream_));
            }
        }
        // copy channel2
        if (va_drm_prime_surface_desc.layers[2].pitch[0] != 0 && destination->pitch[2] != 0 && destination->channel[2] != nullptr) {
            uint32_t chroma_size = destination->pitch[2] * chroma_height;
            uint8_t *layer2_mem = yuv_dev_mem_ + va_drm_prime_surface_desc.layers[2].offset[0];
            if (destination->pitch[2] == va_drm_prime_surface_desc.layers[2].pitch[0]) {
                CHECK_HIP(hipMemcpyDtoDAsync(destination->channel[2], layer2_mem, chroma_size, hip_stream_));
            } else {
                CHECK_HIP(hipMemcpy2DAsync(destination->channel[2], destination->pitch[2], layer2_mem, va_drm_prime_surface_desc.layers[2].pitch[0],
                    destination->pitch[2], chroma_height, hipMemcpyDeviceToDevice, hip_stream_));
            }
        }

        CHECK_HIP(hipStreamSynchronize(hip_stream_));

        CHECK_HIP(hipFree(yuv_dev_mem_));
        CHECK_HIP(hipDestroyExternalMemory(hip_ext_mem_));
        for (int i = 0; i < (int)va_drm_prime_surface_desc.num_objects; ++i) {
            close(va_drm_prime_surface_desc.objects[i].fd);
        }
    }

    return ROCJPEG_STATUS_SUCCESS;

}

RocJpegStatus ROCJpegDecoder::GetImageInfo(const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights){
    std::lock_guard<std::mutex> lock(mutex_);
    if (widths == nullptr || heights == nullptr || num_components == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    if (!jpeg_parser_.ParseJpegStream(data, length)) {
        std::cerr << "ERROR: jpeg parser failed!" << std::endl;
        return ROCJPEG_STATUS_BAD_JPEG;
    }
    const JpegStreamParameters *jpeg_stream_params = jpeg_parser_.GetJpegStreamParameters();

    *num_components = jpeg_stream_params->picture_parameter_buffer.num_components;
    widths[0] = jpeg_stream_params->picture_parameter_buffer.picture_width;
    heights[0] = jpeg_stream_params->picture_parameter_buffer.picture_height;

    switch (jpeg_stream_params->chroma_subsampling) {
        case CSS_444:
            *subsampling = ROCJPEG_CSS_444;
            widths[2] = widths[1] = widths[0];
            heights[2] = heights[1] = heights[0];
            widths[3] = 0;
            heights[3] = 0;
            break;
        case CSS_422:
            *subsampling = ROCJPEG_CSS_422;
            break;
        case CSS_420:
            *subsampling = ROCJPEG_CSS_420;
            widths[1] = widths[0];
            heights[1] = heights[0] >> 1;
            widths[3] = widths[2] = 0;
            heights[3] = heights[2] = 0;
             // the output of the YUV420 chroma subsampling is NV12
            *num_components -= 1;
            break;
        case CSS_411:
            *subsampling = ROCJPEG_CSS_411;
            break;
        case CSS_400:
            *subsampling = ROCJPEG_CSS_400;
            widths[3] = widths[2] = widths[1] = 0;
            heights[3] = heights[2] = heights[1] = 0;
            break;
        default:
            *subsampling = ROCJPEG_CSS_UNKNOWN;
            break;
    }

    return ROCJPEG_STATUS_SUCCESS;
}

bool ROCJpegDecoder::ConvertYUVtoRGB(const void *yuv_dev_mem, const size_t yuv_image_size, uint32_t width, uint32_t height, uint32_t yuv_image_stride, RocJpegChromaSubsampling subsampling,
    void *rgb_dev_mem, const size_t rgb_dev_mem_size, const size_t rgb_image_stride) {

    hipError_t hip_status = hipSuccess;
    size_t luma_size = (yuv_image_stride * align(height, 16));

    switch (subsampling) {
        case ROCJPEG_CSS_420:
            HipExecColorConvertNV12ToRGB(hip_stream_, width, height, (uint8_t *)rgb_dev_mem, rgb_image_stride,
                (const uint8_t *)yuv_dev_mem, yuv_image_stride, (const uint8_t *)yuv_dev_mem + luma_size, yuv_image_stride);
            break;
        case ROCJPEG_CSS_444:
            HipExecColorConvertYUV444ToRGB(hip_stream_, width, height, (uint8_t *)rgb_dev_mem, rgb_image_stride,
                (const uint8_t *)yuv_dev_mem, yuv_image_stride, luma_size);
            break;
        default:
            std::cout << "Error! surface format is not supported!" << std::endl;
            return false;
    }

    hip_status = hipStreamSynchronize(hip_stream_);
    if (hip_status != hipSuccess) {
        std::cout << "ERROR: hipStreamSynchronize failed! (" << hip_status << ")" << std::endl;
        return false;
    }

    return true;
    }