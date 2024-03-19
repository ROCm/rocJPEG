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
    num_devices_{0}, device_id_ {device_id}, hip_stream_ {0}, backend_{backend}, hip_interop_{} {}

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

        rocjpeg_status = GetHipInteropMem(va_drm_prime_surface_desc);
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            return rocjpeg_status;
        }

        uint16_t chroma_height = 0;
        rocjpeg_status = GetChromaHeight(jpeg_stream_params->picture_parameter_buffer.picture_height, chroma_height);
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            return rocjpeg_status;
        }

        switch (output_format) {
            case ROCJPEG_OUTPUT_UNCHANGED:
                // copy the native decoded output buffers from interop memory directly to the destination buffers
                rocjpeg_status = CopyLuma(destination, jpeg_stream_params->picture_parameter_buffer.picture_height);
                if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                    return rocjpeg_status;
                }
                rocjpeg_status = CopyChroma(destination, chroma_height);
                if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                    return rocjpeg_status;
                }
                break;
            case ROCJPEG_OUTPUT_YUV:
                if (hip_interop_.surface_format == 0x56595559 /*YUYV*/) {
                    //TODO extract Y channel from the packed YUYV
                } else {
                    rocjpeg_status = CopyLuma(destination, jpeg_stream_params->picture_parameter_buffer.picture_height);
                    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                        return rocjpeg_status;
                    }
                }

                if (hip_interop_.surface_format == VA_FOURCC_NV12) {
                    // Extract the interleaved UV channels and copy them into the second and thrid channels of the destination.
                    HipExecChannelExtractU16ToU8U8(hip_stream_,
                                                   jpeg_stream_params->picture_parameter_buffer.picture_width >> 1,
                                                   jpeg_stream_params->picture_parameter_buffer.picture_height >> 1,
                                                   destination->channel[1], destination->channel[2], destination->pitch[1],
                                                   hip_interop_.hip_mapped_device_mem + hip_interop_.offset[1] , hip_interop_.pitch[1]);

                } else if (hip_interop_.surface_format == 0x56595559 /*YUYV*/) {
                    // Extract the U and V channels from the packed YUYV and copy them into the second and thrid channels of the destination.

                } else {
                    rocjpeg_status = CopyChroma(destination, chroma_height);
                    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                        return rocjpeg_status;
                    }
                }
                break;
            case ROCJPEG_OUTPUT_Y:
                // copy Luma (Y channel) to the first-channel of the destination.
                if (hip_interop_.surface_format == 0x56595559 /*YUYV*/) {
                    HipExexChannelExtractUYVYToY(hip_stream_,
                                                 jpeg_stream_params->picture_parameter_buffer.picture_width,
                                                 jpeg_stream_params->picture_parameter_buffer.picture_height,
                                                 destination->channel[0], destination->pitch[0],
                                                 hip_interop_.hip_mapped_device_mem, hip_interop_.pitch[0]);

                } else {
                    rocjpeg_status = CopyLuma(destination, jpeg_stream_params->picture_parameter_buffer.picture_height);
                    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                        return rocjpeg_status;
                    }
                }
                break;
            case ROCJPEG_OUTPUT_RGBI:
                switch (hip_interop_.surface_format) {
                    case VA_FOURCC_444P:
                        HipExecColorConvertYUV444ToRGB(hip_stream_, jpeg_stream_params->picture_parameter_buffer.picture_width,
                                                                    jpeg_stream_params->picture_parameter_buffer.picture_height,
                                                                    destination->channel[0], destination->pitch[0],
                                                                    hip_interop_.hip_mapped_device_mem, hip_interop_.pitch[0], hip_interop_.offset[1]);
                        break;
                    case 0x56595559:
                        HipExecColorConvertYUYVToRGB(hip_stream_, jpeg_stream_params->picture_parameter_buffer.picture_width,
                                                                  jpeg_stream_params->picture_parameter_buffer.picture_height,
                                                                  destination->channel[0], destination->pitch[0],
                                                                  hip_interop_.hip_mapped_device_mem, hip_interop_.pitch[0]);
                        break;
                    case VA_FOURCC_NV12:
                        HipExecColorConvertNV12ToRGB(hip_stream_, jpeg_stream_params->picture_parameter_buffer.picture_width,
                                                                  jpeg_stream_params->picture_parameter_buffer.picture_height,
                                                                  destination->channel[0], destination->pitch[0],
                                                                  hip_interop_.hip_mapped_device_mem, hip_interop_.pitch[0],
                                                                  hip_interop_.hip_mapped_device_mem + hip_interop_.offset[1], hip_interop_.pitch[1]);
                        break;
                    case VA_FOURCC_Y800:
                    //TODO add support
                        break;
                    default:
                        std::cerr << "Error! surface format is not supported!" << std::endl;
                        return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
                }
                break;
            default:
                break;
        }

        CHECK_HIP(hipStreamSynchronize(hip_stream_));

        rocjpeg_status = ReleaseHipInteropMem();
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            return rocjpeg_status;
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
            widths[2] = widths[1] = widths[0] >> 1;
            heights[2] = heights[1] = heights[0];
            widths[3] = 0;
            heights[3] = 0;
            break;
        case CSS_420:
            *subsampling = ROCJPEG_CSS_420;
            widths[2] = widths[1] = widths[0] >> 1;
            heights[2] = heights[1] = heights[0] >> 1;
            widths[3] = 0;
            heights[3] = 0;
            break;
        case CSS_400:
            *subsampling = ROCJPEG_CSS_400;
            widths[3] = widths[2] = widths[1] = 0;
            heights[3] = heights[2] = heights[1] = 0;
            break;
        case CSS_411:
            *subsampling = ROCJPEG_CSS_411;
            widths[2] = widths[1] = widths[0] >> 2;
            heights[2] = heights[1] = heights[0];
            widths[3] = 0;
            heights[3] = 0;
            break;
        case CSS_440:
            *subsampling = ROCJPEG_CSS_440;
            widths[2] = widths[1] = widths[0] >> 1;
            heights[2] = heights[1] = heights[0] >> 1;
            widths[3] = 0;
            heights[3] = 0;
            break;
        default:
            *subsampling = ROCJPEG_CSS_UNKNOWN;
            break;
    }

    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::ConvertYUVtoRGB(const void *yuv_dev_mem, uint32_t width, uint32_t height, uint32_t yuv_image_stride, RocJpegChromaSubsampling subsampling,
    void *rgb_dev_mem, const size_t rgb_image_stride) {

    size_t luma_size = (yuv_image_stride * align(height, 16));

    switch (subsampling) {
        case ROCJPEG_CSS_444:
            HipExecColorConvertYUV444ToRGB(hip_stream_, width, height, (uint8_t *)rgb_dev_mem, rgb_image_stride,
                (const uint8_t *)yuv_dev_mem, yuv_image_stride, luma_size);
            break;
        case ROCJPEG_CSS_422:
            //TODO add support
            break;
        case ROCJPEG_CSS_420:
            HipExecColorConvertNV12ToRGB(hip_stream_, width, height, (uint8_t *)rgb_dev_mem, rgb_image_stride,
                (const uint8_t *)yuv_dev_mem, yuv_image_stride, (const uint8_t *)yuv_dev_mem + luma_size, yuv_image_stride);
            break;
        case ROCJPEG_CSS_400:
            //TODO add support
            break;
        default:
            std::cout << "Error! surface format is not supported!" << std::endl;
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }

    return ROCJPEG_STATUS_SUCCESS;
    }

RocJpegStatus ROCJpegDecoder::GetHipInteropMem(VADRMPRIMESurfaceDescriptor &va_drm_prime_surface_desc) {
    hipExternalMemoryHandleDesc external_mem_handle_desc = {};
    hipExternalMemoryBufferDesc external_mem_buffer_desc = {};

    external_mem_handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    external_mem_handle_desc.handle.fd = va_drm_prime_surface_desc.objects[0].fd;
    external_mem_handle_desc.size = va_drm_prime_surface_desc.objects[0].size;

    CHECK_HIP(hipImportExternalMemory(&hip_interop_.hip_ext_mem, &external_mem_handle_desc));

    external_mem_buffer_desc.size = va_drm_prime_surface_desc.objects[0].size;
    CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&hip_interop_.hip_mapped_device_mem, hip_interop_.hip_ext_mem, &external_mem_buffer_desc));

    hip_interop_.surface_format = va_drm_prime_surface_desc.fourcc;
    hip_interop_.width = va_drm_prime_surface_desc.width;
    hip_interop_.height = va_drm_prime_surface_desc.height;

    hip_interop_.offset[0] = va_drm_prime_surface_desc.layers[0].offset[0];
    hip_interop_.offset[1] = va_drm_prime_surface_desc.layers[1].offset[0];
    hip_interop_.offset[2] = va_drm_prime_surface_desc.layers[2].offset[0];

    hip_interop_.pitch[0] = va_drm_prime_surface_desc.layers[0].pitch[0];
    hip_interop_.pitch[1] = va_drm_prime_surface_desc.layers[1].pitch[0];
    hip_interop_.pitch[2] = va_drm_prime_surface_desc.layers[2].pitch[0];

    hip_interop_.num_layers = va_drm_prime_surface_desc.num_layers;

    for (int i = 0; i < (int)va_drm_prime_surface_desc.num_objects; ++i) {
        close(va_drm_prime_surface_desc.objects[i].fd);
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::ReleaseHipInteropMem() {
    CHECK_HIP(hipFree(hip_interop_.hip_mapped_device_mem));
    CHECK_HIP(hipDestroyExternalMemory(hip_interop_.hip_ext_mem));
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::CopyLuma(RocJpegImage *destination, uint16_t picture_height) {
    if (hip_interop_.pitch[0] != 0 && destination->pitch[0] != 0 && destination->channel[0] != nullptr) {
        if (destination->pitch[0] == hip_interop_.pitch[0]) {
            uint32_t luma_size = destination->pitch[0] * picture_height;
            CHECK_HIP(hipMemcpyDtoDAsync(destination->channel[0], hip_interop_.hip_mapped_device_mem, luma_size, hip_stream_));
        } else {
            CHECK_HIP(hipMemcpy2DAsync(destination->channel[0], destination->pitch[0], hip_interop_.hip_mapped_device_mem, hip_interop_.pitch[0],
            destination->pitch[0], picture_height, hipMemcpyDeviceToDevice, hip_stream_));
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::CopyChroma(RocJpegImage *destination, uint16_t chroma_height) {
    // copy channel2
    if (hip_interop_.pitch[1] != 0 && destination->pitch[1] != 0 && destination->channel[1] != nullptr) {
        uint32_t chroma_size = destination->pitch[1] * chroma_height;
        uint8_t *layer1_mem = hip_interop_.hip_mapped_device_mem + hip_interop_.offset[1];
        if (destination->pitch[1] == hip_interop_.pitch[1]) {
            CHECK_HIP(hipMemcpyDtoDAsync(destination->channel[1], layer1_mem, chroma_size, hip_stream_));
        } else {
            CHECK_HIP(hipMemcpy2DAsync(destination->channel[1], destination->pitch[1], layer1_mem, hip_interop_.pitch[1],
                destination->pitch[1], chroma_height, hipMemcpyDeviceToDevice, hip_stream_));
        }
    }
    // copy channel2
    if (hip_interop_.pitch[2] != 0 && destination->pitch[2] != 0 && destination->channel[2] != nullptr) {
        uint32_t chroma_size = destination->pitch[2] * chroma_height;
        uint8_t *layer2_mem = hip_interop_.hip_mapped_device_mem + hip_interop_.offset[2];
        if (destination->pitch[2] == hip_interop_.pitch[2]) {
            CHECK_HIP(hipMemcpyDtoDAsync(destination->channel[2], layer2_mem, chroma_size, hip_stream_));
        } else {
            CHECK_HIP(hipMemcpy2DAsync(destination->channel[2], destination->pitch[2], layer2_mem, hip_interop_.pitch[2],
                destination->pitch[2], chroma_height, hipMemcpyDeviceToDevice, hip_stream_));
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::GetChromaHeight(uint16_t picture_height, uint16_t &chroma_height) {
    switch (hip_interop_.surface_format) {
        case VA_FOURCC_NV12: /*NV12: two-plane 8-bit YUV 4:2:0*/
            chroma_height = picture_height >> 1;
            break;
        case VA_FOURCC_444P: /*444P: three-plane 8-bit YUV 4:4:4*/
            chroma_height = picture_height;
            break;
        case VA_FOURCC_Y800: /*Y800: one-plane 8-bit greyscale YUV 4:0:0*/
            chroma_height = 0;
            break;
        case 0x56595559: /*YUYV: one-plane packed 8-bit YUV 4:2:2. Four bytes per pair of pixels: Y, U, Y, V*/
                         /*Note: va.h doesn't have VA_FOURCC_YUYV defined but vaExportSurfaceHandle returns 0x56595559 for packed YUYV for YUV 4:2:2*/
            chroma_height = picture_height;
            break;
        default:
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }
    return ROCJPEG_STATUS_SUCCESS;
}