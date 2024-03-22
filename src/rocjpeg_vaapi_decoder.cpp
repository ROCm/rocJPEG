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

#include "rocjpeg_vaapi_decoder.h"

RocJpegVappiDecoder::RocJpegVappiDecoder(int device_id) : device_id_{device_id}, drm_fd_{-1}, min_picture_width_{64}, min_picture_height_{64},
    max_picture_width_{4096}, max_picture_height_{4096}, va_display_{0}, va_config_attrib_{{}}, va_config_id_{0}, va_profile_{VAProfileJPEGBaseline},
    va_context_id_{0}, va_surface_ids_{}, va_picture_parameter_buf_id_{0}, va_quantization_matrix_buf_id_{0}, va_huffmantable_buf_id_{0},
    va_slice_param_buf_id_{0}, va_slice_data_buf_id_{0} {};

RocJpegVappiDecoder::~RocJpegVappiDecoder() {
    if (drm_fd_ != -1) {
        close(drm_fd_);
    }
    if (va_display_) {
        RocJpegStatus rocjpeg_status = DestroyDataBuffers();
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            ERR("Error: Failed to destroy VAAPI buffer");
        }
        VAStatus va_status;
        if (va_surface_ids_.size() > 0) {
            va_status = vaDestroySurfaces(va_display_, va_surface_ids_.data(), va_surface_ids_.size());
            if (va_status != VA_STATUS_SUCCESS) {
                ERR("ERROR: vaDestroySurfaces failed!");
            }
        }
        if (va_context_id_) {
            va_status = vaDestroyContext(va_display_, va_context_id_);
            if (va_status != VA_STATUS_SUCCESS) {
                ERR("ERROR: vaDestroyContext failed!");
            }
        }
        if (va_config_id_) {
            va_status = vaDestroyConfig(va_display_, va_config_id_);
            if (va_status != VA_STATUS_SUCCESS) {
                ERR("ERROR: vaDestroyConfig failed!");
            }
        }
        va_status = vaTerminate(va_display_);
        if (va_status != VA_STATUS_SUCCESS) {
            ERR("ERROR: vaTerminate failed!");
        }

    }
}

RocJpegStatus RocJpegVappiDecoder::InitializeDecoder(std::string gcn_arch_name) {
    // There are 8 renderDXXX per physical device on gfx940, gfx941, and gfx942
    int num_render_cards_per_device = ((gcn_arch_name.compare("gfx940") == 0) ||
                                       (gcn_arch_name.compare("gfx941") == 0) ||
                                       (gcn_arch_name.compare("gfx942") == 0)) ? 8 : 1;
    std::string drm_node = "/dev/dri/renderD" + std::to_string(128 + device_id_ * num_render_cards_per_device);
    CHECK_ROCJPEG(InitVAAPI(drm_node));
    CHECK_ROCJPEG(CreateDecoderConfig());

    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus RocJpegVappiDecoder::InitVAAPI(std::string drm_node) {
    drm_fd_ = open(drm_node.c_str(), O_RDWR);
    if (drm_fd_ < 0) {
        ERR("ERROR: failed to open drm node " + drm_node);
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    va_display_ = vaGetDisplayDRM(drm_fd_);
    if (!va_display_) {
        ERR("ERROR: failed to create va_display!");
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    vaSetInfoCallback(va_display_, NULL, NULL);
    int major_version = 0, minor_version = 0;
    CHECK_VAAPI(vaInitialize(va_display_, &major_version, &minor_version))
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus RocJpegVappiDecoder::CreateDecoderConfig() {
    int max_num_entrypoints = vaMaxNumEntrypoints(va_display_);
    std::vector<VAEntrypoint> jpeg_entrypoint_list;
    jpeg_entrypoint_list.resize(max_num_entrypoints);
    int num_entrypoints = 0;
    CHECK_VAAPI(vaQueryConfigEntrypoints(va_display_, va_profile_, jpeg_entrypoint_list.data(), &num_entrypoints));
    bool hw_jpeg_decoder_supported = false;
    if (num_entrypoints > 0) {
        for (auto entry_point : jpeg_entrypoint_list) {
            if (entry_point == VAEntrypointVLD) {
                hw_jpeg_decoder_supported = true;
                break;
            }
        }
    } else {
        return ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED;
    }

    if (hw_jpeg_decoder_supported) {
        va_config_attrib_.resize(3);
        va_config_attrib_[0].type = VAConfigAttribRTFormat;
        va_config_attrib_[1].type = VAConfigAttribMaxPictureWidth;
        va_config_attrib_[2].type = VAConfigAttribMaxPictureHeight;
        CHECK_VAAPI(vaGetConfigAttributes(va_display_, va_profile_, VAEntrypointVLD, va_config_attrib_.data(), va_config_attrib_.size()));
        CHECK_VAAPI(vaCreateConfig(va_display_, va_profile_, VAEntrypointVLD, &va_config_attrib_[0], 1, &va_config_id_));
        if (va_config_attrib_[1].value != VA_ATTRIB_NOT_SUPPORTED) {
            max_picture_width_ = va_config_attrib_[1].value;
        }
        if (va_config_attrib_[2].value != VA_ATTRIB_NOT_SUPPORTED) {
            max_picture_height_ = va_config_attrib_[2].value;
        }
        return ROCJPEG_STATUS_SUCCESS;
    } else {
        return ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED;
    }
}

RocJpegStatus RocJpegVappiDecoder::DestroyDataBuffers() {
    if (va_picture_parameter_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_picture_parameter_buf_id_));
        va_picture_parameter_buf_id_ = 0;
    }
    if (va_quantization_matrix_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_quantization_matrix_buf_id_));
        va_quantization_matrix_buf_id_ = 0;
    }
    if (va_huffmantable_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_huffmantable_buf_id_));
        va_huffmantable_buf_id_ = 0;
    }
    if (va_slice_param_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_slice_param_buf_id_));
        va_slice_param_buf_id_ = 0;
    }
    if (va_slice_data_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, va_slice_data_buf_id_));
        va_slice_data_buf_id_ = 0;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus RocJpegVappiDecoder::SubmitDecode(const JpegStreamParameters *jpeg_stream_params, uint32_t &surface_id) {
    if (jpeg_stream_params == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }

    if (sizeof(jpeg_stream_params->picture_parameter_buffer) != sizeof(VAPictureParameterBufferJPEGBaseline) ||
        sizeof(jpeg_stream_params->quantization_matrix_buffer) != sizeof(VAIQMatrixBufferJPEGBaseline) ||
        sizeof(jpeg_stream_params->huffman_table_buffer) != sizeof(VAHuffmanTableBufferJPEGBaseline) ||
        sizeof(jpeg_stream_params->slice_parameter_buffer) != sizeof(VASliceParameterBufferJPEGBaseline)) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }

    if (jpeg_stream_params->picture_parameter_buffer.picture_width < min_picture_width_ ||
        jpeg_stream_params->picture_parameter_buffer.picture_height < min_picture_height_ ||
        jpeg_stream_params->picture_parameter_buffer.picture_width > max_picture_width_ ||
        jpeg_stream_params->picture_parameter_buffer.picture_height > max_picture_height_) {
            ERR("The JPEG image resolution is not supported!");
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
        }

    uint8_t surface_format;
    switch (jpeg_stream_params->chroma_subsampling) {
        case CSS_444:
            surface_format = VA_RT_FORMAT_YUV444;
            break;
        case CSS_422:
            surface_format = VA_RT_FORMAT_YUV422;
            break;
        case CSS_420:
            surface_format = VA_RT_FORMAT_YUV420;
            break;
        case CSS_400:
            surface_format = VA_RT_FORMAT_YUV400;
            break;
        default:
            ERR("ERROR: The chroma subsampling is not supported by the VCN hardware!");
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
            break;
    }

    VASurfaceID va_surface_id;
    CHECK_VAAPI(vaCreateSurfaces(va_display_, surface_format, jpeg_stream_params->picture_parameter_buffer.picture_width, jpeg_stream_params->picture_parameter_buffer.picture_height, &va_surface_id, 1, nullptr, 1));
    va_surface_ids_.push_back(va_surface_id);
    surface_id = va_surface_id;

    if (va_context_id_) {
        vaDestroyContext(va_display_, va_context_id_);
        va_context_id_ = 0;
    }
    CHECK_VAAPI(vaCreateContext(va_display_, va_config_id_, jpeg_stream_params->picture_parameter_buffer.picture_width, jpeg_stream_params->picture_parameter_buffer.picture_height, VA_PROGRESSIVE, &va_surface_id, 1, &va_context_id_));

    CHECK_ROCJPEG(DestroyDataBuffers());

    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAPictureParameterBufferType, sizeof(VAPictureParameterBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->picture_parameter_buffer, &va_picture_parameter_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAIQMatrixBufferType, sizeof(VAIQMatrixBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->quantization_matrix_buffer, &va_quantization_matrix_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAHuffmanTableBufferType, sizeof(VAHuffmanTableBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->huffman_table_buffer, &va_huffmantable_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VASliceParameterBufferType, sizeof(VASliceParameterBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->slice_parameter_buffer, &va_slice_param_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VASliceDataBufferType, jpeg_stream_params->slice_parameter_buffer.slice_data_size, 1, (void *)jpeg_stream_params->slice_data_buffer, &va_slice_data_buf_id_));

    CHECK_VAAPI(vaBeginPicture(va_display_, va_context_id_, va_surface_id));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_picture_parameter_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_quantization_matrix_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_huffmantable_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_slice_param_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &va_slice_data_buf_id_, 1));
    CHECK_VAAPI(vaEndPicture(va_display_, va_context_id_));

    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus RocJpegVappiDecoder::ExportSurface(VASurfaceID surface_id, VADRMPRIMESurfaceDescriptor &va_drm_prime_surface_desc) {

    bool is_surface_id_found = false;
    int idx = 0;
    for (idx = 0; idx < va_surface_ids_.size(); idx++) {
        if (va_surface_ids_[idx] == surface_id) {
            is_surface_id_found = true;
            break;
        }
    }
    if (!is_surface_id_found) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    CHECK_VAAPI(vaExportSurfaceHandle(va_display_, surface_id,
                    VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                    VA_EXPORT_SURFACE_READ_ONLY |
                    VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                    &va_drm_prime_surface_desc));

    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus RocJpegVappiDecoder::SyncSurface(VASurfaceID surface_id) {
    VASurfaceStatus surface_status;
    bool is_surface_id_found = false;
    int idx = 0;

    for (idx = 0; idx < va_surface_ids_.size(); idx++) {
        if (va_surface_ids_[idx] == surface_id) {
            is_surface_id_found = true;
            break;
        }
    }

    if (!is_surface_id_found) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }

    CHECK_VAAPI(vaQuerySurfaceStatus(va_display_, surface_id, &surface_status));
    while (surface_status != VASurfaceReady) {
        VAStatus va_status = vaSyncSurface(va_display_, surface_id);
        if (va_status != VA_STATUS_SUCCESS) {
            if (va_status == 0x26 /*VA_STATUS_ERROR_TIMEDOUT*/) {
                CHECK_VAAPI(vaQuerySurfaceStatus(va_display_, surface_id, &surface_status));
            } else {
                std::cout << "vaSyncSurface() failed with error code: 0x" << std::hex << va_status <<
                    std::dec << "', status: " << vaErrorStr(va_status) << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;
                return ROCJPEG_STATUS_RUNTIME_ERROR;
            }
        } else {
            break;
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus RocJpegVappiDecoder::ReleaseSurface(VASurfaceID surface_id) {
    bool is_surface_id_found = false;
    int idx = 0;

    for (idx = 0; idx < va_surface_ids_.size(); idx++) {
        if (va_surface_ids_[idx] == surface_id) {
            is_surface_id_found = true;
            break;
        }
    }

    if (!is_surface_id_found) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }

    CHECK_VAAPI(vaDestroySurfaces(va_display_, &va_surface_ids_[idx], 1));
    va_surface_ids_.erase(va_surface_ids_.begin() + idx);

    return ROCJPEG_STATUS_SUCCESS;
}