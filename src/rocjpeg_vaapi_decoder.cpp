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

/**
 * @brief Default constructor for RocJpegVappiMemoryPool class.
 *
 * This constructor initializes the memory pool for different surface formats used in RocJpegVappiDecoder.
 * It creates an empty vector for each surface format and stores it in the mem_pool_ map.
 *
 * @param None
 * @return None
 */
RocJpegVappiMemoryPool::RocJpegVappiMemoryPool() {
std::vector<uint32_t> surface_formats = {VA_RT_FORMAT_RGB32, VA_RT_FORMAT_RGBP, VA_RT_FORMAT_YUV444,
                                         VA_RT_FORMAT_YUV422, VA_RT_FORMAT_YUV420, VA_RT_FORMAT_YUV400};
    for (auto surface_format : surface_formats) {
        mem_pool_[surface_format] = std::vector<RocJpegVappiMemPoolEntry>();
    }
}

/**
 * @brief Releases the resources used by the RocJpegVappiMemoryPool.
 *
 * This function releases the VA-API contexts, surfaces, HIP device memory, and HIP external memory
 * associated with the memory pool. It iterates over each entry in the memory pool and checks if
 * the VA-API context ID, VA-API surface ID, HIP mapped device memory, or HIP external memory is
 * non-zero. If so, it destroys the corresponding resource using the appropriate API function.
 * Finally, it resets the HIP interop structure for each entry in the memory pool.
 */
void RocJpegVappiMemoryPool::ReleaseResources() {
    VAStatus va_status;
    hipError_t hip_status;
    for (auto& pair : mem_pool_) {
        for (auto& entry : pair.second) {
            if (entry.va_context_id != 0) {
                va_status = vaDestroyContext(va_display_, entry.va_context_id);
                if (va_status != VA_STATUS_SUCCESS) {
                    ERR("ERROR: vaDestroyContext failed!");
                }
            }
            if (entry.va_surface_id != 0) {
                va_status = vaDestroySurfaces(va_display_, &entry.va_surface_id, 1);
                if (va_status != VA_STATUS_SUCCESS) {
                    ERR("ERROR: vaDestroySurfaces failed!");
                }
            }
            if (entry.hip_interop.hip_mapped_device_mem != nullptr) {
                hip_status = hipFree(entry.hip_interop.hip_mapped_device_mem);
                 if (hip_status != hipSuccess) {
                    ERR("ERROR: hipFree failed!");
                 }
            }
            if (entry.hip_interop.hip_ext_mem != nullptr) {
                hip_status = hipDestroyExternalMemory(entry.hip_interop.hip_ext_mem);
                 if (hip_status != hipSuccess) {
                    ERR("ERROR: hipDestroyExternalMemory failed!");
                 }
            }
            memset((void*)&entry.hip_interop, 0, sizeof(entry.hip_interop));
        }
    }
}

void RocJpegVappiMemoryPool::SetPoolSize(int32_t max_pool_size) {
    for (auto& pair : mem_pool_) {
        pair.second.reserve(max_pool_size);
    }
}

void RocJpegVappiMemoryPool::SetVaapiDisplay(const VADisplay& va_display) {
    va_display_ = va_display;
}

/**
 * @brief Adds a pool entry to the memory pool for a specific surface format.
 *
 * This function adds a pool entry to the memory pool for a specific surface format.
 * If the memory pool for the given surface format is not full, the new entry is added to the pool.
 * If the memory pool is full, the oldest entry is removed from the pool and replaced with the new entry.
 * If the removed entry has associated resources (VA context, VA surface, HIP memory), they are destroyed and freed.
 *
 * @param surface_format The surface format for which the pool entry is being added.
 * @param pool_entry The pool entry to be added.
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if the operation is successful.
 */
RocJpegStatus RocJpegVappiMemoryPool::AddPoolEntry(uint32_t surface_format, const RocJpegVappiMemPoolEntry& pool_entry) {
    auto& entires = mem_pool_[surface_format];
    if (entires.size() < entires.capacity()) {
        entires.push_back(pool_entry);
    } else {
        if (entires.front().va_context_id != 0) {
            CHECK_VAAPI(vaDestroyContext(va_display_, entires.front().va_context_id));
            entires.front().va_context_id = 0;
        }
        if (entires.front().va_surface_id != 0) {
            CHECK_VAAPI(vaDestroySurfaces(va_display_, &entires.front().va_surface_id, 1));
            entires.front().va_surface_id = 0;
        }
        if (entires.front().hip_interop.hip_mapped_device_mem != nullptr) {
            CHECK_HIP(hipFree(entires.front().hip_interop.hip_mapped_device_mem));
        }
        if (entires.front().hip_interop.hip_ext_mem != nullptr) {
            CHECK_HIP(hipDestroyExternalMemory(entires.front().hip_interop.hip_ext_mem));
        }
        memset((void*)&entires.front().hip_interop, 0, sizeof(entires.front().hip_interop));
        entires.erase(entires.begin());
        entires.push_back(pool_entry);
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Retrieves a `RocJpegVappiMemPoolEntry` from the memory pool based on the specified surface format, image width, and image height.
 *
 * @param surface_format The surface format of the entry to retrieve.
 * @param image_width The width of the image of the entry to retrieve.
 * @param image_height The height of the image of the entry to retrieve.
 * @return The matching `RocJpegVappiMemPoolEntry` if found, or a default-initialized entry if not found.
 */
RocJpegVappiMemPoolEntry RocJpegVappiMemoryPool::GetEntry(uint32_t surface_format, uint32_t image_width, uint32_t image_height) {
    for (const auto& entry : mem_pool_[surface_format]) {
        if (entry.image_width == image_width && entry.image_height == image_height) {
            return entry;
        }
    }
    return {0, 0, 0 , 0, {0}};
}

bool RocJpegVappiMemoryPool::FindSurfaceId(VASurfaceID surface_id) {
    for (auto& pair : mem_pool_) {
        for (auto& entry : pair.second) {
            if (entry.va_surface_id == surface_id) {
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Deletes a surface ID from the memory pool.
 *
 * This function deletes the specified surface ID from the memory pool. It performs the following actions:
 * 1. If the surface ID has a valid context ID, it destroys the context associated with the surface ID.
 * 2. It destroys the surface ID itself.
 * 3. If the surface ID has a valid HIP mapped device memory, it frees the memory.
 * 4. If the surface ID has a valid HIP external memory, it destroys the memory.
 * 5. It clears the HIP interop structure associated with the surface ID.
 * 6. It removes the entry from the memory pool.
 *
 * @param surface_id The surface ID to be deleted.
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if the surface ID was successfully deleted.
 */
RocJpegStatus RocJpegVappiMemoryPool::DeleteSurfaceId(VASurfaceID surface_id) {
    for (auto& pair : mem_pool_) {
        auto& entries = pair.second;
        auto it = std::find_if(entries.begin(), entries.end(),
                              [surface_id](const RocJpegVappiMemPoolEntry& entry) {return entry.va_surface_id == surface_id;});
        if (it != entries.end()) {
            if (it->va_context_id != 0) {
                CHECK_VAAPI(vaDestroyContext(va_display_, it->va_context_id));
                it->va_context_id = 0;
            }
            if (it->va_surface_id != 0) {
                CHECK_VAAPI(vaDestroySurfaces(va_display_, &it->va_surface_id, 1));
                it->va_surface_id = 0;
            }
            if (it->hip_interop.hip_mapped_device_mem != nullptr) {
                CHECK_HIP(hipFree(it->hip_interop.hip_mapped_device_mem));
            }
            if (it->hip_interop.hip_ext_mem != nullptr) {
                CHECK_HIP(hipDestroyExternalMemory(it->hip_interop.hip_ext_mem));
            }
            memset((void*)&it->hip_interop, 0, sizeof(it->hip_interop));

            entries.erase(it);
            break;
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Retrieves the HipInteropDeviceMem associated with a given VASurfaceID from the memory pool.
 *
 * This function searches the memory pool for the entry that matches the provided VASurfaceID.
 * If a matching entry is found and the associated HipInteropDeviceMem is not already initialized,
 * it initializes the HipInteropDeviceMem by exporting the VASurfaceID as a DRM prime surface handle,
 * importing it as an external memory object, and getting the mapped buffer for the external memory.
 * The function then updates the HipInteropDeviceMem with the surface format, width, height, offsets,
 * pitches, and number of layers from the exported surface descriptor.
 *
 * @param surface_id The VASurfaceID to retrieve the HipInteropDeviceMem for.
 * @param hip_interop [out] The retrieved HipInteropDeviceMem.
 * @return RocJpegStatus Returns ROCJPEG_STATUS_SUCCESS if the HipInteropDeviceMem is successfully retrieved,
 *         ROCJPEG_STATUS_INVALID_PARAMETER if the requested surface_id is not found in the memory pool.
 */
RocJpegStatus RocJpegVappiMemoryPool::GetHipInteropMem(VASurfaceID surface_id, HipInteropDeviceMem& hip_interop) {
    for (auto& pair : mem_pool_) {
        auto& entries = pair.second;
        auto it = std::find_if(entries.begin(), entries.end(),
                              [surface_id](const RocJpegVappiMemPoolEntry& entry) {return entry.va_surface_id == surface_id;});
        if (it != entries.end()) {
            if (it->hip_interop.hip_mapped_device_mem == nullptr) {
                VADRMPRIMESurfaceDescriptor va_drm_prime_surface_desc = {};
                CHECK_VAAPI(vaExportSurfaceHandle(va_display_, surface_id, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                    VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                    &va_drm_prime_surface_desc));

                hipExternalMemoryHandleDesc external_mem_handle_desc = {};
                hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
                external_mem_handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
                external_mem_handle_desc.handle.fd = va_drm_prime_surface_desc.objects[0].fd;
                external_mem_handle_desc.size = va_drm_prime_surface_desc.objects[0].size;

                CHECK_HIP(hipImportExternalMemory(&it->hip_interop.hip_ext_mem, &external_mem_handle_desc));
                external_mem_buffer_desc.size = va_drm_prime_surface_desc.objects[0].size;
                CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&it->hip_interop.hip_mapped_device_mem, it->hip_interop.hip_ext_mem, &external_mem_buffer_desc));

                it->hip_interop.surface_format = va_drm_prime_surface_desc.fourcc;
                it->hip_interop.width = va_drm_prime_surface_desc.width;
                it->hip_interop.height = va_drm_prime_surface_desc.height;
                it->hip_interop.offset[0] = va_drm_prime_surface_desc.layers[0].offset[0];
                it->hip_interop.offset[1] = va_drm_prime_surface_desc.layers[1].offset[0];
                it->hip_interop.offset[2] = va_drm_prime_surface_desc.layers[2].offset[0];
                it->hip_interop.pitch[0] = va_drm_prime_surface_desc.layers[0].pitch[0];
                it->hip_interop.pitch[1] = va_drm_prime_surface_desc.layers[1].pitch[0];
                it->hip_interop.pitch[2] = va_drm_prime_surface_desc.layers[2].pitch[0];
                it->hip_interop.num_layers = va_drm_prime_surface_desc.num_layers;

                for (uint32_t i = 0; i < va_drm_prime_surface_desc.num_objects; ++i) {
                    close(va_drm_prime_surface_desc.objects[i].fd);
                }
            }
            hip_interop = it->hip_interop;
            return ROCJPEG_STATUS_SUCCESS;
        }
    }
    // it shouldn't reach here unless the requested surface_id is not in the memory pool.
    ERR("the surface_id: " + TOSTR(surface_id) + " was not found in the memory pool!");
    return ROCJPEG_STATUS_INVALID_PARAMETER;
}
/**
 * @brief Constructs a RocJpegVappiDecoder object.
 *
 * This constructor initializes a RocJpegVappiDecoder object with the specified device ID and default values for other member variables.
 *
 * @param device_id The ID of the device to be used for decoding.
 */
RocJpegVappiDecoder::RocJpegVappiDecoder(int device_id) : device_id_{device_id}, drm_fd_{-1}, min_picture_width_{64}, min_picture_height_{64},
    max_picture_width_{4096}, max_picture_height_{4096}, va_display_{0}, va_config_attrib_{{}}, va_config_id_{0}, va_profile_{VAProfileJPEGBaseline},
    vaapi_mem_pool_(std::make_unique<RocJpegVappiMemoryPool>()), current_vcn_jpeg_spec_{0}, va_picture_parameter_buf_id_{0}, va_quantization_matrix_buf_id_{0}, va_huffmantable_buf_id_{0},
    va_slice_param_buf_id_{0}, va_slice_data_buf_id_{0} {
        vcn_jpeg_spec_ = {{"gfx908", {2, false, false}},
                          {"gfx90a", {2, false, false}},
                          {"gfx940", {24, true, true}},
                          {"gfx941", {32, true, true}},
                          {"gfx942", {32, true, true}},
                          {"gfx1030", {2, false, false}},
                          {"gfx1031", {2, false, false}},
                          {"gfx1032", {2, false, false}},
                          {"gfx1100", {2, false, false}},
                          {"gfx1101", {1, false, false}},
                          {"gfx1102", {2, false, false}}};
};

/**
 * @brief Destructor for the RocJpegVappiDecoder class.
 *
 * This destructor is responsible for cleaning up the resources used by the RocJpegVappiDecoder object.
 * It closes the DRM file descriptor, releases the VAAPI memory pool resources, destroys the VAAPI data buffers,
 * destroys the VAAPI configuration, and terminates the VAAPI display.
 *
 * @note If any of the cleanup operations fail, an error message will be printed.
 */
RocJpegVappiDecoder::~RocJpegVappiDecoder() {
    if (drm_fd_ != -1) {
        close(drm_fd_);
    }
    if (va_display_) {
        vaapi_mem_pool_->ReleaseResources();
        RocJpegStatus rocjpeg_status = DestroyDataBuffers();
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            ERR("Error: Failed to destroy VAAPI buffer");
        }
        VAStatus va_status;
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

/**
 * @brief Initializes the VAAPI decoder for RocJpeg.
 *
 * This function initializes the VAAPI decoder for RocJpeg by setting the device ID, GCN architecture name,
 * and other necessary parameters. It also sets up the VAAPI display and creates the decoder configuration.
 *
 * @param device_name The name of the device.
 * @param gcn_arch_name The name of the GCN architecture.
 * @param device_id The ID of the device.
 * @return The status of the initialization process.
 */
RocJpegStatus RocJpegVappiDecoder::InitializeDecoder(std::string device_name, std::string gcn_arch_name, int device_id) {
    device_id_ = device_id;
    std::size_t pos = gcn_arch_name.find_first_of(":");
    std::string gcn_arch_name_base = (pos != std::string::npos) ? gcn_arch_name.substr(0, pos) : gcn_arch_name;

    auto it = vcn_jpeg_spec_.find(gcn_arch_name_base);
    if (it != vcn_jpeg_spec_.end()) {
        current_vcn_jpeg_spec_ = it->second;
    }
    std::vector<int> visible_devices;
    GetVisibleDevices(visible_devices);

    int offset = 0;
    if (gcn_arch_name_base.compare("gfx940") == 0 ||
        gcn_arch_name_base.compare("gfx941") == 0 ||
        gcn_arch_name_base.compare("gfx942") == 0) {
            std::vector<ComputePartition> current_compute_partitions;
            GetCurrentComputePartition(current_compute_partitions);
            if (current_compute_partitions.empty()) {
                //if the current_compute_partitions is empty then the default SPX mode is assumed.
                if (device_id_ < visible_devices.size()) {
                    offset = visible_devices[device_id_] * 7;
                } else {
                    offset = device_id_ * 7;
                }
            } else {
                GetDrmNodeOffset(device_name, device_id_, visible_devices, current_compute_partitions, offset);
            }
    }

    std::string drm_node = "/dev/dri/renderD";
    if (device_id_ < visible_devices.size()) {
        drm_node += std::to_string(128 + offset + visible_devices[device_id_]);
    } else {
        drm_node += std::to_string(128 + offset + device_id_);
    }
    CHECK_ROCJPEG(InitVAAPI(drm_node));
    CHECK_ROCJPEG(CreateDecoderConfig());

    vaapi_mem_pool_->SetVaapiDisplay(va_display_);
    vaapi_mem_pool_->SetPoolSize(current_vcn_jpeg_spec_.num_jpeg_cores * 2);

    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Initializes the VAAPI decoder.
 *
 * This function initializes the VAAPI decoder by opening the DRM node, creating the va_display,
 * setting the info callback, and initializing the va_display.
 *
 * @param drm_node The path to the DRM node.
 * @return The status of the initialization process.
 *         - ROCJPEG_STATUS_SUCCESS if the initialization is successful.
 *         - ROCJPEG_STATUS_NOT_INITIALIZED if the initialization fails.
 */
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

/**
 * @brief Creates the decoder configuration for the RocJpegVappiDecoder.
 *
 * This function creates the decoder configuration by querying the VA API for supported entrypoints
 * and checking if the hardware JPEG decoder is supported. If the hardware JPEG decoder is supported,
 * it retrieves the maximum picture width and height attributes from the VA API and creates the configuration.
 *
 * @return The status of the decoder configuration creation.
 *         - ROCJPEG_STATUS_SUCCESS if the configuration is created successfully.
 *         - ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED if the hardware JPEG decoder is not supported.
 */
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

/**
 * @brief Destroys the data buffers used by the RocJpegVappiDecoder.
 *
 * This function destroys the data buffers used by the RocJpegVappiDecoder, including the picture parameter buffer,
 * quantization matrix buffer, Huffman table buffer, slice parameter buffer, and slice data buffer.
 *
 * @return The status of the operation. Returns ROCJPEG_STATUS_SUCCESS if the data buffers were successfully destroyed.
 */
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

/**
 * @brief Submits a JPEG decode operation to the VAAPI decoder.
 *
 * This function submits a JPEG decode operation to the VAAPI decoder using the provided JPEG stream parameters.
 * It checks for invalid parameters and unsupported image resolutions before proceeding with the decode operation.
 * The output format is determined based on the requested format and the capabilities of the hardware decoder.
 *
 * @param jpeg_stream_params The JPEG stream parameters for the decode operation.
 * @param surface_id [out] The ID of the output surface where the decoded image will be stored.
 * @param output_format The desired output format for the decoded image.
 * @return The status of the decode operation.
 *         - ROCJPEG_STATUS_SUCCESS if the decode operation was successful.
 *         - ROCJPEG_STATUS_INVALID_PARAMETER if the provided parameters are invalid.
 *         - ROCJPEG_STATUS_JPEG_NOT_SUPPORTED if the JPEG image resolution or chroma subsampling is not supported.
 */
RocJpegStatus RocJpegVappiDecoder::SubmitDecode(const JpegStreamParameters *jpeg_stream_params, uint32_t &surface_id, RocJpegOutputFormat output_format) {
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

    uint32_t surface_format;
    VASurfaceAttrib surface_attrib;
    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeInteger;

    // If RGB output format is requested, and the HW JPEG decoder has a built-in format conversion,
    // set the RGB surface format and attributes to obtain the RGB output directly from the JPEG HW decoder.
    // otherwise set the appropriate surface format and attributes based on the chroma subsampling of the image.
    if ((output_format == ROCJPEG_OUTPUT_RGB || output_format == ROCJPEG_OUTPUT_RGB_PLANAR) && current_vcn_jpeg_spec_.can_convert_to_rgb) {
        if (output_format == ROCJPEG_OUTPUT_RGB) {
            surface_format = VA_RT_FORMAT_RGB32;
            surface_attrib.value.value.i = VA_FOURCC_RGBA;
        } else if (output_format == ROCJPEG_OUTPUT_RGB_PLANAR) {
            surface_format = VA_RT_FORMAT_RGBP;
            surface_attrib.value.value.i = VA_FOURCC_RGBP;
        }
    } else {
        switch (jpeg_stream_params->chroma_subsampling) {
            case CSS_444:
                surface_format = VA_RT_FORMAT_YUV444;
                surface_attrib.value.value.i = VA_FOURCC_444P;
                break;
            case CSS_422:
                surface_format = VA_RT_FORMAT_YUV422;
                surface_attrib.value.value.i = ROCJPEG_FOURCC_YUYV;
                break;
            case CSS_420:
                surface_format = VA_RT_FORMAT_YUV420;
                surface_attrib.value.value.i = VA_FOURCC_NV12;
                break;
            case CSS_400:
                surface_format = VA_RT_FORMAT_YUV400;
                surface_attrib.value.value.i = VA_FOURCC_Y800;
                break;
            default:
                ERR("ERROR: The chroma subsampling is not supported by the VCN hardware!");
                return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
                break;
        }
    }

    RocJpegVappiMemPoolEntry mem_pool_entry = vaapi_mem_pool_->GetEntry(surface_format, jpeg_stream_params->picture_parameter_buffer.picture_width, jpeg_stream_params->picture_parameter_buffer.picture_height);
    VAContextID va_context_id;
    if (mem_pool_entry.va_context_id == 0 && mem_pool_entry.va_surface_id == 0) {
        CHECK_VAAPI(vaCreateSurfaces(va_display_, surface_format, jpeg_stream_params->picture_parameter_buffer.picture_width, jpeg_stream_params->picture_parameter_buffer.picture_height, &surface_id, 1, &surface_attrib, 1));
        CHECK_VAAPI(vaCreateContext(va_display_, va_config_id_, jpeg_stream_params->picture_parameter_buffer.picture_width, jpeg_stream_params->picture_parameter_buffer.picture_height, VA_PROGRESSIVE, &surface_id, 1, &va_context_id));
        mem_pool_entry.image_width = jpeg_stream_params->picture_parameter_buffer.picture_width;
        mem_pool_entry.image_height = jpeg_stream_params->picture_parameter_buffer.picture_height;
        mem_pool_entry.va_surface_id = surface_id;
        mem_pool_entry.va_context_id = va_context_id;
        mem_pool_entry.hip_interop = {};
        CHECK_ROCJPEG(vaapi_mem_pool_->AddPoolEntry(surface_format, mem_pool_entry));
    } else {
        surface_id = mem_pool_entry.va_surface_id;
        va_context_id = mem_pool_entry.va_context_id;
    }

    CHECK_ROCJPEG(DestroyDataBuffers());

    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id, VAPictureParameterBufferType, sizeof(VAPictureParameterBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->picture_parameter_buffer, &va_picture_parameter_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id, VAIQMatrixBufferType, sizeof(VAIQMatrixBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->quantization_matrix_buffer, &va_quantization_matrix_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id, VAHuffmanTableBufferType, sizeof(VAHuffmanTableBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->huffman_table_buffer, &va_huffmantable_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id, VASliceParameterBufferType, sizeof(VASliceParameterBufferJPEGBaseline), 1, (void *)&jpeg_stream_params->slice_parameter_buffer, &va_slice_param_buf_id_));
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id, VASliceDataBufferType, jpeg_stream_params->slice_parameter_buffer.slice_data_size, 1, (void *)jpeg_stream_params->slice_data_buffer, &va_slice_data_buf_id_));

    CHECK_VAAPI(vaBeginPicture(va_display_, va_context_id,  surface_id));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id, &va_picture_parameter_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id, &va_quantization_matrix_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id, &va_huffmantable_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id, &va_slice_param_buf_id_, 1));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id, &va_slice_data_buf_id_, 1));
    CHECK_VAAPI(vaEndPicture(va_display_, va_context_id));

    return ROCJPEG_STATUS_SUCCESS;
}

/**
 * @brief Synchronizes the specified VASurfaceID.
 *
 * This function synchronizes the specified VASurfaceID by querying its status and waiting until it becomes ready.
 * If the surface ID is not found in the VAAPI memory pool, it returns ROCJPEG_STATUS_INVALID_PARAMETER.
 * If any error occurs during synchronization, it returns ROCJPEG_STATUS_RUNTIME_ERROR.
 *
 * @param surface_id The VASurfaceID to synchronize.
 * @return The status of the synchronization operation.
 */
RocJpegStatus RocJpegVappiDecoder::SyncSurface(VASurfaceID surface_id) {
    VASurfaceStatus surface_status;
    if (!vaapi_mem_pool_->FindSurfaceId(surface_id)) {
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

/**
 * @brief Retrieves the HipInteropDeviceMem associated with the specified VASurfaceID.
 *
 * This function retrieves the HipInteropDeviceMem associated with the specified VASurfaceID
 * from the vaapi_mem_pool_ and stores it in the provided `hip_interop` parameter.
 *
 * @param surface_id The VASurfaceID of the surface to retrieve the HipInteropDeviceMem for.
 * @param hip_interop The reference to a HipInteropDeviceMem object where the retrieved memory will be stored.
 * @return The RocJpegStatus indicating the success or failure of the operation.
 */
RocJpegStatus RocJpegVappiDecoder::GetHipInteropMem(VASurfaceID surface_id, HipInteropDeviceMem& hip_interop) {
    return vaapi_mem_pool_->GetHipInteropMem(surface_id, hip_interop);
}

/**
 * @brief Retrieves the visible devices for the RocJpegVappiDecoder.
 *
 * This function retrieves the visible devices for the RocJpegVappiDecoder by reading the value of the environment variable "HIP_VISIBLE_DEVICES".
 * The visible devices are stored in the provided vector `visible_devices_vector`.
 *
 * @param visible_devices_vector The vector to store the visible devices.
 */
void RocJpegVappiDecoder::GetVisibleDevices(std::vector<int>& visible_devices_vetor) {
    char *visible_devices = std::getenv("HIP_VISIBLE_DEVICES");
    if (visible_devices != nullptr) {
        char *token = std::strtok(visible_devices,",");
        while (token != nullptr) {
            visible_devices_vetor.push_back(std::atoi(token));
            token = std::strtok(nullptr,",");
        }
    std::sort(visible_devices_vetor.begin(), visible_devices_vetor.end());
    }
}

/**
 * Retrieves the current compute partitions from the system.
 *
 * This function searches for the "current_compute_partition" file in the "/sys/devices/" directory
 * and reads the partition value from each file found. The partition value is then compared to known
 * partition names and the corresponding ComputePartition enum value is added to the provided vector.
 *
 * @param current_compute_partitions A vector to store the current compute partitions.
 */
void RocJpegVappiDecoder::GetCurrentComputePartition(std::vector<ComputePartition> &current_compute_partitions) {
    std::string search_path = "/sys/devices/";
    std::string partition_file = "current_compute_partition";
    std::error_code ec;
    if (fs::exists(search_path)) {
        for (auto it = fs::recursive_directory_iterator(search_path, fs::directory_options::skip_permission_denied); it != fs::recursive_directory_iterator(); ) {
            try {
                if (it->path().filename() == partition_file) {
                    std::ifstream file(it->path());
                    if (file.is_open()) {
                        std::string partition;
                        std::getline(file, partition);
                        if (partition.compare("SPX") == 0 || partition.compare("spx") == 0) {
                            current_compute_partitions.push_back(kSpx);
                        } else if (partition.compare("DPX") == 0 || partition.compare("dpx") == 0) {
                            current_compute_partitions.push_back(kDpx);
                        } else if (partition.compare("TPX") == 0 || partition.compare("tpx") == 0) {
                            current_compute_partitions.push_back(kTpx);
                        } else if (partition.compare("QPX") == 0 || partition.compare("qpx") == 0) {
                            current_compute_partitions.push_back(kQpx);
                        } else if (partition.compare("CPX") == 0 || partition.compare("cpx") == 0) {
                            current_compute_partitions.push_back(kCpx);
                        }
                        file.close();
                    }
                }
                ++it;
            } catch (fs::filesystem_error& e) {
                it.increment(ec);
            }
        }
    }
}

/**
 * @brief Calculates the offset for the DRM node based on the device name, device ID, visible devices,
 *        current compute partitions, and the selected compute partition.
 *
 * @param device_name The name of the device.
 * @param device_id The ID of the device.
 * @param visible_devices A vector containing the IDs of the visible devices.
 * @param current_compute_partitions A vector containing the current compute partitions.
 * @param offset The calculated offset for the DRM node.
 */
void RocJpegVappiDecoder::GetDrmNodeOffset(std::string device_name, uint8_t device_id, std::vector<int>& visible_devices,
                                                   std::vector<ComputePartition> &current_compute_partitions,
                                                   int &offset) {
    if (!current_compute_partitions.empty()) {
        switch (current_compute_partitions[0]) {
            case kSpx:
                if (device_id < visible_devices.size()) {
                    offset = visible_devices[device_id] * 7;
                } else {
                    offset = device_id * 7;
                }
                break;
            case kDpx:
                if (device_id < visible_devices.size()) {
                    offset = (visible_devices[device_id] / 2) * 6;
                } else {
                    offset = (device_id / 2) * 6;
                }
                break;
            case kTpx:
                // Please note that although there are only 6 XCCs per socket on MI300A,
                // there are two dummy render nodes added by the driver.
                // This needs to be taken into account when creating drm_node on each socket in TPX mode.
                if (device_id < visible_devices.size()) {
                    offset = (visible_devices[device_id] / 3) * 5;
                } else {
                    offset = (device_id / 3) * 5;
                }
                break;
            case kQpx:
                if (device_id < visible_devices.size()) {
                    offset = (visible_devices[device_id] / 4) * 4;
                } else {
                    offset = (device_id / 4) * 4;
                }
                break;
            case kCpx:
                // Please note that both MI300A and MI300X have the same gfx_arch_name which is
                // gfx942. Therefore we cannot use the gfx942 to identify MI300A.
                // instead use the device name and look for MI300A
                // Also, as explained aboe in the TPX mode section, we need to be taken into account
                // the extra two dummy nodes when creating drm_node on each socket in CPX mode as well.
                std::string mi300a = "MI300A";
                size_t found_mi300a = device_name.find(mi300a);
                if (found_mi300a != std::string::npos) {
                    if (device_id < visible_devices.size()) {
                        offset = (visible_devices[device_id] / 6) * 2;
                    } else {
                        offset = (device_id / 6) * 2;
                    }
                }
                break;
        }
    }
}