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
    num_devices_{0}, device_id_ {device_id}, hip_stream_ {0}, backend_{backend} {}

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
        ERR("ERROR: Failed to initilize the HIP!");
        return rocjpeg_status;
    }
    if (backend_ == ROCJPEG_BACKEND_HARDWARE) {
        rocjpeg_status = jpeg_vaapi_decoder_.InitializeDecoder(hip_dev_prop_.name, hip_dev_prop_.gcnArchName, device_id_);
        if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
            ERR("ERROR: Failed to initialize the VA-API JPEG decoder!");
            return rocjpeg_status;
        }
    } else if (backend_ == ROCJPEG_BACKEND_HYBRID) {
        return ROCJPEG_STATUS_NOT_IMPLEMENTED;
    }
    return rocjpeg_status;
}

RocJpegStatus ROCJpegDecoder::Decode(const uint8_t *data, size_t length, const RocJpegDecodeParams *decode_params, RocJpegImage *destination) {
    std::lock_guard<std::mutex> lock(mutex_);
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;

    if (!jpeg_parser_.ParseJpegStream(data, length)) {
        ERR("ERROR: Failed to parse the jpeg stream!");
        return ROCJPEG_STATUS_BAD_JPEG;
    }

    const JpegStreamParameters *jpeg_stream_params = jpeg_parser_.GetJpegStreamParameters();
    VASurfaceID current_surface_id;
    CHECK_ROCJPEG(jpeg_vaapi_decoder_.SubmitDecode(jpeg_stream_params, current_surface_id, decode_params->output_format));

    HipInteropDeviceMem hip_interop_dev_mem = {};
    CHECK_ROCJPEG(jpeg_vaapi_decoder_.SyncSurface(current_surface_id));
    CHECK_ROCJPEG(jpeg_vaapi_decoder_.GetHipInteropMem(current_surface_id, hip_interop_dev_mem));

    uint16_t chroma_height = 0;

    switch (decode_params->output_format) {
        case ROCJPEG_OUTPUT_NATIVE:
            // Copy the native decoded output buffers from interop memory directly to the destination buffers
            CHECK_ROCJPEG(GetChromaHeight(hip_interop_dev_mem.surface_format, jpeg_stream_params->picture_parameter_buffer.picture_height, chroma_height));
            // Copy Luma (first channel) for any surface format
            CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, jpeg_stream_params->picture_parameter_buffer.picture_height, 0, destination));
            if (hip_interop_dev_mem.surface_format == VA_FOURCC_NV12) {
                // Copy the second channel (UV interleaved) for NV12
                CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, chroma_height, 1, destination));
            } else if (hip_interop_dev_mem.surface_format == VA_FOURCC_444P) {
                // Copy the second and third channels for YUV444
                CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, chroma_height, 1, destination));
                CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, chroma_height, 2, destination));
            }
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            CHECK_ROCJPEG(GetChromaHeight(hip_interop_dev_mem.surface_format, jpeg_stream_params->picture_parameter_buffer.picture_height, chroma_height));
            CHECK_ROCJPEG(GetPlanarYUVOutputFormat(hip_interop_dev_mem, jpeg_stream_params->picture_parameter_buffer.picture_width,
                                                   jpeg_stream_params->picture_parameter_buffer.picture_height, chroma_height, destination));
            break;
        case ROCJPEG_OUTPUT_Y:
            CHECK_ROCJPEG(GetYOutputFormat(hip_interop_dev_mem, jpeg_stream_params->picture_parameter_buffer.picture_width,
                                           jpeg_stream_params->picture_parameter_buffer.picture_height, destination));
            break;
        case ROCJPEG_OUTPUT_RGB:
            CHECK_ROCJPEG(ColorConvertToRGB(hip_interop_dev_mem, jpeg_stream_params->picture_parameter_buffer.picture_width,
                                                    jpeg_stream_params->picture_parameter_buffer.picture_height, destination));
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            CHECK_ROCJPEG(ColorConvertToRGBPlanar(hip_interop_dev_mem, jpeg_stream_params->picture_parameter_buffer.picture_width,
                                                    jpeg_stream_params->picture_parameter_buffer.picture_height, destination));
            break;
        default:
            break;
    }

    CHECK_HIP(hipStreamSynchronize(hip_stream_));


    return ROCJPEG_STATUS_SUCCESS;

}

RocJpegStatus ROCJpegDecoder::GetImageInfo(const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights){
    std::lock_guard<std::mutex> lock(mutex_);
    if (widths == nullptr || heights == nullptr || num_components == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    if (!jpeg_parser_.ParseJpegStream(data, length)) {
        ERR("ERROR: jpeg parser failed!");
        return ROCJPEG_STATUS_BAD_JPEG;
    }
    const JpegStreamParameters *jpeg_stream_params = jpeg_parser_.GetJpegStreamParameters();
    *num_components = jpeg_stream_params->picture_parameter_buffer.num_components;
    widths[0] = jpeg_stream_params->picture_parameter_buffer.picture_width;
    heights[0] = jpeg_stream_params->picture_parameter_buffer.picture_height;
    widths[3] = 0;
    heights[3] = 0;

    switch (jpeg_stream_params->chroma_subsampling) {
        case CSS_444:
            *subsampling = ROCJPEG_CSS_444;
            widths[2] = widths[1] = widths[0];
            heights[2] = heights[1] = heights[0];
            break;
        case CSS_422:
            *subsampling = ROCJPEG_CSS_422;
            widths[2] = widths[1] = widths[0] >> 1;
            heights[2] = heights[1] = heights[0];
            break;
        case CSS_420:
            *subsampling = ROCJPEG_CSS_420;
            widths[2] = widths[1] = widths[0] >> 1;
            heights[2] = heights[1] = heights[0] >> 1;
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
            break;
        case CSS_440:
            *subsampling = ROCJPEG_CSS_440;
            widths[2] = widths[1] = widths[0] >> 1;
            heights[2] = heights[1] = heights[0] >> 1;
            break;
        default:
            *subsampling = ROCJPEG_CSS_UNKNOWN;
            break;
    }

    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::CopyChannel(HipInteropDeviceMem& hip_interop_dev_mem, uint16_t channel_height, uint8_t channel_index, RocJpegImage *destination) {
    if (hip_interop_dev_mem.pitch[channel_index] != 0 && destination->pitch[channel_index] != 0 && destination->channel[channel_index] != nullptr) {
        if (destination->pitch[channel_index] == hip_interop_dev_mem.pitch[channel_index]) {
            uint32_t channel_size = destination->pitch[channel_index] * channel_height;
            CHECK_HIP(hipMemcpyDtoDAsync(destination->channel[channel_index], hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[channel_index], channel_size, hip_stream_));
        } else {
            CHECK_HIP(hipMemcpy2DAsync(destination->channel[channel_index], destination->pitch[channel_index], hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[channel_index], hip_interop_dev_mem.pitch[channel_index],
            destination->pitch[channel_index], channel_height, hipMemcpyDeviceToDevice, hip_stream_));
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::GetChromaHeight(uint32_t surface_format, uint16_t picture_height, uint16_t &chroma_height) {
    switch (surface_format) {
        case VA_FOURCC_NV12: /*NV12: two-plane 8-bit YUV 4:2:0*/
            chroma_height = picture_height >> 1;
            break;
        case VA_FOURCC_444P: /*444P: three-plane 8-bit YUV 4:4:4*/
            chroma_height = picture_height;
            break;
        case VA_FOURCC_Y800: /*Y800: one-plane 8-bit greyscale YUV 4:0:0*/
            chroma_height = 0;
            break;
        case ROCJPEG_FOURCC_YUYV: /*YUYV: one-plane packed 8-bit YUV 4:2:2. Four bytes per pair of pixels: Y, U, Y, V*/
            chroma_height = picture_height;
            break;
        default:
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::ColorConvertToRGB(HipInteropDeviceMem& hip_interop_dev_mem, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination) {
    switch (hip_interop_dev_mem.surface_format) {
        case VA_FOURCC_444P:
            ColorConvertYUV444ToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                  hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0], hip_interop_dev_mem.offset[1]);
            break;
        case ROCJPEG_FOURCC_YUYV:
            ColorConvertYUYVToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0]);
            break;
        case VA_FOURCC_NV12:
            ColorConvertNV12ToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[1], hip_interop_dev_mem.pitch[1]);
            break;
        case VA_FOURCC_Y800:
            ColorConvertYUV400ToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0]);
           break;
        case VA_FOURCC_RGBA:
            ColorConvertRGBAToRGB(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0]);
           break;
        default:
            ERR("ERROR! surface format is not supported!");
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::ColorConvertToRGBPlanar(HipInteropDeviceMem& hip_interop_dev_mem, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination) {
    switch (hip_interop_dev_mem.surface_format) {
        case VA_FOURCC_444P:
            ColorConvertYUV444ToRGBPlanar(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2], destination->pitch[0],
                                                  hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0], hip_interop_dev_mem.offset[1]);
            break;
        case ROCJPEG_FOURCC_YUYV:
            ColorConvertYUYVToRGBPlanar(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0]);
            break;
        case VA_FOURCC_NV12:
            ColorConvertNV12ToRGBPlanar(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[1], hip_interop_dev_mem.pitch[1]);
            break;
        case VA_FOURCC_Y800:
            ColorConvertYUV400ToRGBPlanar(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2], destination->pitch[0],
                                                hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0]);
           break;
        case VA_FOURCC_RGBP:
            // Copy red, green, and blue channels from the interop memory into the destination
            for (uint8_t channel_index = 0; channel_index < 3; channel_index++) {
                CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_height, channel_index, destination));
            }
           break;
        default:
            ERR("ERROR! surface format is not supported!");
            return ROCJPEG_STATUS_JPEG_NOT_SUPPORTED;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::GetPlanarYUVOutputFormat(HipInteropDeviceMem& hip_interop_dev_mem, uint32_t picture_width, uint32_t picture_height, uint16_t chroma_height, RocJpegImage *destination) {
    if (hip_interop_dev_mem.surface_format == ROCJPEG_FOURCC_YUYV) {
        // Extract the packed YUYV and copy them into the first, second, and thrid channels of the destination.
        ConvertPackedYUYVToPlanarYUV(hip_stream_, picture_width, picture_height, destination->channel[0], destination->channel[1], destination->channel[2],
                                                  destination->pitch[0], destination->pitch[1], hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0]);
    } else {
        // Copy Luma
        CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_height, 0, destination));
        if (hip_interop_dev_mem.surface_format == VA_FOURCC_NV12) {
            // Extract the interleaved UV channels and copy them into the second and thrid channels of the destination.
            ConvertInterleavedUVToPlanarUV(hip_stream_, picture_width >> 1, picture_height >> 1, destination->channel[1], destination->channel[2],
                destination->pitch[1], hip_interop_dev_mem.hip_mapped_device_mem + hip_interop_dev_mem.offset[1] , hip_interop_dev_mem.pitch[1]);
        } else if (hip_interop_dev_mem.surface_format == VA_FOURCC_444P) {
            CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, chroma_height, 1, destination));
            CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, chroma_height, 2, destination));
        }
    }
    return ROCJPEG_STATUS_SUCCESS;
}

RocJpegStatus ROCJpegDecoder::GetYOutputFormat(HipInteropDeviceMem& hip_interop_dev_mem, uint32_t picture_width, uint32_t picture_height, RocJpegImage *destination) {
    if (hip_interop_dev_mem.surface_format == ROCJPEG_FOURCC_YUYV) {
        ExtractYFromPackedYUYV(hip_stream_, picture_width, picture_height, destination->channel[0], destination->pitch[0],
                              hip_interop_dev_mem.hip_mapped_device_mem, hip_interop_dev_mem.pitch[0]);
    } else {
        // Copy Luma
        CHECK_ROCJPEG(CopyChannel(hip_interop_dev_mem, picture_height, 0, destination));
    }
    return ROCJPEG_STATUS_SUCCESS;
}