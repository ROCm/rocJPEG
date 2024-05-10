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

#include "rocjpeg_api_stream_handle.h"
#include "rocjpeg_api_decoder_handle.h"
#include "rocjpeg_commons.h"

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegStreamCreate(RocJpegStreamHandle *jpeg_stream_handle)
//! \ingroup group_amd_rocjpeg
//! Create the rocJPEG stream parser handle. This handle is used to parse and retrieve the information from the JPEG stream.
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegStreamCreate(RocJpegStreamHandle *jpeg_stream_handle) {
    if (jpeg_stream_handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegStreamHandle rocjpeg_stream_handle = nullptr;
    try {
        rocjpeg_stream_handle = new RocJpegStreamParserHandle();
    }
    catch(const std::exception& e) {
        ERR(STR("Failed to init the rocJPEG stream handle, ") + STR(e.what()));
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    *jpeg_stream_handle = rocjpeg_stream_handle;
    return ROCJPEG_STATUS_SUCCESS;
}

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegStreamParse(const unsigned char *data, size_t length, RocJpegStreamHandle jpeg_stream_handle)
//! \ingroup group_amd_rocjpeg
//! Parses a jpeg stream and stores the information from the stream to be used in subsequent API calls for retrieving the image information and decoding it.
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegStreamParse(const unsigned char *data, size_t length, RocJpegStreamHandle jpeg_stream_handle) {
    if (data == nullptr || jpeg_stream_handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    auto rocjpeg_stream_handle = static_cast<RocJpegStreamParserHandle*>(jpeg_stream_handle);
    if (!rocjpeg_stream_handle->rocjpeg_stream->ParseJpegStream(data, length)) {
        return ROCJPEG_STATUS_BAD_JPEG;
    }
    return ROCJPEG_STATUS_SUCCESS;
}

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegStreamDestroy(RocJpegStreamHandle jpeg_stream_handle)
//! \ingroup group_amd_rocjpeg
//! Release the jpeg parser object and resources.
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegStreamDestroy(RocJpegStreamHandle jpeg_stream_handle) {
    if (jpeg_stream_handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    auto rocjpeg_stream_handle = static_cast<RocJpegStreamParserHandle*>(jpeg_stream_handle);
    delete rocjpeg_stream_handle;
    return ROCJPEG_STATUS_SUCCESS;
}

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle *handle)
//! \ingroup group_amd_rocjpeg
//! Create the decoder object based on backend and device_id. A handle to the created decoder is returned
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle *handle) {
    if (handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegHandle rocjpeg_handle = nullptr;
    try {
        rocjpeg_handle = new RocJpegDecoderHandle(backend, device_id);
    } catch(const std::exception& e) {
        ERR(STR("Failed to init the rocJPEG handle, ") + STR(e.what()));
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    *handle = rocjpeg_handle;
    return static_cast<RocJpegDecoderHandle *>(rocjpeg_handle)->rocjpeg_decoder->InitializeDecoder();
}

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle)
//! \ingroup group_amd_rocjpeg
//! Release the decoder object and resources.
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle) {
    if (handle == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    auto rocjpeg_handle = static_cast<RocJpegDecoderHandle*>(handle);
    delete rocjpeg_handle;
    return ROCJPEG_STATUS_SUCCESS;
}

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle,
//!                     int *num_components, RocJpegChromaSubsampling *subsampling, int *widths, int *heights)
//! \ingroup group_amd_rocjpeg
//! Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle, uint8_t *num_components,
    RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights) {
    if (handle == nullptr || num_components == nullptr ||
        subsampling == nullptr || widths == nullptr || heights == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;
    auto rocjpeg_handle = static_cast<RocJpegDecoderHandle*>(handle);
    try {
        rocjpeg_status = rocjpeg_handle->rocjpeg_decoder->GetImageInfo(jpeg_stream_handle, num_components, subsampling, widths, heights);
    } catch (const std::exception& e) {
        rocjpeg_handle->CaptureError(e.what());
        ERR(e.what());
        return ROCJPEG_STATUS_RUNTIME_ERROR;
    }

    return rocjpeg_status;
}

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle, const RocJpegDecodeParams *decode_params, RocJpegImage *destination);
//! \ingroup group_amd_rocjpeg
//! Decodes single image based on the backend used to create the rocJpeg handle in rocJpegCreate API.
//! Destination buffers should be large enough to be able to store output of specified format.
//! These buffers should be pre-allocted by the user in the device memories.
//! For each color plane (channel) sizes could be retrieved for image using rocJpegGetImageInfo API
//! and minimum required memory buffer for each plane is plane_height * plane_pitch where plane_pitch >= plane_width for
//! planar output formats and plane_pitch >= plane_width * num_components for interleaved output format.
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, RocJpegStreamHandle jpeg_stream_handle, const RocJpegDecodeParams *decode_params,
    RocJpegImage *destination) {

    if (handle == nullptr || decode_params == nullptr || destination == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;
    auto rocjpeg_handle = static_cast<RocJpegDecoderHandle*>(handle);
    try {
        rocjpeg_status = rocjpeg_handle->rocjpeg_decoder->Decode(jpeg_stream_handle, decode_params, destination);
    } catch (const std::exception& e) {
        rocjpeg_handle->CaptureError(e.what());
        ERR(e.what());
        return ROCJPEG_STATUS_RUNTIME_ERROR;
    }

    return rocjpeg_status;
}

/*****************************************************************************************************/
//! \fn extern const char* ROCDECAPI rocJpegGetErrorName(RocJpegStatus rocjpeg_status);
//! \ingroup group_amd_rocjpeg
//! Return name of the specified error code in text form.
/*****************************************************************************************************/
extern const char* ROCJPEGAPI rocJpegGetErrorName(RocJpegStatus rocjpeg_status) {
    switch (rocjpeg_status) {
        case ROCJPEG_STATUS_SUCCESS:
            return "ROCJPEG_STATUS_SUCCESS";
        case ROCJPEG_STATUS_NOT_INITIALIZED:
            return "ROCJPEG_STATUS_NOT_INITIALIZED";
        case ROCJPEG_STATUS_INVALID_PARAMETER:
            return "ROCJPEG_STATUS_INVALID_PARAMETER";
        case ROCJPEG_STATUS_BAD_JPEG:
            return "ROCJPEG_STATUS_BAD_JPEG";
        case ROCJPEG_STATUS_JPEG_NOT_SUPPORTED:
            return "ROCJPEG_STATUS_JPEG_NOT_SUPPORTED";
        case ROCJPEG_STATUS_EXECUTION_FAILED:
            return "ROCJPEG_STATUS_EXECUTION_FAILED";
        case ROCJPEG_STATUS_ARCH_MISMATCH:
            return "ROCJPEG_STATUS_ARCH_MISMATCH";
        case ROCJPEG_STATUS_INTERNAL_ERROR:
            return "ROCJPEG_STATUS_INTERNAL_ERROR";
        case ROCJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
            return "ROCJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
        case ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED:
            return "ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED";
        case ROCJPEG_STATUS_RUNTIME_ERROR:
            return "ROCJPEG_STATUS_RUNTIME_ERROR";
        case ROCJPEG_STATUS_OUTOF_MEMORY:
            return "ROCJPEG_STATUS_OUTOF_MEMORY";
        case ROCJPEG_STATUS_NOT_IMPLEMENTED:
            return "ROCJPEG_STATUS_NOT_IMPLEMENTED";
        default:
            return "UNKNOWN_ERROR";
    }
}

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegDecodeBatchedInitialize(RocJpegHandle handle, int batch_size, int max_cpu_threads, RocJpegOutputFormat output_format);
//! \ingroup group_amd_rocjpeg
//! Resets and initializes batch decoder for working on the batches of specified size
//! Should be called once for decoding batches of this specific size, also use to reset failed batches
//! \return ROCJPEG_STATUS_SUCCESS if successful
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegDecodeBatchedInitialize(RocJpegHandle handle, int batch_size, int max_cpu_threads, RocJpegOutputFormat output_format) {
    return ROCJPEG_STATUS_NOT_IMPLEMENTED;
}


/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegDecodeBatched(RocJpegHandle handle, const uint8_t *data, const size_t *lengths, RocJpegImage *destinations, hipStream_t stream);
//! \ingroup group_amd_rocjpeg
//! Decodes batch of images. Output buffers should be large enough to be able to store
//! outputs of specified format, see single image decoding description for details. Call to
//! rocjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as
//! parameter to this batch initialization function.
//! \return ROCJPEG_STATUS_SUCCESS if successful
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegDecodeBatched(RocJpegHandle handle, const uint8_t *data, const size_t *lengths, RocJpegImage *destinations) {
    return ROCJPEG_STATUS_NOT_IMPLEMENTED;
}
