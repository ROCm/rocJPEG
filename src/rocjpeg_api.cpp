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

#include "rocjpeg_handle.h"
#include "commons.h"

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle *handle)
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
        ERR(STR("Failed to init the rocDecode handle, ") + STR(e.what()));
        return ROCJPEG_STATUS_NOT_INITIALIZED;
    }
    *handle = rocjpeg_handle;
    return static_cast<RocJpegDecoderHandle *>(rocjpeg_handle)->rocjpeg_decoder->InitializeDecoder();
}

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle)
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
//! \fn RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, const uint8_t *data, size_t length,
//!                     int *num_components, RocJpegChromaSubsampling *subsampling, int *widths, int *heights)
//! Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, const uint8_t *data, size_t length, uint8_t *num_components,
    RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights) {
    if (handle == nullptr || data == nullptr || num_components == nullptr || subsampling == nullptr || widths == nullptr || heights == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;
    auto rocjpeg_handle = static_cast<RocJpegDecoderHandle*>(handle);
    try {
        rocjpeg_status = rocjpeg_handle->rocjpeg_decoder->GetImageInfo(data, length, num_components, subsampling, widths, heights);
    } catch (const std::exception& e) {
        rocjpeg_handle->CaptureError(e.what());
        ERR(e.what());
        return ROCJPEG_STATUS_RUNTIME_ERROR;
    }

    return rocjpeg_status;
}

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, const uint8_t *data, size_t length,
//!                     int *num_components, RocJpegChromaSubsampling *subsampling, int *widths, int *heights)
//! Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, const uint8_t *data, size_t length, RocJpegOutputFormat output_format,
    RocJpegImage *destination, hipStream_t stream) {

    if (handle == nullptr || data == nullptr || destination == nullptr) {
        return ROCJPEG_STATUS_INVALID_PARAMETER;
    }
    RocJpegStatus rocjpeg_status = ROCJPEG_STATUS_SUCCESS;
    auto rocjpeg_handle = static_cast<RocJpegDecoderHandle*>(handle);
    try {
        rocjpeg_status = rocjpeg_handle->rocjpeg_decoder->Decode(data, length, output_format, destination, stream);
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
        case ROCJPEG_STATUS_ALLOCATOR_FAILURE:
            return "ROCJPEG_STATUS_ALLOCATOR_FAILURE";
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
