/* Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ROC_JPEG_H
#define ROC_JPEG_H

#define ROCJPEGAPI

#pragma once
#include "hip/hip_runtime.h"

/*****************************************************************************************************************/
//! \file rocjpeg.h
//! \brief The AMD rocJPEG Library.
//! \defgroup group_amd_rocjepg rocJPEG: AMD ROCm JPEG Decode API
//! \brief  rocJPEG API is a toolkit to decode JPEG images using a hardware-accelerated JPEG decoder on AMD’s GPUs.
/******************************************************************************************************************/

#if defined(__cplusplus)
extern "C" {
#endif // __cplusplus

//! \def
//! \ingroup group_amd_rocjpeg
//! Maximum number of channels rocJPEG supports
#define ROCJPEG_MAX_COMPONENT 4

/*****************************************************/
//! \enum RocJpegStatus
//! \ingroup group_amd_rocjpeg
//! rocJPEG return status enums
//! These enums are used in all API calls to rocJPEG
/*****************************************************/
typedef enum {
    ROCJPEG_STATUS_SUCCESS = 0,
    ROCJPEG_STATUS_NOT_INITIALIZED = -1,
    ROCJPEG_STATUS_INVALID_PARAMETER = -2,
    ROCJPEG_STATUS_BAD_JPEG = -3,
    ROCJPEG_STATUS_JPEG_NOT_SUPPORTED = -4,
    ROCJPEG_STATUS_OUTOF_MEMORY = -5,
    ROCJPEG_STATUS_EXECUTION_FAILED = -6,
    ROCJPEG_STATUS_ARCH_MISMATCH = -7,
    ROCJPEG_STATUS_INTERNAL_ERROR = -8,
    ROCJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED = -9,
    ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED = -10,
    ROCJPEG_STATUS_RUNTIME_ERROR = -11,
    ROCJPEG_STATUS_NOT_IMPLEMENTED = -12,
} RocJpegStatus;

/*****************************************************/
//! \enum RocJpegChromaSubsampling
//! \ingroup group_amd_rocjpeg
//! RocJpegChromaSubsampling enum identifies image chroma subsampling values stored inside JPEG input stream
/*****************************************************/
typedef enum {
    ROCJPEG_CSS_444 = 0,
    ROCJPEG_CSS_440 = 1,
    ROCJPEG_CSS_422 = 2,
    ROCJPEG_CSS_420 = 3,
    ROCJPEG_CSS_411 = 4,
    ROCJPEG_CSS_400 = 5,
    ROCJPEG_CSS_UNKNOWN = -1
} RocJpegChromaSubsampling;

/*****************************************************/
//! \struct RocJpegImage
//! \ingroup group_amd_rocjpeg
//! this structure is jpeg image descriptor used to return the decoded output image. User must allocate device
//! memories for each channel for this structure and pass it to the decoder API.
//! the decoder APIs then copies the decode image to this struct based on the requested output format (see RocJpegOutputFormat).
/*****************************************************/
typedef struct {
    uint8_t* channel[ROCJPEG_MAX_COMPONENT];
    uint32_t pitch[ROCJPEG_MAX_COMPONENT]; // pitch of each channel
} RocJpegImage;

/*****************************************************/
//! \enum RocJpegOutputFormat
//! \ingroup group_amd_rocjpeg
//! RocJpegOutputFormat enum specifies what type of output user wants for image decoding
/*****************************************************/
typedef enum {
    // return native unchanged decoded YUV image from the VCN JPEG deocder.
    // For ROCJPEG_CSS_444 write Y, U, and V to first, second, and third channels of RocJpegImage
    // For ROCJPEG_CSS_422 write YUYV (packed) to first channel of RocJpegImage
    // For ROCJPEG_CSS_420 write Y to first channel and UV (interleaved) to second channel of RocJpegImage
    // For ROCJPEG_CSS_400 write Y to first channel of RocJpegImage
    ROCJPEG_OUTPUT_NATIVE = 0,
    // extract Y, U, and V channels from the decoded YUV image from the VCN JPEG deocder and write into first, second, and thrid channel of RocJpegImage.
    // For ROCJPEG_CSS_400 write Y to first channel of RocJpegImage
    ROCJPEG_OUTPUT_YUV_PLANAR = 1,
    // return luma component (Y) and write to first channel of RocJpegImage
    ROCJPEG_OUTPUT_Y = 2,
    // convert to interleaved RGB using HIP kernels and write to first channel of RocJpegImage
    ROCJPEG_OUTPUT_RGB = 3,
    // maximum allowed value
    ROCJPEG_OUTPUT_FORMAT_MAX = 4
} RocJpegOutputFormat;

/*****************************************************/
//! \enum RocJpegBackend
//! \ingroup group_amd_rocjpeg
//! RocJpegBackend enum specifies what type of backend to use for JPEG decoding
//! ROCJPEG_BACKEND_HARDWARE : supports baseline JPEG bitstream using VCN hardware-accelarted JPEG decoder
//! ROCJPEG_BACKEND_HYBRID : uses CPU for Huffman decode and GPU for IDCT using HIP kernesl. This mode doesn't use VCN JPEG hardware decoder
/*****************************************************/
typedef enum {
    ROCJPEG_BACKEND_HARDWARE = 0,
    ROCJPEG_BACKEND_HYBRID = 1
} RocJpegBackend;

/*****************************************************/
// Opaque library handle identifier.
//struct RocJpegDecoderHandle;
//typedef struct RocJpegDecoderHandle* RocJpegHandle;
//! Used in subsequent API calls after rocJpegCreate
/*****************************************************/
typedef void *RocJpegHandle;

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle *handle);
//! \ingroup group_amd_rocjpeg
//! Create the decoder object based on backend and device_id. A handle to the created decoder is returned
//! Initalization of rocjpeg handle. This handle is used for all consecutive calls
//! IN backend : Backend to use.
//! IN device_id : the GPU device id for which a decoder should be created. For example, use 0 for the first GPU device,
//!                 and 1 for the second GPU device on the system, etc.
//! IN/OUT handle : rocjpeg handle, jpeg decoder instance to use for 
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle *handle);

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle);
//! \ingroup group_amd_rocjpeg
//! Release the decoder object and resources.
//! IN/OUT handle: instance handle to release
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle);

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights);
//! \ingroup group_amd_rocjpeg
//! Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
//! If less than ROCJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information
//! If the image is 3-channel, all three groups are valid.
//! IN handle : rocJpeg handle
//! IN data : Pointer to the buffer containing the jpeg stream data to be decoded.
//! IN length : Length of the jpeg image buffer.
//! OUT num_component : Number of channels in the decoded output image
//! OUT subsampling : Chroma subsampling used in this JPEG, see RocJpegChromaSubsampling.
//! OUT widths : pointer to ROCJPEG_MAX_COMPONENT of uint32_t, returns width of each channel.
//! OUT heights : pointer to ROCJPEG_MAX_COMPONENT of uint32_t, returns height of each channel.
//! \return ROCJPEG_STATUS_SUCCESS if successful
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights);

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, const uint8_t *data, size_t length, RocJpegOutputFormat output_format, RocJpegImage *destination, hipStream_t stream);
//! \ingroup group_amd_rocjpeg
//! Decodes single image based on the backend used to create the rocJpeg handle in rocJpegCreate API.
//! Destination buffers should be large enough to be able to store output of specified format. These buffers should be pre-allocted by the user in the device memories.
//! For each color plane (channel) sizes could be retrieved for image using rocJpegGetImageInfo API
//! and minimum required memory buffer for each plane is plane_height * plane_pitch where plane_pitch >= plane_width for
//! planar output formats and plane_pitch >= plane_width * num_components for interleaved output format.
//! IN handle : rocJpeg handle
//! IN data : Pointer to the buffer containing the jpeg stream to be decoded.
//! IN length : Length of the jpeg image buffer.
//! IN output_format : Output data format. See RocJpegOutputFormat for description
//! IN/OUT destination : Pointer to structure with information about output buffers. See RocJpegImage description.
//! \return ROCJPEG_STATUS_SUCCESS if successful
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, const uint8_t *data, size_t length, RocJpegOutputFormat output_format, RocJpegImage *destination);

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegDecodeBatchedInitialize(RocJpegHandle handle, int batch_size, int max_cpu_threads, RocJpegOutputFormat output_format);
//! \ingroup group_amd_rocjpeg
//! Resets and initializes batch decoder for working on the batches of specified size
//! Should be called once for decoding batches of this specific size, also use to reset failed batches
//! IN/OUT     handle          : Library handle
//! IN         batch_size      : Size of the batch
//! IN         max_cpu_threads : Maximum number of CPU threads that will be processing this batch
//! IN         output_format   : Output data format. Will be the same for every image in batch
//! \return ROCJPEG_STATUS_SUCCESS if successful
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegDecodeBatchedInitialize(RocJpegHandle handle, int batch_size, int max_cpu_threads, RocJpegOutputFormat output_format);

/*****************************************************************************************************/
//! \fn RocJpegStatus ROCJPEGAPI rocJpegDecodeBatched(RocJpegHandle handle, const uint8_t *data, const size_t *lengths, RocJpegImage *destinations, hipStream_t stream);
//! \ingroup group_amd_rocjpeg
//! Decodes batch of images. Output buffers should be large enough to be able to store 
//! outputs of specified format, see single image decoding description for details. Call to 
//! rocjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as 
//! parameter to this batch initialization function.
//! 
//! IN/OUT     handle        : Library handle
//! INT/OUT    jpeg_handle   : Decoded jpeg image state handle
//! IN         data          : Array of size batch_size of pointers to the input buffers containing the jpeg images to be decoded. 
//! IN         lengths       : Array of size batch_size with lengths of the jpeg images' buffers in the batch.
//! IN/OUT     destinations  : Array of size batch_size with pointers to structure with information about output buffers, 
//! \return ROCJPEG_STATUS_SUCCESS if successful
/*****************************************************************************************************/
RocJpegStatus ROCJPEGAPI rocJpegDecodeBatched(RocJpegHandle handle, const uint8_t *data, const size_t *lengths, RocJpegImage *destinations);

/*****************************************************************************************************/
//! \fn extern const char* ROCDECAPI rocJpegGetErrorName(RocJpegStatus rocjpeg_status);
//! \ingroup group_amd_rocjpeg
//! Return name of the specified error code in text form.
/*****************************************************************************************************/
extern const char* ROCJPEGAPI rocJpegGetErrorName(RocJpegStatus rocjpeg_status);

#if defined(__cplusplus)
  }
#endif

#endif // ROC_JPEG_H
