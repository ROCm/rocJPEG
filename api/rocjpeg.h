/* Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

/*****************************************************************************************************/
//! \file rocjpeg.h
//! rocJPEG API provides jpeg decoding interface to AMD GPU devices.
//! This file contains constants, structure definitions and function prototypes used for JPEG decoding.
/*****************************************************************************************************/

#if defined(__cplusplus)
extern "C" {
#endif // __cplusplus

// Maximum number of channels ROCJPEG decoder supports
#define ROCJPEG_MAX_COMPONENT 4

// ROCJPEG version information
#define ROCJPEG_VER_MAJOR 0
#define ROCJPEG_VER_MINOR 0
#define ROCJPEG_VER_PATCH 0
#define ROCJPEG_VER_BUILD 0

/* rocJPEG status enums, returned by rocJPEG API */
typedef enum {
    ROCJPEG_STATUS_SUCCESS                       = 0,
    ROCJPEG_STATUS_NOT_INITIALIZED               = -1,
    ROCJPEG_STATUS_INVALID_PARAMETER             = -2,
    ROCJPEG_STATUS_BAD_JPEG                      = -3,
    ROCJPEG_STATUS_JPEG_NOT_SUPPORTED            = -4,
    ROCJPEG_STATUS_ALLOCATOR_FAILURE             = -5,
    ROCJPEG_STATUS_EXECUTION_FAILED              = -6,
    ROCJPEG_STATUS_ARCH_MISMATCH                 = -7,
    ROCJPEG_STATUS_INTERNAL_ERROR                = -8,
    ROCJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED  = -9,
    ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED = -10,
} RocJpegStatus;

// Enum identifies image chroma subsampling values stored inside JPEG input stream
// In the case of ROCJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream
// Otherwise both chroma planes are present
typedef enum {
    ROCJPEG_CSS_444 = 0,
    ROCJPEG_CSS_422 = 1,
    ROCJPEG_CSS_420 = 2,
    ROCJPEG_CSS_411 = 3,
    ROCJPEG_CSS_GRAY = 4,
    ROCJPEG_CSS_UNKNOWN = -1
} RocJpegChromaSubsampling;

// Parameter of this type specifies what type of output user wants for image decoding
typedef enum {
    // return decompressed image as it is - write planar output
    ROCJPEG_OUTPUT_UNCHANGED   = 0,
    // return planar luma and chroma, assuming YCbCr colorspace
    ROCJPEG_OUTPUT_YUV         = 1, 
    // return luma component only, if YCbCr colorspace, 
    // or try to convert to grayscale,
    // writes to 1-st channel of RocJpegImage
    ROCJPEG_OUTPUT_Y           = 2,
    // convert to planar RGB 
    ROCJPEG_OUTPUT_RGB         = 3,
    // convert to planar BGR
    ROCJPEG_OUTPUT_BGR         = 4, 
    // convert to interleaved RGB and write to 1-st channel of RocJpegImage
    ROCJPEG_OUTPUT_RGBI        = 5, 
    // convert to interleaved BGR and write to 1-st channel of RocJpegImage
    ROCJPEG_OUTPUT_BGRI        = 6,
    // maximum allowed value
    ROCJPEG_OUTPUT_FORMAT_MAX  = 6  
} RocJpegOutputFormat;

// Implementation
// ROCJPEG_BACKEND_DEFAULT    : default value
// ROCJPEG_BACKEND_HYBRID     : uses CPU for Huffman decode
// ROCJPEG_BACKEND_GPU_HYBRID : uses GPU assisted Huffman decode. rocjpegDecodeBatched will use GPU decoding for baseline JPEG bitstreams with
//                             interleaved scan when batch size is bigger than 100
// ROCJPEG_BACKEND_HARDWARE   : supports baseline JPEG bitstream with single scan. 410 and 411 sub-samplings are not supported
typedef enum {
    ROCJPEG_BACKEND_DEFAULT = 0,
    ROCJPEG_BACKEND_HYBRID  = 1,
    ROCJPEG_BACKEND_GPU_HYBRID = 2,
    ROCJPEG_BACKEND_HARDWARE = 3
} RocJpegBackend;

// Output descriptor.
// Data that is written to planes depends on output format
typedef struct {
    uint8_t* channel[ROCJPEG_MAX_COMPONENT];
    size_t pitch[ROCJPEG_MAX_COMPONENT];
} RocJpegImage;

// Opaque library handle identifier.
//struct RocJpegDecoderHandle;
//typedef struct RocJpegDecoderHandle* RocJpegHandle;
typedef void *RocJpegHandle;

// Initalization of rocjpeg handle. This handle is used for all consecutive calls
// IN         backend : Backend to use.
// IN         device_id : device id to use.
// INT/OUT    handle  : Codec instance, use for other calls
RocJpegStatus ROCJPEGAPI rocJpegCreate(RocJpegBackend backend, int device_id, RocJpegHandle *handle);

// Release the handle and resources.
// IN/OUT     handle: instance handle to release 
RocJpegStatus ROCJPEGAPI rocJpegDestroy(RocJpegHandle handle);

// Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
// If less than ROCJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information
// If the image is 3-channel, all three groups are valid.
// This function is thread safe.
// IN         handle      : Library handle
// IN         data        : Pointer to the buffer containing the jpeg stream data to be decoded. 
// IN         length      : Length of the jpeg image buffer.
// OUT        num_component  : Number of componenets of the image, currently only supports 1-channel (grayscale) or 3-channel.
// OUT        subsampling : Chroma subsampling used in this JPEG, see nvjpegChromaSubsampling_t
// OUT        widths      : pointer to ROCJPEG_MAX_COMPONENT of ints, returns width of each channel. 0 if channel is not encoded  
// OUT        heights     : pointer to ROCJPEG_MAX_COMPONENT of ints, returns height of each channel. 0 if channel is not encoded 
RocJpegStatus ROCJPEGAPI rocJpegGetImageInfo(RocJpegHandle handle, const uint8_t *data, size_t length, uint8_t *num_components, RocJpegChromaSubsampling *subsampling, uint32_t *widths, uint32_t *heights);

// Decodes single image. The API is back-end agnostic. It will decide on which implementation to use internally
// Destination buffers should be large enough to be able to store  output of specified format.
// For each color plane sizes could be retrieved for image using nvjpegGetImageInfo()
// and minimum required memory buffer for each plane is nPlaneHeight*nPlanePitch where nPlanePitch >= nPlaneWidth for
// planar output formats and nPlanePitch >= nPlaneWidth*nOutputComponents for interleaved output format.
// 
// IN/OUT     handle        : Library handle
// IN         data          : Pointer to the buffer containing the jpeg image to be decoded. 
// IN         length        : Length of the jpeg image buffer.
// IN         output_format : Output data format. See RocJpegOutputFormat for description
// IN/OUT     destination   : Pointer to structure with information about output buffers. See RocJpegImage description.
// IN/OUT     stream        : HIP stream where to submit all GPU work
// 
// \return ROCJPEG_STATUS_SUCCESS if successful
RocJpegStatus ROCJPEGAPI rocJpegDecode(RocJpegHandle handle, const uint8_t *data, size_t length, RocJpegOutputFormat output_format, RocJpegImage *destination, hipStream_t stream);


//////////////////////////////////////////////
/////////////// Batch decoding ///////////////
//////////////////////////////////////////////

// Resets and initizlizes batch decoder for working on the batches of specified size
// Should be called once for decoding bathes of this specific size, also use to reset failed batches
// IN/OUT     handle          : Library handle
// IN         batch_size      : Size of the batch
// IN         max_cpu_threads : Maximum number of CPU threads that will be processing this batch
// IN         output_format   : Output data format. Will be the same for every image in batch
//
// \return ROCJPEG_STATUS_SUCCESS if successful
RocJpegStatus ROCJPEGAPI rocJpegDecodeBatchedInitialize(RocJpegHandle handle, int batch_size, int max_cpu_threads, RocJpegOutputFormat output_format);

// Decodes batch of images. Output buffers should be large enough to be able to store 
// outputs of specified format, see single image decoding description for details. Call to 
// nvjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as 
// parameter to this batch initialization function.
// 
// IN/OUT     handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
// IN         data          : Array of size batch_size of pointers to the input buffers containing the jpeg images to be decoded. 
// IN         lengths       : Array of size batch_size with lengths of the jpeg images' buffers in the batch.
// IN/OUT     destinations  : Array of size batch_size with pointers to structure with information about output buffers, 
// IN/OUT     stream        : CUDA stream where to submit all GPU work
// 
// \return NVJPEG_STATUS_SUCCESS if successful
RocJpegStatus ROCJPEGAPI rocJpegDecodeBatched(RocJpegHandle handle, const uint8_t *const *data, const size_t *lengths, RocJpegImage *destinations, hipStream_t stream);

// Allocates the internal buffers as a pre-allocation step
// IN    handle          : Library handle
// IN    jpeg_handle     : Decoded jpeg image state handle
// IN    width   : frame width
// IN    height  : frame height
// IN    chroma_subsampling   : chroma subsampling of images to be decoded
// IN    output_format : out format

RocJpegStatus ROCJPEGAPI rocJpegDecodeBatchedPreAllocate(RocJpegHandle handle, int batch_size, int width, int height, RocJpegChromaSubsampling chroma_subsampling, RocJpegOutputFormat output_format);



#if defined(__cplusplus)
  }
#endif

#endif // ROC_JPEG_H
