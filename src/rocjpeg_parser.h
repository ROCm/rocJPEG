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


#ifndef ROC_JPEG_PARSER_H_
#define ROC_JPEG_PARSER_H_

#include <stdint.h>
#include <iostream>
#include <cstring>
#include <mutex>
#include "rocjpeg_commons.h"

#pragma once

#define NUM_COMPONENTS 4
#define HUFFMAN_TABLES 2
#define AC_HUFFMAN_TABLE_VALUES_SIZE 162
#define DC_HUFFMAN_TABLE_VALUES_SIZE 12
#define swap_bytes(x) (((x)[0] << 8) | (x)[1])

/***************************************************************/
//! \enum enum JpegMarker
//! common JPEG markers
/***************************************************************/
enum JpegMarkers {
    SOI = 0xD8, /* Start Of Image */
    SOF = 0xC0, /* Start Of Frame for a baseline DCT-based JPEG. */
    DHT = 0xC4, /* Define Huffman Table */
    DQT = 0xDB, /* Define Quantization Table */
    DRI = 0xDD, /* Define Restart Interval */
    SOS = 0xDA, /* Start of Scan */
    EOI = 0xD9, /* End Of Image */
};

/***************************************************************/
//! \struct Picture parameter for JPEG decoding.
//! This structure holds information from the frame
//!  header and definitions from additional segments.
 /**************************************************************/
typedef struct PictureParameterBufferType {
    uint16_t picture_width;
    uint16_t picture_height;
    struct {
        uint8_t component_id;
        uint8_t h_sampling_factor;
        uint8_t v_sampling_factor;
        uint8_t quantiser_table_selector;
    } components[255];
    uint8_t num_components;
    uint8_t color_space;
    uint32_t rotation;
    uint32_t reserved[7];
} PictureParameterBuffer;

/***************************************************************/
//! \struct Quantization table for JPEG decoding.
//! This structure holds the quantization tables.
//! The maximum number of quatization tables is four.
//! The #load_quantization_table array can be used as a hint to notify
//! which table(s) has valid values.
//! The #quantiser_table values are specified in zig-zag scan order.
/***************************************************************/
typedef struct QuantizationMatrixBufferType {
    uint8_t load_quantiser_table[4];
    uint8_t quantiser_table[4][64];
    uint32_t reserved[4];
} QuantizationMatrixBuffer;

/***************************************************************/
//! \struct Huffman table for JPEG decoding.
//! This structure holds the Huffman tables.
//! The maximum number of Huffman tables is two.
//! The #load_huffman_table array can be used as a hint to notify the
//! which table(s) has valid values.
/***************************************************************/
typedef struct HuffmanTableBufferType {
    uint8_t load_huffman_table[2];
    struct {
        uint8_t num_dc_codes[16];
        uint8_t dc_values[12];
        uint8_t num_ac_codes[16];
        uint8_t ac_values[162];
        uint8_t pad[2];
    } huffman_table[2];
    uint32_t reserved[4];
} HuffmanTableBuffer;

/***************************************************************/
//! \struct Slice parameter for JPEG decoding.
//! This structure holds information from the scan header, and
//! definitions from additional segments.
/***************************************************************/
typedef struct SliceParameterBufferType {
    uint32_t slice_data_size;
    uint32_t slice_data_offset;
    uint32_t slice_data_flag;
    uint32_t slice_horizontal_position;
    uint32_t slice_vertical_position;
    struct {
        uint8_t component_selector;
        uint8_t dc_table_selector;
        uint8_t ac_table_selector;
    } components[4];
    uint8_t num_components;
    uint16_t restart_interval;
    uint32_t num_mcus;
    uint32_t reserved[4];
} SliceParameterBuffer;

/***************************************************************/
//! \struct Enum identifies image chroma subsampling values stored inside JPEG input stream
/***************************************************************/
typedef enum {
    CSS_444 = 0,
    CSS_440 = 1,
    CSS_422 = 2,
    CSS_420 = 3,
    CSS_411 = 4,
    CSS_400 = 5,
    CSS_UNKNOWN = -1
} ChromaSubsampling;

/***************************************************************/
//! \struct Jpeg stream parameters.
//! This structure holds all information from a JPEG stream for decoding
/***************************************************************/
typedef struct JpegParameterBuffersType {
    PictureParameterBuffer picture_parameter_buffer;
    QuantizationMatrixBuffer quantization_matrix_buffer;
    HuffmanTableBuffer huffman_table_buffer;
    SliceParameterBuffer slice_parameter_buffer;
    ChromaSubsampling chroma_subsampling;
    const uint8_t* slice_data_buffer;
} JpegStreamParameters;

class RocJpegStreamParser {
    public:
        RocJpegStreamParser();
        ~RocJpegStreamParser();
        bool ParseJpegStream(const uint8_t* jpeg_stream, uint32_t jpeg_stream_size);
        const JpegStreamParameters* GetJpegStreamParameters() const {return &jpeg_stream_parameters_;};
    private:
        bool ParseSOI();
        bool ParseSOF();
        bool ParseDQT();
        bool ParseSOS();
        bool ParseDHT();
        bool ParseDRI();
        bool ParseEOI();
        ChromaSubsampling GetChromaSubsampling(uint8_t c1_h_sampling_factor, uint8_t c2_h_sampling_factor, uint8_t c3_h_sampling_factor,
                                               uint8_t c1_v_sampling_factor, uint8_t c2_v_sampling_factor, uint8_t c3_v_sampling_factor);
        const uint8_t *stream_;
        const uint8_t *stream_end_;
        uint32_t stream_length_;
        JpegStreamParameters jpeg_stream_parameters_;
        std::mutex mutex_;
};

#endif  // ROC_JPEG_PARSER_H_