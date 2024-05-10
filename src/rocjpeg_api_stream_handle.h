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

#ifndef ROC_JPEG_STREAM_HANDLE_H
#define ROC_JPEG_STREAM_HANDLE_H

#pragma once

#include <memory>
#include "rocjpeg_parser.h"

/**
 * @brief RocJpegStreamParserHandle class
 * This class wraps around RocJpegStreamParser to parse and store information from a JPEG stream.
 */
class RocJpegStreamParserHandle {
    public:
        explicit RocJpegStreamParserHandle() : rocjpeg_stream(std::make_shared<RocJpegStreamParser>()) {};
        ~RocJpegStreamParserHandle() { ClearErrors(); }
        std::shared_ptr<RocJpegStreamParser> rocjpeg_stream;
        bool NoError() { return error_.empty(); }
        const char* ErrorMsg() { return error_.c_str(); }
        void CaptureError(const std::string& err_msg) { error_ = err_msg; }
    private:
        void ClearErrors() { error_ = "";}
        std::string error_;
};

#endif //ROC_JPEG_STREAM_HANDLE_H