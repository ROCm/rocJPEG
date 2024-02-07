/*
Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include <stdexcept>
#include <exception>
#include <string>
#include <iostream>
#include <cstring>

#define TOSTR(X) std::to_string(static_cast<int>(X))
#define STR(X) std::string(X)

#if DBGINFO
#define INFO(X) std::clog << "[INF] " << " {" << __func__ <<"} " << " " << X << std::endl;
#else
#define INFO(X) ;
#endif
#define ERR(X) std::cerr << "[ERR] "  << " {" << __func__ <<"} " << " " << X << std::endl;

#define CHECK_VAAPI(call) {                                               \
    VAStatus va_status = (call);                                          \
    if (va_status != VA_STATUS_SUCCESS) {                                 \
        std::cout << "VAAPI failure: 'status: " << vaErrorStr(va_status) << "' at " << __FILE__ << ":" << __LINE__ << std::endl;\
        return ROCJPEG_STATUS_EXECUTION_FAILED;                           \
    }                                                                     \
}

#define CHECK_HIP(call) {                                             \
    hipError_t hip_status = (call);                                   \
    if (hip_status != hipSuccess) {                                   \
        std::cout << "HIP failure: 'status: " << hipGetErrorName(hip_status) << "' at " << __FILE__ << ":" << __LINE__ << std::endl;\
        return ROCJPEG_STATUS_EXECUTION_FAILED;                       \
    }                                                                 \
}

static bool GetEnv(const char *name, char *value, size_t valueSize) {
    const char *v = getenv(name);
    if (v) {
        strncpy(value, v, valueSize);
        value[valueSize - 1] = 0;
    }
    return v ? true : false;
}

static inline int align(int value, int alignment) {
   return (value + alignment - 1) & ~(alignment - 1);
}

class RocJpegException : public std::exception {
    public:
        explicit RocJpegException(const std::string& message):message_(message){}
        virtual const char* what() const throw() override {
            return message_.c_str();
        }
    private:
        std::string message_;
};

#define THROW(X) throw RocJpegException(" { "+std::string(__func__)+" } " + X);
