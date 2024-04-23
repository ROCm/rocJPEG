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
#ifndef ROC_JPEG_SAMPLES_COMMON
#define ROC_JPEG_SAMPLES_COMMON
#pragma once

#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <chrono>
#include <sys/stat.h>
#include <libgen.h>
#include <filesystem>
#include <fstream>
#include <queue>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <atomic>
#include "rocjpeg.h"

#define CHECK_ROCJPEG(call) {                                             \
    RocJpegStatus rocjpeg_status = (call);                                \
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                       \
        std::cerr << #call << " returned " << rocJpegGetErrorName(rocjpeg_status) << " at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                          \
    }                                                                     \
}

#define CHECK_HIP(call) {                                             \
    hipError_t hip_status = (call);                                   \
    if (hip_status != hipSuccess) {                                   \
        std::cout << "rocJPEG failure: '#" << hip_status << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                      \
    }                                                                 \
}

void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options:" << std::endl
    << "-i Path to single image or directory of images - required" << std::endl
    << "-be Select rocJPEG backend (0 for ROCJPEG_BACKEND_HARDWARE, using VCN hardware-accelarated JPEG decoder, 1 ROCJPEG_BACKEND_HYBRID, " <<
        "using CPU and GPU HIP kernles for JPEG decoding); optional; default: 0" << std::endl
    << "-fmt Select rocJPEG output format for decoding, one of the [native, yuv, y, rgb, rgb_planar]; optional; default: native" << std::endl
    << "-o Output file path or directory - Write decoded images based on the selected outfut format to this file or directory; optional;" << std::endl
    << "-d GPU device id (0 for the first GPU device, 1 for the second GPU device, etc.); optional; default: 0" << std::endl
    << "-t Number of threads - optional; default: 2" << std::endl;
    exit(0);
}

void ParseCommandLine(std::string &input_path, std::string &output_file_path, int &dump_output_frames, int &device_id,
    RocJpegBackend &rocjpeg_backend, RocJpegOutputFormat &output_format, int *num_threads, int argc, char *argv[]) {
    if(argc <= 1) {
        ShowHelpAndExit();
    }
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!strcmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            input_path = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            output_file_path = argv[i];
            dump_output_frames = 1;
            continue;
        }
        if (!strcmp(argv[i], "-d")) {
            if (++i == argc) {
                ShowHelpAndExit("-d");
            }
            device_id = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-be")) {
            if (++i == argc) {
                ShowHelpAndExit("-be");
            }
            rocjpeg_backend = static_cast<RocJpegBackend>(atoi(argv[i]));
            continue;
        }
        if (!strcmp(argv[i], "-fmt")) {
            if (++i == argc) {
                ShowHelpAndExit("-fmt");
            }
            std::string selected_output_format = argv[i];
            if (selected_output_format == "native") {
                output_format = ROCJPEG_OUTPUT_NATIVE;
            } else if (selected_output_format == "yuv") {
                output_format = ROCJPEG_OUTPUT_YUV_PLANAR;
            } else if (selected_output_format == "y") {
                output_format = ROCJPEG_OUTPUT_Y;
            } else if (selected_output_format == "rgb") {
                output_format = ROCJPEG_OUTPUT_RGB;
            } else if (selected_output_format == "rgb_planar") {
                output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
            } else {
                ShowHelpAndExit(argv[i]);
            }
            continue;
        }
        if (!strcmp(argv[i], "-t")) {
            if (++i == argc) {
                ShowHelpAndExit("-t");
            }
            if (num_threads != nullptr)
                *num_threads = atoi(argv[i]);
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

void SaveImage(std::string output_file_name, RocJpegImage *output_image, uint32_t img_width, uint32_t img_height,
    RocJpegChromaSubsampling subsampling, RocJpegOutputFormat output_format) {
    uint8_t *hst_ptr = nullptr;
    FILE *fp;
    hipError_t hip_status = hipSuccess;

    if (output_image == nullptr || output_image->channel[0] == nullptr || output_image->pitch[0] == 0) {
        return;
    }

    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};

    switch (output_format) {
        case ROCJPEG_OUTPUT_NATIVE:
            switch (subsampling) {
                case ROCJPEG_CSS_444:
                    widths[2] = widths[1] = widths[0] = img_width;
                    heights[2] = heights[1] = heights[0] = img_height;
                    break;
                case ROCJPEG_CSS_422:
                    widths[0] = img_width * 2;
                    heights[0] = img_height;
                    break;
                case ROCJPEG_CSS_420:
                    widths[1] = widths[0] = img_width;
                    heights[0] = img_height;
                    heights[1] = img_height >> 1;
                    break;
                case ROCJPEG_CSS_400:
                    widths[0] = img_width;
                    heights[0] = img_height;
                    break;
                default:
                    std::cout << "Unknown chroma subsampling!" << std::endl;
                    return;
            }
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            switch (subsampling) {
                case ROCJPEG_CSS_444:
                    widths[2] = widths[1] = widths[0] = img_width;
                    heights[2] = heights[1] = heights[0] = img_height;
                    break;
                case ROCJPEG_CSS_422:
                    widths[0] = img_width;
                    widths[2] = widths[1] = widths[0] >> 1;
                    heights[2] = heights[1] = heights[0] = img_height;
                    break;
                case ROCJPEG_CSS_420:
                    widths[0] = img_width;
                    widths[2] = widths[1] = widths[0] >> 1;
                    heights[0] = img_height;
                    heights[2] = heights[1] = img_height >> 1;
                    break;
                case ROCJPEG_CSS_400:
                    widths[0] = img_width;
                    heights[0] = img_height;
                    break;
                default:
                    std::cout << "Unknown chroma subsampling!" << std::endl;
                    return;
            }
            break;
        case ROCJPEG_OUTPUT_Y:
            widths[0] = img_width;
            heights[0] = img_height;
            break;
        case ROCJPEG_OUTPUT_RGB:
            widths[0] = img_width * 3;
            heights[0] = img_height;
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            widths[2] = widths[1] = widths[0] = img_width;
            heights[2] = heights[1] = heights[0] = img_height;
            break;
        default:
            std::cout << "Unknown output format!" << std::endl;
            return;
    }

    uint32_t channel0_size = output_image->pitch[0] * heights[0];
    uint32_t channel1_size = output_image->pitch[1] * heights[1];
    uint32_t channel2_size = output_image->pitch[2] * heights[2];

    uint32_t output_image_size = channel0_size + channel1_size + channel2_size;

    if (hst_ptr == nullptr) {
        hst_ptr = new uint8_t [output_image_size];
    }

    CHECK_HIP(hipMemcpyDtoH((void *)hst_ptr, output_image->channel[0], channel0_size));

    uint8_t *tmp_hst_ptr = hst_ptr;
    fp = fopen(output_file_name.c_str(), "wb");
    if (fp) {
        // write channel0
        if (widths[0] == output_image->pitch[0]) {
            fwrite(hst_ptr, 1, channel0_size, fp);
        } else {
            for (int i = 0; i < heights[0]; i++) {
                fwrite(tmp_hst_ptr, 1, widths[0], fp);
                tmp_hst_ptr += output_image->pitch[0];
            }
        }
        // write channel1
        if (channel1_size != 0 && output_image->channel[1] != nullptr) {
            uint8_t *channel1_hst_ptr = hst_ptr + channel0_size;
            CHECK_HIP(hipMemcpyDtoH((void *)channel1_hst_ptr, output_image->channel[1], channel1_size));
            if (widths[1] == output_image->pitch[1]) {
                fwrite(channel1_hst_ptr, 1, channel1_size, fp);
            } else {
                for (int i = 0; i < heights[1]; i++) {
                    fwrite(channel1_hst_ptr, 1, widths[1], fp);
                    channel1_hst_ptr += output_image->pitch[1];
                }
            }
        }
        // write channel2
        if (channel2_size != 0 && output_image->channel[2] != nullptr) {
            uint8_t *channel2_hst_ptr = hst_ptr + channel0_size + channel1_size;
            CHECK_HIP(hipMemcpyDtoH((void *)channel2_hst_ptr, output_image->channel[2], channel2_size));
            if (widths[2] == output_image->pitch[2]) {
                fwrite(channel2_hst_ptr, 1, channel2_size, fp);
            } else {
                for (int i = 0; i < heights[2]; i++) {
                    fwrite(channel2_hst_ptr, 1, widths[2], fp);
                    channel2_hst_ptr += output_image->pitch[2];
                }
            }
        }
        fclose(fp);
    }

    if (hst_ptr != nullptr) {
        delete [] hst_ptr;
        hst_ptr = nullptr;
        tmp_hst_ptr = nullptr;
    }
}

bool GetFilePaths(std::string &input_path, std::vector<std::string> &file_paths, bool &is_dir, bool &is_file) {
    is_dir = std::filesystem::is_directory(input_path);
    is_file = std::filesystem::is_regular_file(input_path);
    if (is_dir) {
        for (const auto &entry : std::filesystem::directory_iterator(input_path))
            file_paths.push_back(entry.path());
    } else if (is_file) {
        file_paths.push_back(input_path);
    } else {
        std::cerr << "ERROR: the input path is not valid!" << std::endl;
        return false;
    }
    return true;
}

bool InitHipDevice(int device_id) {
    int num_devices;
    hipDeviceProp_t hip_dev_prop;
    CHECK_HIP(hipGetDeviceCount(&num_devices));
    if (num_devices < 1) {
        std::cerr << "ERROR: didn't find any GPU!" << std::endl;
        return false;
    }
    if (device_id >= num_devices) {
        std::cerr << "ERROR: the requested device_id is not found!" << std::endl;
        return false;
    }
    CHECK_HIP(hipSetDevice(device_id));
    CHECK_HIP(hipGetDeviceProperties(&hip_dev_prop, device_id));

    std::cout << "info: Using GPU device " << device_id << ": " << hip_dev_prop.name << "[" << hip_dev_prop.gcnArchName << "] on PCI bus " <<
    std::setfill('0') << std::setw(2) << std::right << std::hex << hip_dev_prop.pciBusID << ":" << std::setfill('0') << std::setw(2) <<
    std::right << std::hex << hip_dev_prop.pciDomainID << "." << hip_dev_prop.pciDeviceID << std::dec << std::endl;

    return true;
}

void GetChromaSubsamplingStr(RocJpegChromaSubsampling subsampling, std::string &chroma_sub_sampling) {
    switch (subsampling) {
        case ROCJPEG_CSS_444:
            chroma_sub_sampling = "YUV 4:4:4";
            break;
        case ROCJPEG_CSS_440:
            chroma_sub_sampling = "YUV 4:4:0";
            break;
        case ROCJPEG_CSS_422:
            chroma_sub_sampling = "YUV 4:2:2";
            break;
        case ROCJPEG_CSS_420:
            chroma_sub_sampling = "YUV 4:2:0";
            break;
        case ROCJPEG_CSS_411:
            chroma_sub_sampling = "YUV 4:1:1";
            break;
        case ROCJPEG_CSS_400:
            chroma_sub_sampling = "YUV 4:0:0";
            break;
        case ROCJPEG_CSS_UNKNOWN:
            chroma_sub_sampling = "UNKNOWN";
            break;
        default:
            chroma_sub_sampling = "";
            break;
    }
}

void GetFileExtForSaving(RocJpegOutputFormat output_format, std::string &base_file_name, uint32_t image_width, uint32_t image_height, std::string &file_name_for_saving) {
    std::string file_extension;
    std::string::size_type const p(base_file_name.find_last_of('.'));
    std::string file_name_no_ext = base_file_name.substr(0, p);
    switch (output_format) {
        case ROCJPEG_OUTPUT_NATIVE:
            file_extension = "native";
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            file_extension = "yuv";
            break;
        case ROCJPEG_OUTPUT_Y:
            file_extension = "y";
            break;
        case ROCJPEG_OUTPUT_RGB:
            file_extension = "rgb";
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            file_extension = "rgb_planar";
            break;
        default:
            file_extension = "";
            break;
    }
    file_name_for_saving += "//" + file_name_no_ext + "_" + std::to_string(image_width) + "x"
        + std::to_string(image_height) + "." + file_extension;
}

int GetChannelPitchAndSizes(RocJpegOutputFormat output_format, RocJpegChromaSubsampling subsampling, uint32_t *widths, uint32_t *heights,
    uint32_t &num_channels, RocJpegImage &output_image, uint32_t *channel_sizes) {
    switch (output_format) {
        case ROCJPEG_OUTPUT_NATIVE:
            switch (subsampling) {
                case ROCJPEG_CSS_444:
                    num_channels = 3;
                    output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = widths[0];
                    channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * heights[0];
                    break;
                case ROCJPEG_CSS_422:
                    num_channels = 1;
                    output_image.pitch[0] = widths[0] * 2;
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    break;
                case ROCJPEG_CSS_420:
                    num_channels = 2;
                    output_image.pitch[1] = output_image.pitch[0] = widths[0];
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    channel_sizes[1] = output_image.pitch[1] * (heights[0] >> 1);
                    break;
                case ROCJPEG_CSS_400:
                    num_channels = 1;
                    output_image.pitch[0] = widths[0];
                    channel_sizes[0] = output_image.pitch[0] * heights[0];
                    break;
                default:
                    std::cout << "Unknown chroma subsampling!" << std::endl;
                    return EXIT_FAILURE;
            }
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            if (subsampling == ROCJPEG_CSS_400) {
                num_channels = 1;
                output_image.pitch[0] = widths[0];
                channel_sizes[0] = output_image.pitch[0] * heights[0];
            } else {
                num_channels = 3;
                output_image.pitch[0] = widths[0];
                output_image.pitch[1] = widths[1];
                output_image.pitch[2] = widths[2];
                channel_sizes[0] = output_image.pitch[0] * heights[0];
                channel_sizes[1] = output_image.pitch[1] * heights[1];
                channel_sizes[2] = output_image.pitch[2] * heights[2];
            }
            break;
        case ROCJPEG_OUTPUT_Y:
            num_channels = 1;
            output_image.pitch[0] = widths[0];
            channel_sizes[0] = output_image.pitch[0] * heights[0];
            break;
        case ROCJPEG_OUTPUT_RGB:
            num_channels = 1;
            output_image.pitch[0] = widths[0] * 3;
            channel_sizes[0] = output_image.pitch[0] * heights[0];
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            num_channels = 3;
            output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = widths[0];
            channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * heights[0];
            break;
        default:
            std::cout << "Unknown output format!" << std::endl;
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

class ThreadPool {
    public:
        ThreadPool(int nthreads) : shutdown_(false) {
            // Create the specified number of threads
            threads_.reserve(nthreads);
            for (int i = 0; i < nthreads; ++i)
                threads_.emplace_back(std::bind(&ThreadPool::ThreadEntry, this, i));
        }
        ~ThreadPool() {}
        void JoinThreads() {
            {
                // Unblock any threads and tell them to stop
                std::unique_lock<std::mutex> lock(mutex_);
                shutdown_ = true;
                cond_var_.notify_all();
            }
            // Wait for all threads to stop
            for (auto& thread : threads_)
                thread.join();
        }
        void ExecuteJob(std::function<void()> func) {
            // Place a job on the queue and unblock a thread
            std::unique_lock<std::mutex> lock(mutex_);
            decode_jobs_queue_.emplace(std::move(func));
            cond_var_.notify_one();
        }
    protected:
        void ThreadEntry(int i) {
            std::function<void()> execute_decode_job;
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cond_var_.wait(lock, [&] {return shutdown_ || !decode_jobs_queue_.empty();});
                    if (decode_jobs_queue_.empty()) {
                        // No jobs to do; shutting down
                        return;
                    }

                    execute_decode_job = std::move(decode_jobs_queue_.front());
                    decode_jobs_queue_.pop();
                }
                // Execute the decode job without holding any locks
                execute_decode_job();
            }
        }
        std::mutex mutex_;
        std::condition_variable cond_var_;
        bool shutdown_;
        std::queue<std::function<void()>> decode_jobs_queue_;
        std::vector<std::thread> threads_;
};

#endif //ROC_JPEG_SAMPLES_COMMON