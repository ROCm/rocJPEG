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
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
#else
    #include <experimental/filesystem>
#endif
#include <chrono>
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

/**
 * @class RocJpegUtils
 * @brief Utility class for rocJPEG samples.
 *
 * This class provides utility functions for rocJPEG samples, such as parsing command line arguments,
 * getting file paths, initializing HIP device, getting chroma subsampling string, getting channel pitch and sizes,
 * getting output file extension, and saving images.
 */
class RocJpegUtils {
public:
    /**
     * @brief Parses the command line arguments.
     *
     * This function parses the command line arguments and sets the corresponding variables.
     *
     * @param input_path The input path.
     * @param output_file_path The output file path.
     * @param save_images Flag indicating whether to save images.
     * @param device_id The device ID.
     * @param rocjpeg_backend The rocJPEG backend.
     * @param decode_params The rocJPEG decode parameters.
     * @param num_threads The number of threads.
     * @param argc The number of command line arguments.
     * @param argv The command line arguments.
     */
    static void ParseCommandLine(std::string &input_path, std::string &output_file_path, bool &save_images, int &device_id,
                                 RocJpegBackend &rocjpeg_backend, RocJpegDecodeParams &decode_params, int *num_threads, int argc, char *argv[]) {
        if(argc <= 1) {
            ShowHelpAndExit("", num_threads != nullptr);
        }
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "-h")) {
                ShowHelpAndExit("", num_threads != nullptr);
            }
            if (!strcmp(argv[i], "-i")) {
                if (++i == argc) {
                    ShowHelpAndExit("-i", num_threads != nullptr);
                }
                input_path = argv[i];
                continue;
            }
            if (!strcmp(argv[i], "-o")) {
                if (++i == argc) {
                    ShowHelpAndExit("-o", num_threads != nullptr);
                }
                output_file_path = argv[i];
                save_images = true;
                continue;
            }
            if (!strcmp(argv[i], "-d")) {
                if (++i == argc) {
                    ShowHelpAndExit("-d", num_threads != nullptr);
                }
                device_id = atoi(argv[i]);
                continue;
            }
            if (!strcmp(argv[i], "-be")) {
                if (++i == argc) {
                    ShowHelpAndExit("-be", num_threads != nullptr);
                }
                rocjpeg_backend = static_cast<RocJpegBackend>(atoi(argv[i]));
                continue;
            }
            if (!strcmp(argv[i], "-fmt")) {
                if (++i == argc) {
                    ShowHelpAndExit("-fmt", num_threads != nullptr);
                }
                std::string selected_output_format = argv[i];
                if (selected_output_format == "native") {
                    decode_params.output_format = ROCJPEG_OUTPUT_NATIVE;
                } else if (selected_output_format == "yuv") {
                    decode_params.output_format = ROCJPEG_OUTPUT_YUV_PLANAR;
                } else if (selected_output_format == "y") {
                    decode_params.output_format = ROCJPEG_OUTPUT_Y;
                } else if (selected_output_format == "rgb") {
                    decode_params.output_format = ROCJPEG_OUTPUT_RGB;
                } else if (selected_output_format == "rgb_planar") {
                    decode_params.output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
                } else {
                    ShowHelpAndExit(argv[i], num_threads != nullptr);
                }
                continue;
            }
            if (!strcmp(argv[i], "-t")) {
                if (++i == argc) {
                    ShowHelpAndExit("-t", num_threads != nullptr);
                }
                if (num_threads != nullptr)
                    *num_threads = atoi(argv[i]);
                continue;
            }
            ShowHelpAndExit(argv[i], num_threads != nullptr);
        }
    }

    /**
     * @brief Gets the file paths.
     *
     * This function gets the file paths based on the input path and sets the corresponding variables.
     *
     * @param input_path The input path.
     * @param file_paths The vector to store the file paths.
     * @param is_dir Flag indicating whether the input path is a directory.
     * @param is_file Flag indicating whether the input path is a file.
     * @return True if successful, false otherwise.
     */
    static bool GetFilePaths(std::string &input_path, std::vector<std::string> &file_paths, bool &is_dir, bool &is_file) {
    #if __cplusplus >= 201703L && __has_include(<filesystem>)
        is_dir = std::filesystem::is_directory(input_path);
        is_file = std::filesystem::is_regular_file(input_path);
    #else
        is_dir = std::experimental::filesystem::is_directory(input_path);
        is_file = std::experimental::filesystem::is_regular_file(input_path);
    #endif
        if (is_dir) {
    #if __cplusplus >= 201703L && __has_include(<filesystem>)
            for (const auto &entry : std::filesystem::directory_iterator(input_path))
    #else
            for (const auto &entry : std::experimental::filesystem::directory_iterator(input_path))
    #endif
                file_paths.push_back(entry.path());
        } else if (is_file) {
            file_paths.push_back(input_path);
        } else {
            std::cerr << "ERROR: the input path is not valid!" << std::endl;
            return false;
        }
        return true;
    }

    /**
     * @brief Initializes the HIP device.
     *
     * This function initializes the HIP device with the specified device ID.
     *
     * @param device_id The device ID.
     * @return True if successful, false otherwise.
     */
    static bool InitHipDevice(int device_id) {
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

        std::cout << "Using GPU device " << device_id << ": " << hip_dev_prop.name << "[" << hip_dev_prop.gcnArchName << "] on PCI bus " <<
        std::setfill('0') << std::setw(2) << std::right << std::hex << hip_dev_prop.pciBusID << ":" << std::setfill('0') << std::setw(2) <<
        std::right << std::hex << hip_dev_prop.pciDomainID << "." << hip_dev_prop.pciDeviceID << std::dec << std::endl;

        return true;
    }

    /**
     * @brief Gets the chroma subsampling string.
     *
     * This function gets the chroma subsampling string based on the specified subsampling value.
     *
     * @param subsampling The chroma subsampling value.
     * @param chroma_sub_sampling The string to store the chroma subsampling.
     */
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

    /**
     * @brief Gets the channel pitch and sizes.
     *
     * This function gets the channel pitch and sizes based on the specified output format, chroma subsampling,
     * output image, and channel sizes.
     *
     * @param output_format The output format.
     * @param subsampling The chroma subsampling.
     * @param widths The array to store the channel widths.
     * @param heights The array to store the channel heights.
     * @param num_channels The number of channels.
     * @param output_image The output image.
     * @param channel_sizes The array to store the channel sizes.
     * @return The channel pitch.
     */
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

    /**
     * @brief Gets the output file extension.
     *
     * This function gets the output file extension based on the specified output format, base file name,
     * image width, image height, and file name for saving.
     *
     * @param output_format The output format.
     * @param base_file_name The base file name.
     * @param image_width The image width.
     * @param image_height The image height.
     * @param file_name_for_saving The string to store the file name for saving.
     */
    void GetOutputFileExt(RocJpegOutputFormat output_format, std::string &base_file_name, uint32_t image_width, uint32_t image_height, std::string &file_name_for_saving) {
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

    /**
     * @brief Saves the image.
     *
     * This function saves the image to the specified output file name based on the output image, image width,
     * image height, chroma subsampling, and output format.
     *
     * @param output_file_name The output file name.
     * @param output_image The output image.
     * @param img_width The image width.
     * @param img_height The image height.
     * @param subsampling The chroma subsampling.
     * @param output_format The output format.
     */
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

private:
    /**
     * @brief Shows the help message and exits.
     *
     * This function shows the help message and exits the program.
     *
     * @param option The option to display in the help message (optional).
     * @param show_threads Flag indicating whether to show the number of threads in the help message.
     */
    static void ShowHelpAndExit(const char *option = nullptr, bool show_threads = false) {
        std::cout  << "Options:\n"
        "-i     [input path] - input path to a single JPEG image or a directory containing JPEG images - [required]\n"
        "-be    [backend] - select rocJPEG backend (0 for hardware-accelerated JPEG decoding using VCN,\n"
        "                                           1 for hybrid JPEG decoding using CPU and GPU HIP kernels (currently not supported)) [optional - default: 0]\n"
        "-fmt   [output format] - select rocJPEG output format for decoding, one of the [native, yuv, y, rgb, rgb_planar] - [optional - default: native]\n"
        "-o     [output path] - path to an output file or a path to a directory - write decoded images to a file or directory based on selected output format - [optional]\n"
        "-d     [device id] - specify the GPU device id for the desired device (use 0 for the first device, 1 for the second device, and so on) [optional - default: 0]\n";
        if (show_threads) {
            std::cout << "-t     [threads] - number of threads for parallel JPEG decoding - [optional - default: 2]\n";
        }
        exit(0);
    }
};
#endif //ROC_JPEG_SAMPLES_COMMON