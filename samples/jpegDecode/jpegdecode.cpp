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

#include <iostream>
#include <unistd.h>
#include <vector>
#include <string>
#include <chrono>
#include <sys/stat.h>
#include <libgen.h>
#include <filesystem>
#include <fstream>
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
    << "-be Select rocJPEG backend (0 for ROCJPEG_BACKEND_HARDWARE, using VCN hardware-accelarated JPEG decoder, 1 ROCJPEG_BACKEND_HYBRID, using CPU and GPU HIP kernles for JPEG decoding); optional; default: 0" << std::endl
    << "-fmt Select rocJPEG output format for decoding, one of the [unchanged, yuv, y, rgbi]; optional; default: unchanged" << std::endl
    << "-o Output file path or directory - Write decoded images based on the selected outfut format to this file or directory; optional;" << std::endl
    << "-d GPU device id (0 for the first GPU device, 1 for the second GPU device, etc.); optional; default: 0" << std::endl;
    exit(0);
}

void ParseCommandLine(std::string &input_path, std::string &output_file_path, int &dump_output_frames, int &device_id, int &is_output_rgb, RocJpegBackend &rocjpeg_backend, RocJpegOutputFormat &output_format, int argc, char *argv[]) {
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
        if (!strcmp(argv[i], "-c")) {
            if (++i == argc) {
                ShowHelpAndExit("-c");
            }
            is_output_rgb = std::stoi(argv[i]);
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
            if (selected_output_format == "unchanged") {
                output_format = ROCJPEG_OUTPUT_UNCHANGED;
            } else if (selected_output_format == "yuv") {
                output_format = ROCJPEG_OUTPUT_YUV;
            } else if (selected_output_format == "y") {
                output_format = ROCJPEG_OUTPUT_Y;
            } else if (selected_output_format == "rgbi") {
                output_format = ROCJPEG_OUTPUT_RGBI;
            } else {
                ShowHelpAndExit(argv[i]);
            }
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

static inline int align(int value, int alignment) {
   return (value + alignment - 1) & ~(alignment - 1);
}

void SaveImage(std::string output_file_name, RocJpegImage *output_image, uint32_t *img_width, uint32_t *img_height, RocJpegChromaSubsampling subsampling, RocJpegOutputFormat output_format) {

    uint8_t *hst_ptr = nullptr;
    FILE *fp;
    hipError_t hip_status = hipSuccess;

    if (output_image == nullptr || img_width == nullptr || img_height == nullptr || output_image->channel[0] == nullptr ||
        output_image->pitch[0] == 0 || img_height[0] == 0) {
        return;
    }

    uint32_t channel0_size = output_image->pitch[0] * img_height[0];
    uint32_t channel1_size = output_image->pitch[1] * img_height[1];
    uint32_t channel2_size = output_image->pitch[2] * img_height[2];
    uint32_t output_image_size = channel0_size + channel1_size + channel2_size;

    if (hst_ptr == nullptr) {
        hst_ptr = new uint8_t [output_image_size];
    }

    CHECK_HIP(hipMemcpyDtoH((void *)hst_ptr, output_image->channel[0], channel0_size));

    uint8_t *tmp_hst_ptr = hst_ptr;
    fp = fopen(output_file_name.c_str(), "wb");
    if (fp) {
        // write channel0
        if (img_width[0] == output_image->pitch[0]) {
            fwrite(hst_ptr, 1, channel0_size, fp);
        } else {
            uint32_t width = 0;
            switch (output_format) {
                case ROCJPEG_OUTPUT_UNCHANGED:
                    if (subsampling == ROCJPEG_CSS_422) {
                        width = img_width[0] * 2;
                    }
                    break;
                case ROCJPEG_OUTPUT_RGBI:
                    width = img_width[0] * 3;
                    break;
                default:
                    width = img_width[0];
                    break;
            }
            for (int i = 0; i < img_height[0]; i++) {
                fwrite(tmp_hst_ptr, 1, width, fp);
                tmp_hst_ptr += output_image->pitch[0];
            }
        }
        // write channel1
        if (channel1_size != 0) {
            uint8_t *channel1_hst_ptr = hst_ptr + channel0_size;
            CHECK_HIP(hipMemcpyDtoH((void *)channel1_hst_ptr, output_image->channel[1], channel1_size));
            if (img_width[1] == output_image->pitch[1]) {
                fwrite(channel1_hst_ptr, 1, channel1_size, fp);
            } else {
                for (int i = 0; i < img_height[1]; i++) {
                    fwrite(channel1_hst_ptr, 1, img_width[1], fp);
                    channel1_hst_ptr += output_image->pitch[1];
                }
            }
        }
        // write channel2
        if (channel2_size != 0) {
            uint8_t *channel2_hst_ptr = hst_ptr + channel0_size + channel1_size;
            CHECK_HIP(hipMemcpyDtoH((void *)channel2_hst_ptr, output_image->channel[2], channel2_size));
            if (img_width[2] == output_image->pitch[2]) {
                fwrite(channel2_hst_ptr, 1, channel2_size, fp);
            } else {
                for (int i = 0; i < img_height[2]; i++) {
                    fwrite(channel2_hst_ptr, 1, img_width[2], fp);
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

bool GetFilePaths(std::string &input_path, std::vector<std::string> &file_paths, bool &is_dir, bool &isFile) {
    is_dir = std::filesystem::is_directory(input_path);
    isFile = std::filesystem::is_regular_file(input_path);
    if (is_dir) {
        for (const auto &entry : std::filesystem::directory_iterator(input_path))
            file_paths.push_back(entry.path());
    } else if (isFile) {
        file_paths.push_back(input_path);
    } else {
        std::cerr << "ERROR: the input path is not valid!" << std::endl;
        return false;
    }
    return true;
}

bool InitHipDevice(int device_id, int &num_devices, hipDeviceProp_t &hip_dev_prop) {
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
int main(int argc, char **argv) {
    int device_id = 0;
    int is_output_rgb = 0; // 0 for YUV, 1 for RGB
    int dump_output_frames = 0; // 0 no frame dumps, 1 dumps all the frames
    uint8_t num_components;
    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t num_channels = 0;
    RocJpegChromaSubsampling subsampling;
    hipError_t hip_status = hipSuccess;
    int total_images_all = 0;
    double time_per_image_all = 0;
    double m_pixels_all = 0;
    double image_per_sec_all = 0;
    std::string chroma_sub_sampling = "";
    std::string input_path, output_file_path;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    std::vector<std::string> file_paths = {};
    bool is_dir = false;
    bool isFile = false;
    std::string deviceName, gcnArchName, drmNode;
    int num_devices;
    hipDeviceProp_t hip_dev_prop;
    RocJpegHandle rocjpeg_handle;
    RocJpegImage output_image = {};
    RocJpegOutputFormat output_format = ROCJPEG_OUTPUT_UNCHANGED;

    ParseCommandLine(input_path, output_file_path, dump_output_frames, device_id, is_output_rgb, rocjpeg_backend, output_format, argc, argv);
    if (!GetFilePaths(input_path, file_paths, is_dir, isFile)) {
        std::cerr << "Failed to get input file paths!" << std::endl;
        return -1;
    }
    if (!InitHipDevice(device_id, num_devices, hip_dev_prop)) {
        std::cerr << "Failed to initialize HIP!" << std::endl;
        return -1;
    }

    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));

    int counter = 0;
    std::vector<std::vector<char>> file_data(file_paths.size());
    std::vector<size_t> file_sizes(file_paths.size());

    for (auto file_path : file_paths) {
        std::string base_file_name = file_path.substr(file_path.find_last_of("/\\") + 1);
        int image_count = 0;

        // Read an image from disk.
        std::ifstream input(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open())) {
            std::cerr << "ERROR: Cannot open image: " << file_path << std::endl;
            return 0;
        }
        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (file_data[counter].size() < file_size) {
            file_data[counter].resize(file_size);
        }
        if (!input.read(file_data[counter].data(), file_size)) {
            std::cerr << "Cannot read from file: " << file_path << std::endl;
            return 0;
        }
        file_sizes[counter] = file_size;

        CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, reinterpret_cast<uint8_t*>(file_data[counter].data()), file_size, &num_components, &subsampling, widths, heights));

        std::cout << "info: input file name: " << base_file_name << std::endl;
        std::cout << "info: input image resolution: " << widths[0] << "x" << heights[0] << std::endl;

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
                std::cout << "info: Unknown chroma subsampling" << std::endl;
                return EXIT_FAILURE;
        }
        std::cout << "info: chroma subsampling: " + chroma_sub_sampling  << std::endl;

        if (subsampling == ROCJPEG_CSS_440 || subsampling == ROCJPEG_CSS_411) {
            std::cout << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
            if (is_dir) {
                std::cout << std::endl;
                    continue;
            } else
                return EXIT_FAILURE;
        }

        switch (output_format) {
            case ROCJPEG_OUTPUT_UNCHANGED:
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
            case ROCJPEG_OUTPUT_YUV:
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
            case ROCJPEG_OUTPUT_RGBI:
                num_channels = 1;
                output_image.pitch[0] = widths[0] * 3;
                channel_sizes[0] = output_image.pitch[0] * heights[0];
                break;
            default:
                std::cout << "Unknown output format!" << std::endl;
                return EXIT_FAILURE;
        }
        // allocate memory for each channel
        for (int i = 0; i < num_channels; i++) {
            CHECK_HIP(hipMalloc(&output_image.channel[i], channel_sizes[i]));
        }

        std::cout << "info: decoding started, please wait! ... " << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        CHECK_ROCJPEG(rocJpegDecode(rocjpeg_handle, reinterpret_cast<uint8_t*>(file_data[counter].data()), file_size, output_format, &output_image));
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> decoder_time = end_time - start_time;
        double time_per_image = decoder_time.count() * 1000;
        double ips = (1 / time_per_image) * 1000;
        double mpixels = ((double)widths[0] * (double)heights[0] / 1000000) * ips;
        image_count++;

        if (dump_output_frames) {
            std::string::size_type const p(base_file_name.find_last_of('.'));
            std::string file_name_no_ext = base_file_name.substr(0, p);
            std::string file_name_for_saving = output_file_path + "//" + file_name_no_ext + "_" + std::to_string(widths[0]) + "x"
                + std::to_string(heights[0]) + "." + (is_output_rgb ? "rgbi" : "yuv");
            std::string image_save_path = is_dir ? file_name_for_saving : output_file_path;
            SaveImage(image_save_path, &output_image, widths, heights, subsampling, output_format);
        }

        for (int i = 0; i < num_channels; i++) {
            CHECK_HIP(hipFree((void*)output_image.channel[i]));
        }

        std::cout << "info: total decoded images: " << image_count << std::endl;
        std::cout << "info: average processing time per image (ms): " << time_per_image << std::endl;
        std::cout << "info: average images per sec: " << (1 / time_per_image) * 1000 << std::endl;
        std::cout << "info: total elapsed time (s): " << decoder_time.count() << std::endl;

        if (is_dir) {
            std::cout << std::endl;
            total_images_all += image_count;
            time_per_image_all += time_per_image;
            image_per_sec_all += ips;
            m_pixels_all += mpixels;
        }
        counter++;
    }

    if (is_dir) {
        std::cout << "info: total decoded images: " << total_images_all << std::endl;
        if (total_images_all) {
            std::cout << "info: average processing time per image (ms): " << time_per_image_all / total_images_all << std::endl;
            std::cout << "info: average decoded images per sec: " << image_per_sec_all / total_images_all << std::endl;
            std::cout << "info: average decoded mpixels per sec: " << m_pixels_all / total_images_all << std::endl;
        }
        std::cout << std::endl;
    }

    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle));
    std::cout << "info: decoding completed!" << std::endl;

    return 0;
}