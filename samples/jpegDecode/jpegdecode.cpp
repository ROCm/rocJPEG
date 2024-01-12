/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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
        std::cout << "rocJPEG failure: '#" << rocjpeg_status << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
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
    << "-i Input File Path - required" << std::endl
    << "-o Output File Path - dumps output if requested; optional" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0" << std::endl;
    exit(0);
}

static inline int align(int value, int alignment) {
   return (value + alignment - 1) & ~(alignment - 1);
}

// TODO - change the dev_mem type from void* to RocJpegImage
void SaveImage(std::string output_file_name, void* dev_mem, size_t output_image_size, uint32_t img_width, uint32_t img_height, uint32_t output_image_stride,
    RocJpegChromaSubsampling subsampling, bool is_output_rgb) {

    uint8_t *hst_ptr = nullptr;
    FILE *fp;
    if (hst_ptr == nullptr) {
        hst_ptr = new uint8_t [output_image_size];
    }
    hipError_t hip_status = hipSuccess;
    CHECK_HIP(hipMemcpyDtoH((void *)hst_ptr, dev_mem, output_image_size));

    // no RGB dump if the surface type is YUV400
    if (subsampling == ROCJPEG_CSS_GRAY && is_output_rgb) {
        return;
    }
    uint8_t *tmp_hst_ptr = hst_ptr;
    fp = fopen(output_file_name.c_str(), "wb");
    if (fp) {
        if (img_width == output_image_stride && img_height == align(img_height, 16)) {
            fwrite(hst_ptr, 1, output_image_size, fp);
        } else {
            uint32_t width = is_output_rgb ? img_width * 3 : img_width;
            for (int i = 0; i < img_height; i++) {
                fwrite(tmp_hst_ptr, 1, width, fp);
                tmp_hst_ptr += output_image_stride;
            }
            if (!is_output_rgb) {
                // dump chroma
                uint8_t *uv_hst_ptr = hst_ptr + output_image_stride * align(img_height, 16);
                for (int i = 0; i < img_height >> 1; i++) {
                    fwrite(uv_hst_ptr, 1, width, fp);
                    uv_hst_ptr += output_image_stride;
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

int main(int argc, char **argv) {
    int device_id = 0;
    int is_output_rgb = 0; // 0 for YUV, 1 for RGB
    int dump_output_frames = 0; // 0 no frame dumps, 1 dumps all the frames
    int scaling_width = 0;
    int scaling_height = 0;
    uint8_t num_components;
    uint32_t widths, heights;
    RocJpegChromaSubsampling subsampling;
    hipError_t hip_status = hipSuccess;
    size_t yuv_image_size, rgb_image_size;
    uint32_t yuv_image_stride, rgb_image_stride;
    int total_images_all = 0;
    double time_per_image_all = 0;
    double m_pixels_all = 0;
    double image_per_sec_all = 0;
    std::string chroma_sub_sampling = "";
    std::string path, output_file_path;

    // Parse command-line arguments
    if(argc < 1) {
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
            path = argv[i];
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
        ShowHelpAndExit(argv[i]);
    }

    bool isScaling = (scaling_width > 0 && scaling_height > 0) ? true : false;

    std::vector<std::string> file_paths = {};

    bool is_dir = std::filesystem::is_directory(path);
    bool isFile = std::filesystem::is_regular_file(path);

    if (is_dir) {
        for (const auto &entry : std::filesystem::directory_iterator(path))
            file_paths.push_back(entry.path());
    } else if (isFile) {
        file_paths.push_back(path);
    } else {
        std::cout << "ERROR: the input path is not valid !" << std::endl;
        return -1;
    }

    std::string deviceName, gcnArchName, drmNode;
    int pciBusID, pciDomainID, pciDeviceID;
    int num_devices;
    hipDeviceProp_t hip_dev_prop;
    hipStream_t hip_stream;

    CHECK_HIP(hipGetDeviceCount(&num_devices));
    if (num_devices < 1) {
        std::cout << "ERROR: didn't find any GPU!" << std::endl;
        return -1;
    }
    if (device_id >= num_devices) {
        std::cout << "ERROR: the requested device_id is not found! " << std::endl;
        return -1;
    }
    CHECK_HIP(hipSetDevice(device_id));
    CHECK_HIP(hipGetDeviceProperties(&hip_dev_prop, device_id));
    CHECK_HIP(hipStreamCreate(&hip_stream));

    std::cout << "info: Using GPU device " << device_id << ": " << hip_dev_prop.name << "[" << hip_dev_prop.gcnArchName << "] on PCI bus " <<
    std::setfill('0') << std::setw(2) << std::right << std::hex << hip_dev_prop.pciBusID << ":" << std::setfill('0') << std::setw(2) <<
    std::right << std::hex << hip_dev_prop.pciDomainID << "." << hip_dev_prop.pciDeviceID << std::dec << std::endl;

    RocJpegHandle rocjpeg_handle;
    CHECK_ROCJPEG(rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, 0, &rocjpeg_handle));

    int counter = 0;
    std::vector<std::vector<char>> file_data(file_paths.size());
    std::vector<size_t> file_sizes(file_paths.size());


    RocJpegImage output_image = {};
    RocJpegOutputFormat output_format = ROCJPEG_OUTPUT_YUV;
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

        CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, reinterpret_cast<uint8_t*>(file_data[counter].data()), file_size, &num_components, &subsampling, &widths, &heights));

        std::cout << "info: input file name: " << base_file_name << std::endl;
        std::cout << "info: input image resolution: " << widths << "x" << heights << std::endl;
        switch (subsampling) {
            case ROCJPEG_CSS_444:
                chroma_sub_sampling = "YUV 4:4:4";
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
            case ROCJPEG_CSS_GRAY:
                chroma_sub_sampling = "YUV 4:0:0";
                break;
            case ROCJPEG_CSS_UNKNOWN:
                std::cout << "info: Unknown chroma subsampling" << std::endl;
                return EXIT_FAILURE;
        }
        std::cout << "info: "+ chroma_sub_sampling + " chroma subsampling" << std::endl;
        // fix the yuv_image_size calculation
        yuv_image_size = align(widths, 256) * (align(heights, 16) + (align(heights, 16) >> 1));
        output_image.pitch[0] = align(widths, 256);

        hip_status = hipMalloc(&output_image.channel[0], yuv_image_size);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipMalloc failed to allocate the device memory for the output!" << hip_status << std::endl;
            return 0;
        }

        std::cout << "info: decoding started, please wait! ... " << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        CHECK_ROCJPEG(rocJpegDecode(rocjpeg_handle, reinterpret_cast<uint8_t*>(file_data[counter].data()), file_size, output_format, &output_image, hip_stream));
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> decoder_time = end_time - start_time;
        double time_per_image = decoder_time.count() * 1000;
        double ips = (1 / time_per_image) * 1000;
        double mpixels = ((double)widths * (double)heights / 1000000) * ips;
        image_count++;

        if (dump_output_frames) {
            std::string::size_type const p(base_file_name.find_last_of('.'));
            std::string file_name_no_ext = base_file_name.substr(0, p);
            std::string fileName = output_file_path + "//" + file_name_no_ext + "_" + std::to_string(widths) + "x"
                + std::to_string(heights) + "." + (is_output_rgb ? "RGB" : "YUV");

            SaveImage(output_file_path, output_image.channel[0], is_output_rgb ? rgb_image_size : yuv_image_size, widths, heights,
                output_image.pitch[0] , subsampling, is_output_rgb);
        }

        if (output_image.channel[0] != nullptr) {
            hip_status = hipFree((void *)output_image.channel[0]);
            if (hip_status != hipSuccess) {
                std::cout << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
                return -1;
            }
        }

        std::cout << "info: total decoded images: " << image_count << std::endl;
        if (is_output_rgb == 0) {
            std::cout << "info: output image format: " << chroma_sub_sampling << std::endl;
        } else {
            if (subsampling != ROCJPEG_CSS_GRAY)
                std::cout << "info: output frame format: " << "RGB" << std::endl;
            else
                std::cout << "info: output frame format: " << chroma_sub_sampling << std::endl;
        }
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
        std::cout << "info: average processing time per image (ms): " << time_per_image_all / total_images_all << std::endl;
        std::cout << "info: average decoded images per sec: " << image_per_sec_all / total_images_all << std::endl;
        std::cout << "info: average decoded mpixels per sec: " << m_pixels_all / total_images_all << std::endl;
        std::cout << std::endl;
    }

    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle));
    std::cout << "info: decoding completed!" << std::endl;

    return 0;
}