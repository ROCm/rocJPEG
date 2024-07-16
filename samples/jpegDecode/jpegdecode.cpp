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

#include "../rocjpeg_samples_utils.h"

int main(int argc, char **argv) {
    int device_id = 0;
    bool save_images = false;
    uint8_t num_components;
    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t prior_channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t num_channels = 0;
    int total_images = 0;
    double time_per_image_all = 0;
    double mpixels_all = 0;
    double images_per_sec = 0;
    std::string chroma_sub_sampling = "";
    std::string input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    bool is_dir = false;
    bool is_file = false;
    RocJpegChromaSubsampling subsampling;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    RocJpegHandle rocjpeg_handle = nullptr;
    RocJpegStreamHandle rocjpeg_stream_handle = nullptr;
    RocJpegImage output_image = {};
    RocJpegDecodeParams decode_params = {};
    RocJpegUtils rocjpeg_utils;

    RocJpegUtils::ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, decode_params, nullptr, nullptr, argc, argv);
    if (!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!RocJpegUtils::InitHipDevice(device_id)) {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegStreamCreate(&rocjpeg_stream_handle));

    std::vector<char> file_data;
    for (auto file_path : file_paths) {
        std::string base_file_name = file_path.substr(file_path.find_last_of("/\\") + 1);
        int image_count = 0;

        // Read an image from disk.
        std::ifstream input(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open())) {
            std::cerr << "ERROR: Cannot open image: " << file_path << std::endl;
            return EXIT_FAILURE;
        }
        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (file_data.size() < file_size) {
            file_data.resize(file_size);
        }
        if (!input.read(file_data.data(), file_size)) {
            std::cerr << "ERROR: Cannot read from file: " << file_path << std::endl;
            return EXIT_FAILURE;
        }

        CHECK_ROCJPEG(rocJpegStreamParse(reinterpret_cast<uint8_t*>(file_data.data()), file_size, rocjpeg_stream_handle));
        CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, rocjpeg_stream_handle, &num_components, &subsampling, widths, heights));

        rocjpeg_utils.GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
        std::cout << "Input file name: " << base_file_name << std::endl;
        std::cout << "Input image resolution: " << widths[0] << "x" << heights[0] << std::endl;
        std::cout << "Chroma subsampling: " + chroma_sub_sampling  << std::endl;
        if (subsampling == ROCJPEG_CSS_411) {
            std::cerr << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
            if (is_dir) {
                std::cout << std::endl;
                continue;
            } else
                return EXIT_FAILURE;
        }

        if (rocjpeg_utils.GetChannelPitchAndSizes(decode_params, subsampling, widths, heights, num_channels, output_image, channel_sizes)) {
            std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
            return EXIT_FAILURE;
        }

        // allocate memory for each channel and reuse them if the sizes remain unchanged for a new image.
        for (int i = 0; i < num_channels; i++) {
            if (prior_channel_sizes[i] != channel_sizes[i]) {
                if (output_image.channel[i] != nullptr) {
                    CHECK_HIP(hipFree((void *)output_image.channel[i]));
                    output_image.channel[i] = nullptr;
                }
                CHECK_HIP(hipMalloc(&output_image.channel[i], channel_sizes[i]));
            }
        }

        std::cout << "Decoding started, please wait! ... " << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        CHECK_ROCJPEG(rocJpegDecode(rocjpeg_handle, rocjpeg_stream_handle, &decode_params, &output_image));
        auto end_time = std::chrono::high_resolution_clock::now();
        double time_per_image_in_milli_sec = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double image_size_in_mpixels = (static_cast<double>(widths[0]) * static_cast<double>(heights[0]) / 1000000);
        image_count++;

        if (save_images) {
            std::string image_save_path = output_file_path;
            if (is_dir) {
                rocjpeg_utils.GetOutputFileExt(decode_params.output_format, base_file_name, widths[0], heights[0], subsampling, image_save_path);
            }
            rocjpeg_utils.SaveImage(image_save_path, &output_image, widths[0], heights[0], subsampling, decode_params.output_format);
        }

        std::cout << "Average processing time per image (ms): " << time_per_image_in_milli_sec << std::endl;
        std::cout << "Average images per sec: " << 1000 / time_per_image_in_milli_sec << std::endl;

        if (is_dir) {
            std::cout << std::endl;
            total_images += image_count;
            time_per_image_all += time_per_image_in_milli_sec;
            mpixels_all += image_size_in_mpixels;
        }
        for (int i = 0; i < ROCJPEG_MAX_COMPONENT; i++) {
            prior_channel_sizes[i] = channel_sizes[i];
        }
    }

    for (int i = 0; i < num_channels; i++) {
        if (output_image.channel[i] != nullptr) {
            CHECK_HIP(hipFree((void *)output_image.channel[i]));
            output_image.channel[i] = nullptr;
        }
    }

    if (is_dir) {
        time_per_image_all = time_per_image_all / total_images;
        images_per_sec = 1000 / time_per_image_all;
        double mpixels_per_sec = mpixels_all * images_per_sec / total_images;
        std::cout << "Total decoded images: " << total_images << std::endl;
        if (total_images) {
            std::cout << "Average processing time per image (ms): " << time_per_image_all << std::endl;
            std::cout << "Average decoded images per sec (Images/Sec): " << images_per_sec << std::endl;
            std::cout << "Average decoded images size (Mpixels/Sec): " << mpixels_per_sec << std::endl;
        }
        std::cout << std::endl;
    }

    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegStreamDestroy(rocjpeg_stream_handle));
    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}