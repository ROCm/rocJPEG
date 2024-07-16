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
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    std::vector<std::vector<uint32_t>> widths;
    std::vector<std::vector<uint32_t>> heights;
    std::vector<std::vector<uint32_t>> prior_channel_sizes;
    uint32_t num_channels = 0;
    int total_images = 0;
    int batch_size = 2;
    double time_per_image_all = 0;
    double mpixels_all = 0;
    double images_per_sec = 0;
    std::string chroma_sub_sampling = "";
    std::string input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    bool is_dir = false;
    bool is_file = false;
    std::vector<std::vector<char>> batch_images;
    std::vector<RocJpegChromaSubsampling> subsamplings;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    RocJpegHandle rocjpeg_handle = nullptr;
    std::vector<RocJpegStreamHandle> rocjpeg_stream_handles;
    RocJpegImage output_image = {};
    std::vector<RocJpegImage> output_images;
    RocJpegDecodeParams decode_params = {};
    RocJpegUtils rocjpeg_utils;

    RocJpegUtils::ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, decode_params, nullptr, &batch_size, argc, argv);
    if (!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!RocJpegUtils::InitHipDevice(device_id)) {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));
    batch_size = std::min(batch_size, static_cast<int>(file_paths.size()));
    rocjpeg_stream_handles.resize(batch_size);
    // create stream handles of batch size
    for(auto i = 0; i < batch_size; i++) {
        CHECK_ROCJPEG(rocJpegStreamCreate(&rocjpeg_stream_handles[i]));
    }

    batch_images.resize(batch_size);
    output_images.resize(batch_size);
    prior_channel_sizes.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    widths.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    heights.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    subsamplings.resize(batch_size);
    std::vector<std::string> base_file_names(batch_size);
    std::cout << "Decoding started, please wait! ... " << std::endl;
    for (int i = 0; i < file_paths.size(); i += batch_size) {
        int batch_end = std::min(i + batch_size, static_cast<int>(file_paths.size()));
        for (int j = i; j < batch_end; j++) {
            int index = j - i;
            base_file_names[index] = file_paths[j].substr(file_paths[j].find_last_of("/\\") + 1);
            // Read an image from disk.
            std::ifstream input(file_paths[j].c_str(), std::ios::in | std::ios::binary | std::ios::ate);
            if (!(input.is_open())) {
                std::cerr << "ERROR: Cannot open image: " << file_paths[j] << std::endl;
                return EXIT_FAILURE;
            }
            // Get the size
            std::streamsize file_size = input.tellg();
            input.seekg(0, std::ios::beg);
            // resize if buffer is too small
            if (batch_images[index].size() < file_size) {
                batch_images[index].resize(file_size);
            }
            if (!input.read(batch_images[index].data(), file_size)) {
                std::cerr << "ERROR: Cannot read from file: " << file_paths[j] << std::endl;
                return EXIT_FAILURE;
            }

            CHECK_ROCJPEG(rocJpegStreamParse(reinterpret_cast<uint8_t*>(batch_images[index].data()), file_size, rocjpeg_stream_handles[index]));
            CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, rocjpeg_stream_handles[index], &num_components, &subsamplings[index], widths[index].data(), heights[index].data()));

            rocjpeg_utils.GetChromaSubsamplingStr(subsamplings[index], chroma_sub_sampling);
            if (subsamplings[index] == ROCJPEG_CSS_411) {
                std::cerr << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
                if (is_dir) {
                    std::cout << std::endl;
                    continue;
                } else
                    return EXIT_FAILURE;
            }

            if (rocjpeg_utils.GetChannelPitchAndSizes(decode_params, subsamplings[index], widths[index].data(), heights[index].data(), num_channels, output_images[index], channel_sizes)) {
                std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
                return EXIT_FAILURE;
            }

            // allocate memory for each channel and reuse them if the sizes remain unchanged for a new image.
            for (int n = 0; n < num_channels; n++) {
                if (prior_channel_sizes[index][n] != channel_sizes[n]) {
                    if (output_images[index].channel[n] != nullptr) {
                        CHECK_HIP(hipFree((void *)output_images[index].channel[n]));
                        output_images[index].channel[n] = nullptr;
                    }
                    CHECK_HIP(hipMalloc(&output_images[index].channel[n], channel_sizes[n]));
                    prior_channel_sizes[index][n] = channel_sizes[n];
                }
            }
        }
        int current_batch_size = batch_end - i;

        auto start_time = std::chrono::high_resolution_clock::now();
        CHECK_ROCJPEG(rocJpegDecodeBatched(rocjpeg_handle, rocjpeg_stream_handles.data(), current_batch_size, &decode_params, output_images.data()));
        auto end_time = std::chrono::high_resolution_clock::now();
        double time_per_batch_in_milli_sec = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        double image_size_in_mpixels = 0;
        for (int b = 0; b < current_batch_size; b++) {
            image_size_in_mpixels += (static_cast<double>(widths[b][0]) * static_cast<double>(heights[b][0]) / 1000000);
        }

        total_images += current_batch_size;

        if (save_images) {
            for (int b = 0; b < current_batch_size; b++) {
                std::string image_save_path = output_file_path;
                if (is_dir) {
                    rocjpeg_utils.GetOutputFileExt(decode_params.output_format, base_file_names[b], widths[b][0], heights[b][0], subsamplings[b], image_save_path);
                }
                rocjpeg_utils.SaveImage(image_save_path, &output_images[b], widths[b][0], heights[b][0], subsamplings[b], decode_params.output_format);
            }
        }

        if (is_dir) {
            time_per_image_all += time_per_batch_in_milli_sec;
            mpixels_all += image_size_in_mpixels;
        }

        // Clear the batch_images vector after processing each batch
        for (int j = i; j < batch_end; j++) {
            batch_images[j - i].clear();
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
    }

    //cleanup
    for (auto& it : output_images) {
        for (int i = 0; i < ROCJPEG_MAX_COMPONENT; i++) {
            if (it.channel[i] != nullptr) {
                CHECK_HIP(hipFree((void *)it.channel[i]));
                it.channel[i] = nullptr;
            }
        }
    }
    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle));
    for(auto& it : rocjpeg_stream_handles) {
        CHECK_ROCJPEG(rocJpegStreamDestroy(it));
    }

    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}