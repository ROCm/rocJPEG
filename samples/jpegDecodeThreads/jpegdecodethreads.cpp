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

#include "../rocjpeg_samples_common.h"

void ThreadFunction(std::string &file_path, RocJpegHandle rocjpeg_handle, RocJpegImage &output_image, std::mutex& mutex, RocJpegOutputFormat output_format, int save_images, std::string &output_file_path,
                    uint64_t *num_decoded_images, double *image_size_in_mpixels, std::atomic_bool &decoding_complete) {
    std::vector<char> file_data;
    uint8_t num_components;
    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    RocJpegChromaSubsampling subsampling;
    std::string chroma_sub_sampling = "";
    uint32_t num_channels = 0;

    std::string base_file_name = file_path.substr(file_path.find_last_of("/\\") + 1);
    // Read an image from disk.
    std::ifstream input(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
        std::cerr << "ERROR: Cannot open image: " << file_path << std::endl;
        return;
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
        return;
    }

    CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, reinterpret_cast<uint8_t *>(file_data.data()), file_size, &num_components, &subsampling, widths, heights));
    if (subsampling == ROCJPEG_CSS_440 || subsampling == ROCJPEG_CSS_411) {
        std::cout << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
        std::cout << "Skipping decoding file " << base_file_name << std::endl;
        return;
    }

    if (GetChannelPitchAndSizes(output_format, subsampling, widths, heights, num_channels, output_image, channel_sizes)) {
            std::cerr << "Failed to get the channel pitch and sizes" << std::endl;
            return;
    }

    // allocate memory for each channel
    for (int i = 0; i < num_channels; i++) {
        if (output_image.channel[i] != nullptr) {
            CHECK_HIP(hipFree((void*)output_image.channel[i]));
            output_image.channel[i] = nullptr;
        }
        CHECK_HIP(hipMalloc(&output_image.channel[i], channel_sizes[i]));
    }

    CHECK_ROCJPEG(rocJpegDecode(rocjpeg_handle, reinterpret_cast<uint8_t *>(file_data.data()), file_size, output_format, &output_image));
    *image_size_in_mpixels += (static_cast<double>(widths[0]) * static_cast<double>(heights[0]) / 1000000);
    *num_decoded_images += 1;

    if (save_images) {
        std::string::size_type const p(base_file_name.find_last_of('.'));
        std::string file_name_no_ext = base_file_name.substr(0, p);
        std::string file_extension;
        GetFileExtForSaving(output_format, file_extension);
        std::string file_name_for_saving = output_file_path + "//" + file_name_no_ext + "_" + std::to_string(widths[0]) + "x"
            + std::to_string(heights[0]) + "." + file_extension;
        SaveImage(file_name_for_saving, &output_image, widths[0], heights[0], subsampling, output_format);
    }
    decoding_complete = true;
}

int main(int argc, char **argv) {
    int device_id = 0;
    int save_images = 0;
    int num_threads = 2;
    int total_images_all = 0;
    double image_per_sec_all = 0;
    std::string input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    bool is_dir = false;
    bool is_file = false;
    RocJpegChromaSubsampling subsampling;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    RocJpegOutputFormat output_format = ROCJPEG_OUTPUT_NATIVE;
    std::vector<RocJpegHandle> rocjpeg_handles;
    std::mutex mutex;
    std::vector<uint64_t> num_decoded_images_per_thread;
    std::vector<double> image_size_in_mpixels_per_thread;
    std::vector<RocJpegImage> rocjpeg_images;
    struct DecodingStatus {
        std::atomic<bool> decoding_complete;
        DecodingStatus () : decoding_complete(false) {};
    };
    std::vector<std::unique_ptr<DecodingStatus>> decoding_status_per_thread;

    ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, output_format, &num_threads, argc, argv);
    if (!GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "Failed to get input file paths!" << std::endl;
        return -1;
    }
    if (!InitHipDevice(device_id)) {
        std::cerr << "Failed to initialize HIP!" << std::endl;
        return -1;
    }

    ThreadPool thread_pool(num_threads);

    if (num_threads > file_paths.size()) {
        num_threads = file_paths.size();
    }

    std::cout << "info: creating decoder objects, please wait!" << std::endl;
    for (int i = 0; i < num_threads; i++) {
        RocJpegHandle rocjpeg_handle;
        CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));
        rocjpeg_handles.push_back(std::move(rocjpeg_handle));
    }
    num_decoded_images_per_thread.resize(num_threads, 0);
    image_size_in_mpixels_per_thread.resize(num_threads, 0);
    rocjpeg_images.resize(num_threads, {0});
    for (auto i = 0; i < num_threads; i++) {
        decoding_status_per_thread.emplace_back(std::make_unique<DecodingStatus>());
    }
    std::cout << "info: decoding started with " << num_threads << " threads, please wait!" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < file_paths.size(); i++) {
        int thread_idx = i % num_threads;
        if (i >= num_threads) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                while (!decoding_status_per_thread[thread_idx]->decoding_complete);
                decoding_status_per_thread[thread_idx]->decoding_complete = false;
            }
        }
        thread_pool.ExecuteJob(std::bind(ThreadFunction, file_paths[i], rocjpeg_handles[thread_idx], rocjpeg_images[thread_idx], std::ref(mutex), output_format, save_images, std::ref(output_file_path),
            &num_decoded_images_per_thread[thread_idx], &image_size_in_mpixels_per_thread[thread_idx], std::ref(decoding_status_per_thread[thread_idx]->decoding_complete)));
    }

    // Wait for all threads to finish
    thread_pool.JoinThreads();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time_in_milli_sec = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    uint64_t total_decoded_images = 0;
    double total_image_size_in_mpixels = 0;
    for (auto i = 0 ; i < num_threads; i++) {
        total_decoded_images += num_decoded_images_per_thread[i];
        total_image_size_in_mpixels += image_size_in_mpixels_per_thread[i];
        for (int j = 0; j < ROCJPEG_MAX_COMPONENT; j++) {
            if (rocjpeg_images[i].channel[j] != nullptr) {
                CHECK_HIP(hipFree((void *)rocjpeg_images[i].channel[j]));
                rocjpeg_images[i].channel[j] = nullptr;
            }
        }
    }

    double average_decoding_time_in_milli_sec = total_time_in_milli_sec / total_decoded_images;
    double avg_images_per_sec = 1000 / average_decoding_time_in_milli_sec;
    double avg_image_size_in_mpixels_per_sec = total_image_size_in_mpixels * avg_images_per_sec / total_decoded_images;
    std::cout << "info: Total elapsed time (ms): " << total_time_in_milli_sec << std::endl;
    std::cout << "info: total decoded images: " << total_decoded_images << std::endl;
    std::cout << "info: average processing time per image (ms): " << average_decoding_time_in_milli_sec << std::endl;
    std::cout << "info: average decoded images per sec (Images/Sec): " << avg_images_per_sec << std::endl;
    std::cout << "info: average decoded images size (Mpixels/Sec): " << avg_image_size_in_mpixels_per_sec << std::endl;

    for (auto& handle : rocjpeg_handles) {
        CHECK_ROCJPEG(rocJpegDestroy(handle));
    }
    std::cout << "info: decoding completed!" << std::endl;

    return 0;
}