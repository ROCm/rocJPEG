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

/**
 * @brief Decodes a batch of JPEG images and optionally saves the decoded images.
 *
 * @param file_paths A vector of file paths to the JPEG images to be decoded.
 * @param rocjpeg_handle The handle to the RocJpeg library.
 * @param rocjpeg_stream_handles A vector of stream handles for the JPEG images.
 * @param rocjpeg_utils Utility functions for RocJpeg operations.
 * @param decode_params Parameters for decoding the JPEG images.
 * @param save_images A boolean flag indicating whether to save the decoded images.
 * @param output_file_path The file path where the decoded images will be saved.
 * @param num_decoded_images A pointer to a variable that will store the number of successfully decoded images.
 * @param image_size_in_mpixels_per_sec_all A pointer to a variable that will store the decoding speed in megapixels per second.
 * @param images_per_sec A pointer to a variable that will store the number of images decoded per second.
 * @param num_bad_jpegs A pointer to a variable that will store the number of bad JPEG images.
 * @param num_jpegs_with_411_subsampling A pointer to a variable that will store the number of JPEG images with 4:1:1 subsampling.
 * @param num_jpegs_with_unknown_subsampling A pointer to a variable that will store the number of JPEG images with unknown subsampling.
 * @param num_jpegs_with_unsupported_resolution A pointer to a variable that will store the number of JPEG images with unsupported resolution.
 * @param batch_size The number of images to be processed in each batch.
 */
void DecodeImages(std::vector<std::string>& file_paths, RocJpegHandle rocjpeg_handle, std::vector<RocJpegStreamHandle>& rocjpeg_stream_handles, RocJpegUtils rocjpeg_utils,
    RocJpegDecodeParams &decode_params, bool save_images, std::string &output_file_path, uint64_t *num_decoded_images, double *image_size_in_mpixels_per_sec_all, double *images_per_sec,
    uint64_t *num_bad_jpegs, uint64_t *num_jpegs_with_411_subsampling, uint64_t *num_jpegs_with_unknown_subsampling, uint64_t *num_jpegs_with_unsupported_resolution, int batch_size) {

    bool is_roi_valid = false;
    uint32_t roi_width;
    uint32_t roi_height;
    roi_width = decode_params.crop_rectangle.right - decode_params.crop_rectangle.left;
    roi_height = decode_params.crop_rectangle.bottom - decode_params.crop_rectangle.top;

    uint8_t num_components;
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    std::string chroma_sub_sampling = "";
    uint32_t num_channels = 0;
    std::vector<std::vector<uint32_t>> widths;
    std::vector<std::vector<uint32_t>> heights;
    std::vector<std::vector<uint32_t>> prior_channel_sizes;
    std::vector<std::vector<char>> batch_images;
    std::vector<RocJpegChromaSubsampling> subsamplings;
    std::vector<RocJpegImage> output_images;
    std::vector<std::string> base_file_names;
    std::vector<int> bad_image_indices;
    std::vector<RocJpegStreamHandle> valid_rocjpeg_stream_handles;
    std::vector<RocJpegChromaSubsampling> valid_subsamplings;
    std::vector<std::vector<uint32_t>> valid_widths;
    std::vector<std::vector<uint32_t>> valid_heights;
    std::vector<std::vector<uint32_t>> valid_prior_channel_sizes;
    std::vector<RocJpegImage> valid_output_images;
    std::vector<std::string> valid_base_file_names;
    double image_size_in_mpixels_all = 0;
    double total_decode_time_in_milli_sec = 0;

    batch_images.resize(batch_size);
    output_images.resize(batch_size);
    prior_channel_sizes.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    widths.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    heights.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    subsamplings.resize(batch_size);
    base_file_names.resize(batch_size);
    valid_rocjpeg_stream_handles.resize(batch_size);
    valid_output_images.resize(batch_size);
    valid_prior_channel_sizes.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    valid_widths.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    valid_heights.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
    valid_subsamplings.resize(batch_size);
    valid_base_file_names.resize(batch_size);

    for (int i = 0; i < file_paths.size(); i += batch_size) {
        int batch_end = std::min(i + batch_size, static_cast<int>(file_paths.size()));
        for (int j = i; j < batch_end; j++) {
            int index = j - i;
            base_file_names[index] = file_paths[j].substr(file_paths[j].find_last_of("/\\") + 1);
            // Read an image from disk.
            std::ifstream input(file_paths[j].c_str(), std::ios::in | std::ios::binary | std::ios::ate);
            if (!(input.is_open())) {
                std::cerr << "ERROR: Cannot open image: " << file_paths[j] << std::endl;
                return;
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
                return;
            }

            RocJpegStatus rocjpeg_status = rocJpegStreamParse(reinterpret_cast<uint8_t*>(batch_images[index].data()), file_size, rocjpeg_stream_handles[index]);
            if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                bad_image_indices.push_back(index);
                *num_bad_jpegs += 1;
                std::cerr << "Skipping decoding input file: " << file_paths[j] << std::endl;
                continue;
            }

            CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, rocjpeg_stream_handles[index], &num_components, &subsamplings[index], widths[index].data(), heights[index].data()));

            if (roi_width > 0 && roi_height > 0 && roi_width <= widths[index][0] && roi_height <= heights[index][0]) {
                is_roi_valid = true; 
            }

            rocjpeg_utils.GetChromaSubsamplingStr(subsamplings[index], chroma_sub_sampling);
            if (widths[index][0] < 64 || heights[index][0] < 64) {
                bad_image_indices.push_back(index);
                *num_jpegs_with_unsupported_resolution += 1;
                continue;
            }

            if (subsamplings[index] == ROCJPEG_CSS_411 || subsamplings[index] == ROCJPEG_CSS_UNKNOWN) {
                bad_image_indices.push_back(index);
                if (subsamplings[index] == ROCJPEG_CSS_411) {
                    *num_jpegs_with_411_subsampling += 1;
                }
                if (subsamplings[index] == ROCJPEG_CSS_UNKNOWN) {
                    *num_jpegs_with_unknown_subsampling += 1;
                }
                continue;
            }

            if (rocjpeg_utils.GetChannelPitchAndSizes(decode_params, subsamplings[index], widths[index].data(), heights[index].data(), num_channels, output_images[index], channel_sizes)) {
                std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
                return;
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
        int current_batch_size = batch_end - i - bad_image_indices.size();

        // Select valid images for decoding
        if (current_batch_size > 0) {
            if (!bad_image_indices.empty()) {
                // Iterate through the batch images and select only the valid ones
                int valid_idx = 0;
                for (int idx = 0; idx < batch_size; idx++) {
                    // Check if the current image index is not in the list of bad image indices
                    if (std::find(bad_image_indices.begin(), bad_image_indices.end(), idx) == bad_image_indices.end()) {
                        // Add the valid image index to the corresponding vectors
                        valid_rocjpeg_stream_handles[valid_idx] = rocjpeg_stream_handles[idx];
                        valid_subsamplings[valid_idx] = subsamplings[idx];
                        valid_widths[valid_idx] = widths[idx];
                        valid_heights[valid_idx] = heights[idx];
                        valid_prior_channel_sizes[valid_idx] = prior_channel_sizes[idx];
                        valid_output_images[valid_idx] = output_images[idx];
                        valid_base_file_names[valid_idx] = base_file_names[idx];
                        valid_idx++;
                    }
                }
            } else {
                // If there are no bad images, select all the batch images
                valid_rocjpeg_stream_handles = rocjpeg_stream_handles;
                valid_subsamplings = subsamplings;
                valid_widths = widths;
                valid_heights = heights;
                valid_prior_channel_sizes = prior_channel_sizes;
                valid_output_images = output_images;
                valid_base_file_names = base_file_names;
            }
        }

        double time_per_batch_in_milli_sec = 0;
        if (current_batch_size > 0) {
            auto start_time = std::chrono::high_resolution_clock::now();
            CHECK_ROCJPEG(rocJpegDecodeBatched(rocjpeg_handle, valid_rocjpeg_stream_handles.data(), current_batch_size, &decode_params, valid_output_images.data()));
            auto end_time = std::chrono::high_resolution_clock::now();
            time_per_batch_in_milli_sec = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }

        double image_size_in_mpixels = 0;
        for (int b = 0; b < current_batch_size; b++) {
            image_size_in_mpixels += (static_cast<double>(valid_widths[b][0]) * static_cast<double>(valid_heights[b][0]) / 1000000);
        }

        *num_decoded_images += current_batch_size;

        if (save_images) {
            for (int b = 0; b < current_batch_size; b++) {
                std::string image_save_path = output_file_path;
                //if ROI is present, need to pass roi_width and roi_height
                uint32_t width = is_roi_valid ? roi_width : valid_widths[b][0];
                uint32_t height = is_roi_valid ? roi_height : valid_heights[b][0];
                rocjpeg_utils.GetOutputFileExt(decode_params.output_format, valid_base_file_names[b], width, height, valid_subsamplings[b], image_save_path);
                rocjpeg_utils.SaveImage(image_save_path, &valid_output_images[b], width, height, valid_subsamplings[b], decode_params.output_format);
            }
        }

        total_decode_time_in_milli_sec += time_per_batch_in_milli_sec;
        image_size_in_mpixels_all += image_size_in_mpixels;

        bad_image_indices.clear();
    }

    double avg_time_per_image = total_decode_time_in_milli_sec / *num_decoded_images;
    *images_per_sec = 1000 / avg_time_per_image;
    *image_size_in_mpixels_per_sec_all = *images_per_sec * (image_size_in_mpixels_all / *num_decoded_images);

    for (auto& it : output_images) {
        for (int i = 0; i < ROCJPEG_MAX_COMPONENT; i++) {
            if (it.channel[i] != nullptr) {
                CHECK_HIP(hipFree((void *)it.channel[i]));
                it.channel[i] = nullptr;
            }
        }
    }
}

int main(int argc, char **argv) {
    int device_id = 0;
    bool save_images = false;
    int num_threads = 1;
    int batch_size = 1;
    bool is_dir = false;
    bool is_file = false;
    std::string input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    std::vector<std::vector<std::string>> jpeg_files_per_thread;
    std::vector<RocJpegHandle> rocjpeg_handles;
    std::vector<std::vector<RocJpegStreamHandle>> rocjpeg_streams;
    std::vector<uint64_t> num_decoded_images_per_thread;
    std::vector<double> image_size_in_mpixels_per_sec_per_thread;
    std::vector<double> images_per_sec_per_thread;
    std::vector<uint64_t> num_bad_jpegs;
    std::vector<uint64_t> num_jpegs_with_411_subsampling;
    std::vector<uint64_t> num_jpegs_with_unknown_subsampling;
    std::vector<uint64_t> num_jpegs_with_unsupported_resolution;
    RocJpegChromaSubsampling subsampling;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    RocJpegDecodeParams decode_params = {};
    RocJpegUtils rocjpeg_utils;

    RocJpegUtils::ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, decode_params, &num_threads, &batch_size, argc, argv);
    if (!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!RocJpegUtils::InitHipDevice(device_id)) {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    if (num_threads > file_paths.size()) {
        num_threads = file_paths.size();
    }

    for (int i = 0; i < num_threads; i++) {
        std::vector<RocJpegStreamHandle> rocjpeg_stream_handles(batch_size);
        RocJpegHandle rocjpeg_handle;
        CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));
        rocjpeg_handles.push_back(std::move(rocjpeg_handle));
        for(auto i = 0; i < batch_size; i++) {
            CHECK_ROCJPEG(rocJpegStreamCreate(&rocjpeg_stream_handles[i]));
        }
        rocjpeg_streams.push_back(std::move(rocjpeg_stream_handles));
    }
    num_decoded_images_per_thread.resize(num_threads, 0);
    image_size_in_mpixels_per_sec_per_thread.resize(num_threads, 0);
    images_per_sec_per_thread.resize(num_threads, 0);
    num_bad_jpegs.resize(num_threads, 0);
    num_jpegs_with_411_subsampling.resize(num_threads, 0);
    num_jpegs_with_unknown_subsampling.resize(num_threads, 0);
    num_jpegs_with_unsupported_resolution.resize(num_threads, 0);
    jpeg_files_per_thread.resize(num_threads);

    ThreadPool thread_pool(num_threads);

    size_t files_per_thread = file_paths.size() / num_threads;
    size_t remaining_files = file_paths.size() % num_threads;
    size_t start_index = 0;
    for (int i = 0; i < num_threads; i++) {
        size_t end_index = start_index + files_per_thread + (i < remaining_files ? 1 : 0);
        jpeg_files_per_thread[i].assign(file_paths.begin() + start_index, file_paths.begin() + end_index);
        start_index = end_index;
    }

    std::cout << "Decoding started with " << num_threads << " threads, please wait!" << std::endl;
    for (int i = 0; i < num_threads; ++i) {
        thread_pool.ExecuteJob(std::bind(DecodeImages, std::ref(jpeg_files_per_thread[i]), rocjpeg_handles[i], std::ref(rocjpeg_streams[i]), rocjpeg_utils, std::ref(decode_params), save_images,
            std::ref(output_file_path), &num_decoded_images_per_thread[i], &image_size_in_mpixels_per_sec_per_thread[i], &images_per_sec_per_thread[i], &num_bad_jpegs[i],
            &num_jpegs_with_411_subsampling[i], &num_jpegs_with_unknown_subsampling[i], &num_jpegs_with_unsupported_resolution[i], batch_size));
    }
    thread_pool.JoinThreads();

    uint64_t total_decoded_images = 0;
    double total_images_per_sec = 0;
    double total_image_size_in_mpixels_per_sec = 0;
    uint64_t total_num_bad_jpegs = 0;
    uint64_t total_num_jpegs_with_411_subsampling = 0;
    uint64_t total_num_jpegs_with_unknown_subsampling = 0;
    uint64_t total_num_jpegs_with_unsupported_resolution = 0;

    for (auto i = 0 ; i < num_threads; i++) {
        total_decoded_images += num_decoded_images_per_thread[i];
        total_image_size_in_mpixels_per_sec += image_size_in_mpixels_per_sec_per_thread[i];
        total_images_per_sec += images_per_sec_per_thread[i];
        total_num_bad_jpegs += num_bad_jpegs[i];
        total_num_jpegs_with_411_subsampling += num_jpegs_with_411_subsampling[i];
        total_num_jpegs_with_unknown_subsampling += num_jpegs_with_unknown_subsampling[i];
        total_num_jpegs_with_unsupported_resolution += num_jpegs_with_unsupported_resolution[i];
    }

    std::cout << "Total decoded images: " << total_decoded_images << std::endl;
    if (total_num_bad_jpegs || total_num_jpegs_with_411_subsampling || total_num_jpegs_with_unknown_subsampling || total_num_jpegs_with_unsupported_resolution) {
        std::cout << "Total skipped images: " << total_num_bad_jpegs + total_num_jpegs_with_411_subsampling + total_num_jpegs_with_unknown_subsampling + total_num_jpegs_with_unsupported_resolution;
        if (total_num_bad_jpegs) {
            std::cout << " ,total images that cannot be parsed: " << total_num_bad_jpegs;
        }
        if (total_num_jpegs_with_411_subsampling) {
            std::cout << " ,total images with YUV 4:1:1 chroam subsampling: " << total_num_jpegs_with_411_subsampling;
        }
        if (total_num_jpegs_with_unknown_subsampling) {
            std::cout << " ,total images with unknwon chroam subsampling: " << total_num_jpegs_with_unknown_subsampling;
        }
        if (total_num_jpegs_with_unsupported_resolution) {
            std::cout << " ,total images with unsupported_resolution: " << total_num_jpegs_with_unsupported_resolution;
        }
        std::cout << std::endl;
    }

    if (total_decoded_images > 0) {
        std::cout << "Average processing time per image (ms): " << 1000 / total_images_per_sec << std::endl;
        std::cout << "Average decoded images per sec (Images/Sec): " << total_images_per_sec << std::endl;
        std::cout << "Average decoded images size (Mpixels/Sec): " << total_image_size_in_mpixels_per_sec << std::endl;
    }

    for (auto& handle : rocjpeg_handles) {
        CHECK_ROCJPEG(rocJpegDestroy(handle));
    }
    for (auto& rocjpecg_stream : rocjpeg_streams) {
        for (auto i = 0; i < batch_size; i++)
            CHECK_ROCJPEG(rocJpegStreamDestroy(rocjpecg_stream[i]));
    }
    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}