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
#include "rocjpeg.hpp"
#include <queue>
#include <thread>
#include <mutex>

// Thread functionbreak
void ThreadFunction(std::queue<std::string>& jpeg_files, ROCJpegDecode *jpeg_decoder, std::mutex& mutex) {
    while (true) {
        // Get the next JPEG file to process
        std::string filename;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!jpeg_files.empty()) {
                filename = jpeg_files.front();
                jpeg_files.pop();
            }
        }

        if (filename.empty()) {
            // No more files to process
            break;
        }

        jpeg_decoder->Decode(filename, nullptr, 0);
    }
}

void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options:" << std::endl
    << "-i Input File Path - required" << std::endl
    << "-t num of threads - optional; defaults: 4" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0" << std::endl;
    exit(0);
}

int main(int argc, char **argv) {
    int device_id = 0;
    int num_threads = 4;
    std::string path;

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
        if (!strcmp(argv[i], "-d")) {
            if (++i == argc) {
                ShowHelpAndExit("-d");
            }
            device_id = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-t")) {
            if (++i == argc) {
                ShowHelpAndExit("-t");
            }
            num_threads = std::stoi(argv[i]);
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }


    std::vector<std::string> file_paths = {};
    bool is_dir = std::filesystem::is_directory(path);
    bool is_file = std::filesystem::is_regular_file(path);
    std::vector<std::unique_ptr<ROCJpegDecode>> jpeg_decoders;
    std::mutex mutex; // Mutex for thread synchronization
    std::queue<std::string> jpeg_files; // Create a thread-safe queue to store JPEG file paths
    std::string device_name, gcn_arch_name, drm_node;
    int pci_bus_id, pci_domain_id, pci_device_id;

    if (is_dir) {
        for (const auto &entry : std::filesystem::directory_iterator(path)) {
            file_paths.push_back(entry.path());
            jpeg_files.push(entry.path());
        }
    } else if (is_file) {
        file_paths.push_back(path);
        jpeg_files.push(path);
    } else {
        std::cout << "ERROR: the input path is not valid !" << std::endl;
        return -1;
    }

    int total_files_processed = jpeg_files.size();

    for (int i = 0; i < num_threads; i++) {

        std::unique_ptr<ROCJpegDecode> jpg_dec(new ROCJpegDecode(device_id));
        jpeg_decoders.push_back(std::move(jpg_dec));

        jpeg_decoders[i]->GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id, drm_node);
        std::cout << "info: stream " << i <<  " using GPU device " << device_id << ": (drm node: " << drm_node << ") " << device_name << "[" << gcn_arch_name << "] on PCI bus " <<
        std::setfill('0') << std::setw(2) << std::right << std::hex << pci_bus_id << ":" << std::setfill('0') << std::setw(2) <<
        std::right << std::hex << pci_domain_id << "." << pci_device_id << std::dec << std::endl;
    }
    // Create and start threads
    std::vector<std::thread> threads;

    std::cout << "info: decoding started with " << num_threads << " threads, please wait!" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(ThreadFunction, std::ref(jpeg_files), jpeg_decoders[i].get(), std::ref(mutex));
    }
    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Calculate average decoding time
    double total_decoding_time = 0.0;
    total_decoding_time = duration;
    double average_decoding_time = total_decoding_time / total_files_processed;
    std::cout << "info: total decoded images: " << total_files_processed << std::endl;
    std::cout << "Average decoding time (ms): " << average_decoding_time << std::endl;
    std::cout << "Average FPS: " << 1000 / average_decoding_time << std::endl;
    std::cout << "info: decoding completed!" << std::endl;

    return 0;
}