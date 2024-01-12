# Jpeg Decode Performance Sample
This sample illustrates the JPEG decoding on AMD hardware using VAAPI. This sample supports YUV420, YUV444, and YUV400. This sample uses multiple threads to decode a pool of jpeg images parallely.

## Build and run the sample:
```
mkdir build
cd build
cmake ..
make -j
./jpegdecodeperf -i <input video file - required> -t <optional (default: 4); number of threads to use for parallel decoding> -d <GPU device ID, 0 for the first device, 1 for the second device, etc>
```