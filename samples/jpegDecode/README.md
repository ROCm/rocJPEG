# Jpeg Decode Sample
This sample illustrates the JPEG decoding on AMD hardware using VAAPI. This sample supports YUV420, YUV444, and YUV400.

## Build and run the sample:
```
mkdir build
cd build
cmake ..
make -j
./jpegdecode -i <input video file - required> -o <optional (default: 0); 1 to save the decoded YUV frame into a file> -d <GPU device ID, 0 for the first device, 1 for the second device, etc> -c <optional; 1 to convert the decoded YUV to RGB>
```