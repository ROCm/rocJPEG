# JPEG decode sample

The jpeg decode sample illustrates decoding a JPEG images using rocJPEG library to get the individual decoded images in one of the supported output format (i.e., unchanged, yuv, y, rgbi). This sample can be configured with a device ID and optionally able to dump the output to a file.

## Prerequisites:

* Install [rocJPEG](../../README.md#build-and-install-instructions)

## Build

```shell
mkdir jpeg_decode_sample && cd jpeg_decode_sample
cmake ../
make -j
```

## Run

```shell
    ./jpegdecode -i <Path to single image or directory of images - [required]>
                 -be <Select rocJPEG backend (0 for ROCJPEG_BACKEND_HARDWARE, using VCN hardware-accelarated JPEG decoder, 1 ROCJPEG_BACKEND_HYBRID, using CPU and GPU HIP kernles for JPEG decoding) [optional]>
                 -fmt <Select rocJPEG output format for decoding, one of the [native, yuv, y, rgbi] [optional - default: native]>
                 -o <Output file path or directory - Write decoded images based on the selected outfut format to this file or directory [optional]>
                 -d <GPU device id (0 for the first GPU device, 1 for the second GPU device, etc.) [optional - default: 0]>
```