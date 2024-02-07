# JPEG decode sample

The jpeg decode sample illustrates decoding a JPEG images using rocJPEG library to get the individual decoded images in YUV format. This sample can be configured with a device ID and optionally able to dump the output to a file.

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
./jpegdecode -i <input JPEG file [required]> 
              -o <output path to save decoded image [optional]> 
              -d <GPU device ID - 0:device 0 / 1:device 1/ ... [optional - default:0]>
```