# rocJPEG changelog

Documentation for rocJPEG is available at
[https://rocm.docs.amd.com/projects/rocJPEG/en/latest/](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/)

## (Unreleased) rocJPEG 0.8.0

### Changed

* AMD Clang++ is now the default CXX compiler.
* `rocJPEG-setup.py` setup script updates to common package install: Setup no longer installs public compiler package.
* The jpegDecodeMultiThreads sample has been renamed to jpegDecodePerf, and batch decoding has been added to this sample instead of single image decoding for improved performance.

### Removed

* 

### Resolved issues

* 

### Tested configurations

* Linux
  * Ubuntu - `22.04` / `24.04`
  * RHEL - `8` / `9`
  * SLES - `15 SP5`
* ROCm: `6.3.0`
* libva-amdgpu-dev - `2.16.0`
* mesa-amdgpu-va-drivers - `1:24.3.0`
* rocJPEG Setup Script - `V2.3.0`

## rocJPEG 0.6.0 for ROCm 6.3.0

### Changes

* Supported initial enablement of the rocJPEG library
* Supported JPEG chroma subsampling:
  * YUV 4:4:4
  * YUV 4:4:0
  * YUV 4:2:2
  * YUV 4:2:0
  * YUV 4:0:0
* Supported various output image format:
  * Native (i.e., native unchanged output from VCN Hardware, it can be either packed YUV or planar YUV)​
  * YUV planar​
  * Y only​
  * RGB​
  * RGB planar
* Supported single-image and batch-image decoding​
* Supported Different partition modes on MI300 series​
* Supported region-of-interest (ROI) decoding​
* Supported color space conversion from YUV to RGB