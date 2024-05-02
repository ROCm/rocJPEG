# rocJPEG changelog

Documentation for rocJPEG is available at
[https://rocm.docs.amd.com/projects/rocJPEG/en/latest/](TBD)

## rocJPEG 0.2.0 (Unreleased)

## Additions

* Changelog.md

## Optimizations

* Setup Script - Error Check install & cleanup

### Changes

* Dependencies - Updates to core dependencies
* LibVA Headers - Use public headers
* mesa-amdgpu-va-drivers - RPM Package available on RPM from ROCm 6.2

### Fixes

* Package deps
* RHEL/SLES - Additional required packages `mesa-amdgpu-dri-drivers libdrm-amdgpu`

### Tested configurations

* Linux
  * Ubuntu - `20.04` / `22.04`
  * RHEL - `8` / `9`
* ROCm:
  * rocm-core - `6.1.0.60100-64`
  * amdgpu-core - `1:6.1.60100-1741643`
* libva-dev - `2.7.0-2` / `2.14.0-1`
* mesa-amdgpu-va-drivers - `1:24.1.0`
* mesa-amdgpu-dri-drivers - `24.1.0.60200`
* rocJPEG Setup Script - `V1.2.0`
