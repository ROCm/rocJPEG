# rocJPEG
rocJPEG is a high-performance jpeg decode SDK for decoding images using a hardware-accelerated JPEG decoder on AMD’s GPUs.

## Prerequisites:

* One of the supported GPUs by ROCm: [AMD Radeon&trade; Graphics](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html)
* Linux distribution
  + **Ubuntu** - `20.04` / `22.04`
* Install [ROCm5.5 or later](https://docs.amd.com)
  + **Note** - both rocm and graphics use-cases must be installed (i.e., sudo amdgpu-install --usecase=graphics,rocm --no-32).
* CMake 3.0 or later
* libva-dev 2.7 or later

* **Note** [vcnDECODE-setup.py](vcnDECODE-setup.py) script can be used for installing all the dependencies

## Build instructions:
Please follow the instructions below to build and install the rocJPEG library.
```
 cd rocJPEG
 mkdir build; cd build
 cmake ..
 make -j8
 sudo make install
```

## Samples:
The tool provides a few samples to decode jpeg images [here](samples/). Please refer to the individual folders to build and run the samples.

## Docker:
Docker files to build rocDecode containers are available [here](docker/)
