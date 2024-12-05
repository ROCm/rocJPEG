[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="70%" src="docs/data/AMD_rocJPEG_Logo.png" /></p>

rocJPEG is a high performance JPEG decode SDK for AMD GPUs. Using the rocJPEG API, you can access the JPEG decoding features available on your GPU.

>[!Note]
>The published documentation that includes installation and build instructions is available at [rocJPEG](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html)

# Installation folders

* Libraries are located in: `/opt/rocm/lib`
* Header files are located in:  `/opt/rocm/include/rocjpeg`
* Samples are located in:  `/opt/rocm/share/rocjpeg`
* Documentation is located in: `/opt/rocm/share/doc/rocjpeg`

# Samples

Samples that decode JPEG images are available [under `samples` in this repository](samples/). Refer to the individual folders to build and run the samples.

## Verifying your installation using samples and tests

To verify your installation using a sample application, run:

```shell
mkdir rocjpeg-sample && cd rocjpeg-sample
cmake /opt/rocm/share/rocjpeg/samples/jpegDecode/
make -j8
./jpegdecode -i /opt/rocm/share/rocjpeg/images/mug_420.jpg
```

To verify your installation using the `rocjpeg-test` package, run:

```shell
mkdir rocjpeg-test && cd rocjpeg-test
cmake /opt/rocm/share/rocjpeg/test/
ctest -VV
```

# Docker

You can find rocJPEG Docker containers [under `develop/docker` in this repository](https://github.com/ROCm/rocJPEG/tree/develop/docker).
