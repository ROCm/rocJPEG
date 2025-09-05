.. meta::
  :description: rocJPEG Installation Prerequisites
  :keywords: install, rocJPEG, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocJPEG prerequisites
********************************************************************

rocJPEG requires ROCm running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

ROCm must be installed before installing rocJPEG. See `Quick start installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>`_ for detailed ROCm installation instructions.

rocJPEG has been tested on the following Linux environments:
  
* Ubuntu 22.04 and 24.04
* RHEL 8 and 9
* SLES 15 SP6 and SP7

See `Supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_ for the complete list of ROCm supported Linux environments.

The following prerequisites are installed by the package installer. If you are building and installing using the source code, use the `rocJPEG-setup.py <https://github.com/ROCm/rocJPEG/blob/develop/rocJPEG-setup.py>`_ setup script available in the rocJPEG GitHub repository to install these prerequisites. 

* AMD Clang++
* CMake version 3.10 or later
* Video Acceleration API (VA-API) - libva-amdgpu-dev is an AMD implementation for VA-API
* AMD VA Drivers
* libstdc++-12-dev for installations on Ubuntu 22.04 

