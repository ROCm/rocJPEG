.. meta::
  :description: Install rocJPEG
  :keywords: install, rocJPEG, AMD, ROCm

********************************************************************
Installation
********************************************************************

rocJPEG is a high performance JPEG decode SDK for AMD GPUs. Using the rocJPEG API,
you can access the JPEG decoding features available on your GPU.

Tested configurations
========================================

* Linux

  * Ubuntu: 20.04/22.04
  * RHEL: 8/9
  * SLES: 15-SP5

* ROCm

  * rocm-core: 6.1.0.60100-65
  * amdgpu-core: 1:6.1.60100-1744891

* rocJPEG Setup Script: V1.0

Supported JPEG chroma subsampling
========================================

* YUV 4:4:4
* YUV 4:4:0
* YUV 4:2:2
* YUV 4:2:0
* YUV 4:0:0

Prerequisites
========================================

* Linux distribution

  * Ubuntu: 20.04/22.04
  * RHEL: 8/9
  * SLES: 15-SP5

* `ROCm-supported hardware <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_
  (``gfx908`` or higher is required)

* Install ROCm 6.3.0 or later with
  `amdgpu-install <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_

  * Run: ``--usecase=rocm``
  * To install rocJPEG with minimum requirements, follow the :doc:`quick-start instructions <./quick-start>`

* Video Acceleration API - Version `1.5.0+` - `Libva` is an implementation for VA-API

  .. code:: shell

   sudo apt install libva-dev

* AMD VA Drivers

  .. code:: shell

   sudo apt install mesa-amdgpu-va-drivers

* CMake 3.5 or later

  .. code:: shell

   sudo apt install cmake

* pkg-config

  .. code:: shell

    sudo apt install pkg-config

* If using Ubuntu 22.04, you must install ``libstdc++-12-dev``

  .. code:: shell

    sudo apt install libstdc++-12-dev

.. note::

  All package installs are shown with the ``apt`` package manager. Use the appropriate package
  manager for your operating system.

Prerequisites setup script
----------------------------------------------------------------------------------------------------------

For your convenience, we provide the setup script,
`rocJPEG-setup.py <https://github.com/ROCm/rocJPEG/blob/develop/rocJPEG-setup.py>`_,
which installs all required dependencies. Run this script only once.

.. code:: shell

  python rocJPEG-setup.py  --rocm_path [ ROCm Installation Path - optional (default:/opt/rocm)]

Installation instructions
========================================

To install rocJPEG, you can use :ref:`package-install` or
:ref:`source-install`.

.. _package-install:

Package install
------------------------------------------------------------------------------------------------------------

To install rocJPEG runtime, development, and test packages, run the line of code for your operating
system.

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install rocjpeg rocjpeg-dev rocjpeg-test

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install rocjpeg rocjpeg-devel rocjpeg-test

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install rocjpeg rocjpeg-devel rocjpeg-test

.. note::

  Package install auto installs all dependencies.

* Runtime package: ``rocjpeg`` only provides the rocjpeg library ``librocdecode.so``
* Development package: ``rocjpeg-dev``or ``rocjpeg-devel`` provides the library, header files, and samples
* Test package: ``rocjpeg-test`` provides CTest to verify installation

.. _source-install:

Source install
------------------------------------------------------------------------------------------------------------

To build rocJPEG from source, run:

.. code:: shell

  git clone https://github.com/ROCm/rocJPEG.git
  cd rocJPEG
  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install

Run tests:

.. code:: shell

  make test

To run tests with verbose option, use ``make test ARGS="-VV"``.

Make package:

.. code:: shell

  sudo make package

Verify installation
========================================

The installer copies:

* Libraries into ``/opt/rocm/lib``
* Header files into ``/opt/rocm/include/rocjpeg``
* Samples folder into ``/opt/rocm/share/rocjpeg``
* Documents folder into ``/opt/rocm/share/doc/rocjpeg``

To verify your installation using a sample application, run:

.. code:: shell

  mkdir rocjpeg-sample && cd rocjpeg-sample
  cmake /opt/rocm/share/rocjpeg/samples/videoDecode/
  make -j8
  ./jpegdecode -i /opt/rocm/share/rocjpeg/images/

To verify your installation using the ``rocjpeg-test`` package, run:

.. code:: shell

  mkdir rocjpeg-test && cd rocjpeg-test
  cmake /opt/rocm/share/rocjpeg/test/
  ctest -VV

This test package installs the CTest module.

Samples
========================================

You can access samples to decode your JPEG images in our
`GitHub repository <https://github.com/ROCm/rocJPEG/tree/develop/samples>`_. Refer to the
individual folders to build and run the samples.

Docker
========================================

You can find rocJPEG Docker containers in our
`GitHub repository <https://github.com/ROCm/rocJPEG/tree/develop/docker>`_.

Documentation
========================================

Run the following code to build our documentation locally.

.. code:: shell

  cd docs
  pip3 install -r sphinx/requirements.txt
  python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

For more information on documentation builds, refer to the
:doc:`Building documentation <rocm:contribute/building>` page.

Hardware capabilities
===================================================

The following table shows the capabilities of the VCN and JPEG cores for each supported GPU
architecture.

.. csv-table::
  :header: "GPU Architecture", "VCN Generation", "Number of VCNs", "Number of JPEG cores per VCN", "Max width, Max height"

  "gfx908 - MI1xx", "VCN 2.5.0", "2", "1", "4096, 4096"
  "gfx90a - MI2xx", "VCN 2.6.0", "2", "1", "4096, 4096"
  "gfx940, gfx942 - MI300A", "VCN 3.0", "3", "8", "4096, 4096"
  "gfx941, gfx942 - MI300X", "VCN 3.0", "4", "8", "4096, 4096"
  "gfx1030, gfx1031, gfx1032 - Navi2x", "VCN 3.x", "2", "1", "4096, 4096"
  "gfx1100, gfx1102 - Navi3x", "VCN 4.0", "2", "1", "4096, 4096"
  "gfx1101 - Navi3x", "VCN 4.0", "1", "1", "4096, 4096"
