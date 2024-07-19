.. meta::
  :description: Install rocJPEG
  :keywords: install, rocJPEG, AMD, ROCm

********************************************************************
rocJPEG quick-start installation
********************************************************************

To install the rocJPEG runtime with minimum requirements, follow these steps:

1. Install core ROCm components (ROCm 6.3.0 or later) using the
:doc:`native package manager <rocm-install-on-linux:how-to/native-install/index>`
installation instructions.

* Register repositories
* Register kernel-mode driver
* Register ROCm packages
* Install kernel driver (``amdgpu-dkms``)--only required on bare metal install. Docker runtime uses the
  base ``dkms`` package irrespective of the version installed.

2. Install rocJPEG runtime package. rocJPEG only provides the ``librocjpeg.so`` library (the
runtime package only installs the required core dependencies).

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install rocjpeg

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install rocjpeg

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install rocjpeg
