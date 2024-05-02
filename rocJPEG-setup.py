# Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import sys
import argparse
import platform
if sys.version_info[0] < 3:
    import commands
else:
    import subprocess

__copyright__ = "Copyright (c) 2024, AMD rocJPEG"
__version__ = "1.2.0"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"

# error check calls
def ERROR_CHECK(call):
    status = call
    if(status != 0):
        print('ERROR_CHECK failed with status:'+str(status))
        exit(status)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rocm_path', 	type=str, default='/opt/rocm',
                    help='ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required')

args = parser.parse_args()

ROCM_PATH = args.rocm_path

if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
print("\nROCm PATH set to -- "+ROCM_PATH+"\n")

# check ROCm installation
if os.path.exists(ROCM_PATH):
    print("\nROCm Installation Found -- "+ROCM_PATH+"\n")
    os.system('echo ROCm Info -- && '+ROCM_PATH+'/bin/rocminfo')
else:
    print(
        "WARNING: If ROCm installed, set ROCm Path with \"--rocm_path\" option for full installation [Default:/opt/rocm]\n")
    print("ERROR: rocJPEG Setup requires ROCm install\n")
    exit(-1)

# get platfrom info
platfromInfo = platform.platform()

# sudo requirement check
sudoLocation = ''
userName = ''
if sys.version_info[0] < 3:
    status, sudoLocation = commands.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = commands.getstatusoutput("whoami")
else:
    status, sudoLocation = subprocess.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = subprocess.getstatusoutput("whoami")

# setup for Linux
linuxSystemInstall = ''
linuxCMake = 'cmake'
linuxSystemInstall_check = ''
linuxFlag = ''
sudoValidateOption= '-v'
if "centos" in platfromInfo or "redhat" in platfromInfo or os.path.exists('/usr/bin/yum'):
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    if "centos-7" in platfromInfo or "redhat-7" in platfromInfo:
        print("\nrocJPEG Setup on "+platfromInfo+" is unsupported\n")
        exit(-1)
    if not "centos" in platfromInfo or not "redhat" in platfromInfo:
        platfromInfo = platfromInfo+'-redhat'
elif "Ubuntu" in platfromInfo or os.path.exists('/usr/bin/apt-get'):
    linuxSystemInstall = 'apt-get -y'
    linuxSystemInstall_check = '--allow-unauthenticated'
    linuxFlag = '-S'
    if not "Ubuntu" in platfromInfo:
        platfromInfo = platfromInfo+'-Ubuntu'
elif os.path.exists('/usr/bin/zypper'):
    linuxSystemInstall = 'zypper -n'
    linuxSystemInstall_check = '--no-gpg-checks'
    platfromInfo = platfromInfo+'-SLES'
else:
    print("\nrocJPEG Setup on "+platfromInfo+" is unsupported\n")
    print("\nrocJPEG Setup Supported on: Ubuntu 20/22, RedHat 8/9, & SLES 15 SP4\n")
    exit(-1)

# rocJPEG Setup
print("\nrocJPEG Setup on: "+platfromInfo+"\n")
print("\nrocJPEG Dependencies Installation with rocJPEG-setup.py V-"+__version__+"\n")

if userName == 'root':
    ERROR_CHECK(os.system(linuxSystemInstall+' update'))
    ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

# source install - common package dependencies
commonPackages = [
    'gcc',
    'cmake',
    'git',
    'wget',
    'unzip',
    'pkg-config',
    'inxi',
    'rocm-hip-runtime'
]

# Debian packages
coreDebianPackages = [
    'rocm-hip-runtime-dev',
    'libva2',
    'libva-dev',
    'libdrm-amdgpu1',
    'mesa-amdgpu-va-drivers',
    'vainfo'
]
coreDebianU22Packages = [
    'libstdc++-12-dev'
]

# RPM Packages
libvaNameRPM = "libva"
if os.path.exists('/usr/bin/zypper'):
        libvaNameRPM = "libva2"
coreRPMPackages = [
    'rocm-hip-runtime-devel',
    str(libvaNameRPM),
    'libva',
    'libva-devel',
    'libdrm-amdgpu',
    'mesa-amdgpu-va-drivers',
    'mesa-amdgpu-dri-drivers',
    'libva-utils'
]

# common packages
ERROR_CHECK(os.system('sudo '+sudoValidateOption))
for i in range(len(commonPackages)):
    ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
            ' '+linuxSystemInstall_check+' install '+ commonPackages[i]))

# rocJPEG Core - LibVA Requirements
ERROR_CHECK(os.system('sudo '+sudoValidateOption))
if "Ubuntu" in platfromInfo:
    for i in range(len(coreDebianPackages)):
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                ' '+linuxSystemInstall_check+' install '+ coreDebianPackages[i]))
    if "22.04" in platform.version():
        for i in range(len(coreDebianU22Packages)):
            ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                ' '+linuxSystemInstall_check+' install '+ coreDebianU22Packages[i]))
else:
    for i in range(len(coreRPMPackages)):
        ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +
                ' '+linuxSystemInstall_check+' install '+ coreRPMPackages[i]))

print("\nrocJPEG Dependencies Installed with rocJPEG-setup.py V-"+__version__+"\n")
