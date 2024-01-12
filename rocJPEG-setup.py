# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

__license__ = "MIT"
__version__ = "1.0"
__status__ = "Shipping"

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
if "centos" in platfromInfo or "redhat" in platfromInfo:
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    if "centos-7" in platfromInfo or "redhat-7" in platfromInfo:
        linuxCMake = 'cmake3'
        os.system(linuxSystemInstall+' install cmake3')
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
    print("\nrocJPEG Setup Supported on: Ubuntu 20/22; CentOS 7/8; RedHat 7/8; & SLES 15-SP2\n")
    exit()

# rocJPEG Setup
print("\nrocJPEG Setup on: "+platfromInfo+"\n")

if userName == 'root':
    os.system(linuxSystemInstall+' update')
    os.system(linuxSystemInstall+' install sudo')


# Clean Install
print("\nrocJPEG Dependencies Installation with rocJPEG-setup.py V-"+__version__+"\n")

# install pre-reqs
os.system('sudo -v')
os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' ' +
    linuxSystemInstall_check+' install gcc cmake git wget unzip pkg-config inxi vainfo')

if "Ubuntu" in platfromInfo:
    os.system('sudo -v')
    os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
        ' install autoconf automake build-essential g++-12 git-core libass-dev libfreetype6-dev')
    os.system('sudo -v')
    os.system('sudo '+linuxFlag+' '+linuxSystemInstall+' '+linuxSystemInstall_check +
        ' install libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev')
    os.system('sudo -v')

print("\nrocJPEG Dependencies Installed with rocJPEG-setup.py V-"+__version__+"\n")
