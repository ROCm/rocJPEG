################################################################################
# Copyright (c) 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

cmake_minimum_required (VERSION 3.0)
project(jpegdecodemultithreads)
set(CMAKE_CXX_STANDARD 17)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
set(CMAKE_CXX_COMPILER ${ROCM_PATH}/llvm/bin/clang++)

find_package(HIP QUIET)

# find rocJPEG
find_library(ROCJPEG_LIBRARY NAMES rocjpeg HINTS {ROCM_PATH}/lib)
find_path(ROCJPEG_INCLUDE_DIR NAMES rocjpeg.h PATHS /opt/rocm/include/rocjpeg {ROCM_PATH}/include/rocjpeg)

if(ROCJPEG_LIBRARY AND ROCJPEG_INCLUDE_DIR)
    set(ROCJPEG_FOUND TRUE)
    message("-- ${White}Using rocJPEG -- \n\tLibraries:${ROCJPEG_LIBRARY} \n\tIncludes:${ROCJPEG_INCLUDE_DIR}${ColourReset}")
endif()

# threads
find_package(Threads REQUIRED)

if(HIP_FOUND AND ROCJPEG_FOUND AND Threads_FOUND)
    # HIP
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
    #threads
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)
    #filesystem: c++ compilers less than equal to 8.5 need explicit link with stdc++fs
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL "8.5")
      set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} stdc++fs)
    endif()
    # rocJPEG
    include_directories (${ROCJPEG_INCLUDE_DIR})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${ROCJPEG_LIBRARY})
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR} jpegdecodemultithreads.cpp)
    add_executable(${PROJECT_NAME} ${SOURCES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
else()
    message("-- ERROR!: ${PROJECT_NAME} excluded! please install all the dependencies and try again!")
    if (NOT HIP_FOUND)
        message(FATAL_ERROR "-- ERROR!: HIP Not Found! - please install ROCm and HIP!")
    endif()
    if (NOT ROCJPEG_FOUND)
        message(FATAL_ERROR "-- ERROR!: rocJPEG Not Found! - please install rocJPEG!")
    endif()
    if (NOT Threads_FOUND)
        message(FATAL_ERROR "-- ERROR!: Threads Not Found! - please insatll Threads!")
    endif()
endif()