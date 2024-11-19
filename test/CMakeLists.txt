# ##############################################################################
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
# ##############################################################################
cmake_minimum_required(VERSION 3.5)

project(ROCJPEG-test)

# make test with CTest
enable_testing()
include(CTest)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "${White}${PROJECT_NAME}: Default ROCm installation path${ColourReset}")
elseif(ROCM_PATH)
  message("-- ${White}${PROJECT_NAME} :ROCM_PATH Set -- ${ROCM_PATH}${ColourReset}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "${White}${PROJECT_NAME}: Default ROCm installation path${ColourReset}")
endif()

# find rocJPEG
find_library(ROCJPEG_LIBRARY NAMES ROCJPEG HINTS ${ROCM_PATH}/lib)
find_path(ROCJPEG_INCLUDE_DIR NAMES rocjpeg.h PATHS /opt/rocm/include/rocjpeg ${ROCM_PATH}/include/rocjpeg)

if(ROCJPEG_LIBRARY AND ROCJPEG_INCLUDE_DIR)
    set(ROCJPEG_FOUND TRUE)
    message("-- ${White}Using ROCJPEG -- \n\tLibraries:${ROCJPEG_LIBRARY} \n\tIncludes:${ROCJPEG_INCLUDE_DIR}${ColourReset}")
endif()

if(NOT ROCJPEG_FOUND)
    message("-- ${Yellow}${PROJECT_NAME} requires rocJPEG. Install rocJPEG before running CTests")
endif()

add_test(
  NAME
    jpeg-decode-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/
)

add_test(
  NAME
    jpeg-decode-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt yuv_planar
)

add_test(
  NAME
    jpeg-decode-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt y
)

add_test(
  NAME
    jpeg-decode-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb
)

add_test(
  NAME
    jpeg-decode-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb_planar
)

add_test(
  NAME
    jpeg-decode-threads-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeMultiThreads"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeMultiThreads"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodemultithreads"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -t 2
)

add_test(
  NAME
    jpeg-decode-batch-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/
)

add_test(
  NAME
    jpeg-decode-crop-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt yuv_planar -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt y -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb_planar -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-batch-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -crop 960,540,2880,1620
)