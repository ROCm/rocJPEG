# ##############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc.
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
cmake_minimum_required(VERSION 3.10)

add_test(
  NAME
  jpeg-decode-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/
)

add_test(
  NAME
  jpeg-decode-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -fmt yuv_planar
)

add_test(
  NAME
  jpeg-decode-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -fmt y
)

add_test(
  NAME
  jpeg-decode-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -fmt rgb
)

add_test(
  NAME
  jpeg-decode-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -fmt rgb_planar
)

add_test(
  NAME
  jpeg-decode-perf-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecodePerf"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodePerf"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodeperf"
            -i ${CMAKE_SOURCE_DIR}/data/images/
)

add_test(
  NAME
  jpeg-decode-batch-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${CMAKE_SOURCE_DIR}/data/images/
)

add_test(
  NAME
  jpeg-decode-crop-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -fmt yuv_planar -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -fmt y -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -fmt rgb -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -fmt rgb_planar -crop 960,540,2880,1620
)

add_test(
  NAME
  jpeg-decode-crop-batch-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${CMAKE_SOURCE_DIR}/data/images/ -crop 960,540,2880,1620
)