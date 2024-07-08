.. meta::
  :description: Using rocJPEG
  :keywords: parse JPEG, parse, decode, JPEG decoder, JPEG decoding, rocJPEG, AMD, ROCm

********************************************************************
Using rocJPEG
********************************************************************

To learn how to use the rocJPEG SDK library, follow these instructions:

1. API overview
====================================================

All rocJPEG APIs are exposed in the header file ``rocjpeg.h``. You can find
this file in the `api` folder in the rocJPEG repository.

2. Create a decoder
====================================================

The ``rocJpegCreate()`` function creates a JPEG decoder object and returns a handle upon successful creation.

Below is the signature of ``rocJpegCreate()`` function:

.. code:: cpp

    RocJpegStatus rocJpegCreate(
      RocJpegBackend backend,
      int device_id,
      RocJpegHandle *handle);

The API takes in the following arguments:

* A ``RocJpegBackend`` type, which specifies the backend to use for creating a decoder handle.
  Currently, the rocJPEG library only supports ``ROCJPEG_BACKEND_HARDWARE``, which creates a decoder
  for baseline JPEG bitstream using VCN hardware-accelerated JPEG decoder in AMD GPUs.
* The GPU device ID for which a decoder should be created. The GPU device ID is a zero-based index, where 0 is for the first GPU on a system.
* A decoder handle, which is returned by ``rocJpegCreate()`` and must be retained for the entire decode session,
  as it is passed along with the other decoding APIs.

3. Create a JPEG stream parser
====================================================

The ``rocJpegStreamCreate`` function creates a JPEG stream parser object and returns a handle upon successful creation. This handle is used to parse and retrieve the information from the JPEG stream.

Below is the signature of the ``rocJpegStreamCreate`` function:

.. code:: cpp

    RocJpegStatus rocJpegStreamCreate(RocJpegStreamHandle *jpeg_stream_handle);

The API takes in the following arguments:

* A JPEG stream handle, which is returned by ``rocJpegStreamCreate``. This handle can be used by the user to parse the JPEG stream and retrieve the information from the stream.

4. Parse a JPEG stream and store its information
====================================================

The ``rocJpegStreamParse`` function parses a jpeg stream and stores the information from the stream to be used in subsequent API calls for retrieving the image information and decoding it.

Below is the signature of the ``rocJpegStreamParse`` function:

.. code:: cpp

    RocJpegStatus rocJpegStreamParse(const unsigned char *data, size_t length, RocJpegStreamHandle jpeg_stream_handle);

The API takes in the following arguments:

* A pointer to the JPEG data buffer.
* The length of the JPEG data buffer.
* The JPEG stream handle, which is returned by ``rocJpegStreamCreate``. This handle is used to parse the JPEG stream and retrieve the information from the stream.

5. Retrieve the image info
====================================================
``rocJpegGetImageInfo()`` retrieves the image info, including number of components, width and height of each component, and chroma subsampling.
For each image to be decoded, pass the JPEG data pointer and data length to the ``rocJpegGetImageInfo()`` function. This function is thread safe.

Below is the signature of ``rocJpegGetImageInfo()`` function:

.. code:: cpp

    RocJpegStatus rocJpegGetImageInfo(
      RocJpegHandle handle,
      RocJpegStreamHandle jpeg_stream_handle,
      uint8_t *num_components,
      RocJpegChromaSubsampling *subsampling,
      uint32_t *widths,
      uint32_t *heights);

One of the outputs of the ``rocJpegGetImageInfo()`` function is ``RocJpegChromaSubsampling``. This parameter is an enum type, and its enumerator
list is composed of the chroma subsampling property retrieved from the JPEG image. See the ``RocJpegChromaSubsampling`` enum below.

.. code:: cpp

    typedef enum {
      ROCJPEG_CSS_444 = 0,
      ROCJPEG_CSS_440 = 1,
      ROCJPEG_CSS_422 = 2,
      ROCJPEG_CSS_420 = 3,
      ROCJPEG_CSS_411 = 4,
      ROCJPEG_CSS_400 = 5,
      ROCJPEG_CSS_UNKNOWN = -1
    } RocJpegChromaSubsampling;

.. note::

  The VCN hardware-accelerated JPEG decoder in AMD GPUs only supports decoding JPEG images with ``ROCJPEG_CSS_444``, ``ROCJPEG_CSS_440``, ``ROCJPEG_CSS_422``,
  ``ROCJPEG_CSS_420``, and ``ROCJPEG_CSS_400`` chroma subsampling.

6. Decode a JPEG stream
====================================================
``rocJpegDecode()`` decodes single image based on the backend used to create the rocJpeg handle in rocJpegCreate API. For each image to be decoded,
pass the JPEG data pointer and data length to the ``rocJpegDecode()`` function. This function is thread safe.

See the signature of this function below:

.. code:: cpp

    RocJpegStatus rocJpegDecode(
      RocJpegHandle handle,
      RocJpegStreamHandle jpeg_stream_handle,
      const RocJpegDecodeParams *decode_params,
      RocJpegImage *destination);

In the above ``rocJpegDecode()`` function, you can use the parameters ``RocJpegDecodeParams`` and ``RocJpegImage`` to set
the output behavior of the ``rocJpegDecode()`` function. The ``RocJpegImage`` structure is JPEG image descriptor used to
return the decoded output image. User must allocate device memories for each channel for this structure and pass it to the
``rocJpegDecode()`` API. This API then copies the decoded image to this struct based on the requested output format ``RocJpegOutputFormat``
defined in the ``RocJpegDecodeParams``.
Below is the ``RocJpegImage`` structure.

.. code:: cpp

    typedef struct {
      uint8_t* channel[ROCJPEG_MAX_COMPONENT];
      uint32_t pitch[ROCJPEG_MAX_COMPONENT];
    } RocJpegImage;

You can set the ``RocJpegOutputFormat`` parameter of the ``RocJpegDecodeParams`` to one of the ``output_format`` settings below:

.. csv-table::
  :header: "output_format", "Meaning"

  "ROCJPEG_OUTPUT_NATIVE", "Return native unchanged decoded YUV image from the VCN JPEG deocder."
  "ROCJPEG_OUTPUT_YUV_PLANAR", "Return in the YUV planar format."
  "ROCJPEG_OUTPUT_Y", "Return the Y component only."
  "ROCJPEG_OUTPUT_RGB", "Convert to interleaved RGB."
  "ROCJPEG_OUTPUT_RGB_PLANAR", "Convert to planar RGB."

For example, if ``output_format`` is set to ``ROCJPEG_OUTPUT_NATIVE``, then based on the chroma subsampling of the input image, the
``rocJpegDecode()`` function does one of the following:

* For ``ROCJPEG_CSS_444`` and ``ROCJPEG_CSS_440`` write Y, U, and V to first, second, and third channels of ``RocJpegImage``.
* For ``ROCJPEG_CSS_422`` write YUYV (packed) to first channel of ``RocJpegImage``.
* For ``ROCJPEG_CSS_420`` write Y to first channel and UV (interleaved) to second channel of ``RocJpegImage``.
* For ``ROCJPEG_CSS_400`` write Y to first channel of ``RocJpegImage``.

if ``output_format`` is set to ``ROCJPEG_OUTPUT_Y`` or ``ROCJPEG_OUTPUT_RGB`` then ``rocJpegDecode()`` copies the output to first channel of ``RocJpegImage``.
Alternately, in the case of ``ROCJPEG_OUTPUT_YUV_PLANAR`` or ``ROCJPEG_OUTPUT_RGB_PLANAR``, the data is written to the corresponding channels of the ``RocJpegImage`` destination structure.
The destination buffers should be large enough to be able to store output of specified format. These buffers should be
pre-allocated by the user in the device memories. For each color plane (channel), sizes could be retrieved for image using
``rocJpegGetImageInfo()`` API and minimum required memory buffer for each plane is plane_height * plane_pitch where
plane_pitch >= plane_width for planar output formats and plane_pitch >= plane_width * num_components for interleaved output format.

As mentioned above, you can use the retrieved parameters, ``num_components``, ``subsampling``, ``widths``, and ``heights`` from the ``rocJpegGetImageInfo()`` API to calculate
the required size for the output buffers for a single decode JPEG. To optimally set the destination parameter for the ``rocJpegDecode()`` function, use the following guidelines:

.. csv-table::
  :header: "output_format", "chroma subsampling", "destination.pitch[c] should be atleast:", "destination.channel[c] should be atleast:"

  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_444", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[c] = destination.pitch[c] * heights[0] for c = 0, 1, 2"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_440", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[0] = destination.pitch[0] * heights[0], destination.channel[c] = destination.pitch[c] * heights[0] / 2 for c = 1, 2"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_422", "destination.pitch[0] = widths[0] * 2", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_420", "destination.pitch[1] = destination.pitch[0] = widths[0]", "destination.channel[0] = destination.pitch[0] * heights[0], destination.channel[1] = destination.pitch[1] * (heights[0] >> 1)"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_400", "destination.pitch[0] = widths[0]", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_YUV_PLANAR", "ROCJPEG_CSS_444, ROCJPEG_CSS_440, ROCJPEG_CSS_422, ROCJPEG_CSS_420", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[c] = destination.pitch[c] * heights[c] for c = 0, 1, 2"
  "ROCJPEG_OUTPUT_YUV_PLANAR", "ROCJPEG_CSS_400", "destination.pitch[0] = widths[0]", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_Y", "Any of the supported chroma subsampling", "destination.pitch[0] = widths[0]", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_RGB", "Any of the supported chroma subsampling", "destination.pitch[0] = widths[0] * 3", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_RGB_PLANAR", "Any of the supported chroma subsampling", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[c] = destination.pitch[c] * heights[c] for c = 0, 1, 2"

7. Decode a batch of JPEG streams
====================================================
The ``rocJpegDecodeBatched()`` function decodes a batch of JPEG images using the rocJPEG library.

Below is the signature of the ``rocJpegDecodeBatched()`` function:

.. code:: cpp

  RocJpegStatus rocJpegDecodeBatched(
    RocJpegHandle handle,
    RocJpegStreamHandle *jpeg_stream_handles,
    int batch_size,
    const RocJpegDecodeParams *decode_params,
    RocJpegImage *destinations);

The ``rocJpegDecodeBatched()`` function takes the following arguments:

* ``handle``: The rocJPEG handle.
* ``jpeg_stream_handles``: An array of rocJPEG stream handles, each representing a JPEG image.
* ``batch_size``: The number of images in the batch.
* ``decode_params``: The decode parameters for the JPEG images.
* ``destinations``: An array of rocJPEG images to store the decoded images.

To use the ``rocJpegDecodeBatched()`` function, you need to provide the appropriate rocJPEG handles, stream handles, decode parameters, and destination images. The function will decode the batch of JPEG images and store the decoded images in the ``destinations`` array.
Remember to allocate device memories for each channel of the destination images and pass them to the ``rocJpegDecodeBatched()`` API. The API will then copy the decoded images to the destination images based on the requested output format specified in the ``RocJpegDecodeParams``.

8. Destroy the decoder
====================================================

You must call the ``rocJpegDestroy()`` to destroy the session and free up resources.

9. Destroy the JPEG stream handle
====================================================

You must call the ``rocJpegStreamDestroy()`` to release the stream parser object and resources.

10. Get Error name
====================================================

You can call ``rocJpegGetErrorName`` to retrieve the name of the specified error code in text form returned from rocJPEG APIs.

11. Sample code snippet for decoding a JPEG stream using the rocJPEG APIs
====================================================

The code snippet provided demonstrates how to decode a JPEG stream using the rocJPEG library.
First, the code reads the JPEG image file and stores the data in a vector. Then, it initializes the rocJPEG handle using the ``rocJpegCreate()`` function. If the handle creation is successful, it proceeds to create a JPEG stream using the ``rocJpegStreamCreate()`` function.
Next, the code parses the JPEG stream by calling the ``rocJpegStreamParse()`` function with the JPEG data and its size. If the parsing is successful, it retrieves the image information using the ``rocJpegGetImageInfo()`` function.
After obtaining the image information, the code allocates HIP device memory for the decoded image using the ``RocJpegImage`` structure. It sets the channel and pitch values based on the image width and height.
Finally, the code decodes the JPEG stream by calling the ``rocJpegDecode()`` function with the rocJPEG handle, stream handle, and the decoded image structure. If the decoding is successful, the decoded image can be further processed or displayed.

.. code:: cpp

  #include <iostream>
  #include <fstream>
  #include <vector>
  #include "rocjpeg.h"

  int main() {
    // Read the JPEG image file
    std::ifstream input("mug_420.jpg", std::ios::in | std::ios::binary | std::ios::ate);

    // Get the JPEG image file size
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);

    std::vector<char> file_data;
    // resize if buffer is too small
    if (file_data.size() < file_size) {
      file_data.resize(file_size);
    }
    // Read the JPEG stream
    if (!input.read(file_data.data(), file_size)) {
      std::cerr << "ERROR: cannot read from file: " << std::endl;
      return EXIT_FAILURE;
    }

    // Initialize rocJPEG
    RocJpegHandle handle;
    RocJpegStatus status = rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, 0, &handle);
    if (status != ROCJPEG_STATUS_SUCCESS) {
      std::cerr << "Failed to create rocJPEG handle with error code: " << rocJpegGetErrorName(status) << std::endl;
      return EXIT_FAILURE;
    }

    // Create a JPEG stream
    RocJpegStreamHandle rocjpeg_stream_handle;
    status = rocJpegStreamCreate(&rocjpeg_stream_handle);
    if (status != ROCJPEG_STATUS_SUCCESS) {
      std::cerr << "Failed to create JPEG stream with error code: " << rocJpegGetErrorName(status) << std::endl;
      rocJpegDestroy(handle);
      return EXIT_FAILURE;
    }

    // Parse the JPEG stream
    status = rocJpegStreamParse(reinterpret_cast<uint8_t*>(file_data.data()), file_size, rocjpeg_stream_handle);
    if (status != ROCJPEG_STATUS_SUCCESS) {
      std::cerr << "Failed to parse JPEG stream with error code: " << rocJpegGetErrorName(status) << std::endl;
      rocJpegStreamDestroy(rocjpeg_stream_handle);
      rocJpegDestroy(handle);
      return EXIT_FAILURE;
    }

    // Get the image info
    uint8_t num_components;
    RocJpegChromaSubsampling subsampling;
    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};

    status = rocJpegGetImageInfo(rocjpeg_handle, rocjpeg_stream_handle, &num_components, &subsampling, widths, heights);
    if (status != ROCJPEG_STATUS_SUCCESS) {
      std::cerr << "Failed to get image info with error code: " << rocJpegGetErrorName(status) << std::endl;
      rocJpegStreamDestroy(rocjpeg_stream_handle);
      rocJpegDestroy(handle);
      return EXIT_FAILURE;
    }

    // Allocate device memory for the decoded output image
    RocJpegImage output_image = {};
    RocJpegDecodeParams decode_params = {};
    RocJpegDecodeParams.output_format = ROCJPEG_OUTPUT_NATIVE;

    // For this sample assuming the input image has a YUV420 chroma subsampling.
    // For YUV420 subsampling, the native decoded output image would be NV12 (i.e., the rocJPegDecode API copies Y to first channel and UV (interleaved) to second channel of RocJpegImage)
    output_image.pitch[1] = output_image.pitch[0] = widths[0];
    hipError_t hip_status;
    hip_status = hipMalloc(&output_image.channel[0], output_image.pitch[0] * heights[0]);
    if (hip_status != hipSuccess) {
      std::cerr << "Failed to allocate device memory for the first channel" << std::endl;
      rocJpegStreamDestroy(rocjpeg_stream_handle);
      rocJpegDestroy(handle);
      return EXIT_FAILURE;
    }

    hip_status = hipMalloc(&output_image.channel[1], output_image.pitch[1] * (heights[0] >> 1));
    if (hip_status != hipSuccess) {
      std::cerr << "Failed to allocate device memory for the second channel" << std::endl;
      hipFree((void *)output_image.channel[0]);
      rocJpegStreamDestroy(rocjpeg_stream_handle);
      rocJpegDestroy(handle);
      return EXIT_FAILURE;
    }

    // Decode the JPEG stream
    status = rocJpegDecode(rocjpeg_handle, rocjpeg_stream_handle, &decode_params, &output_image);
    if (status != ROCJPEG_STATUS_SUCCESS) {
      std::cerr << "Failed to decode JPEG stream with error code: " << rocJpegGetErrorName(status) << std::endl;
      hipFree((void *)output_image.channel[0]);
      hipFree((void *)output_image.channel[1]);
      rocJpegStreamDestroy(rocjpeg_stream_handle);
      rocJpegDestroy(handle);
      return EXIT_FAILURE;
    }

    // Perform additional post-processing on the decoded image or optionally save it
    // ...

    // Clean up resources
    hipFree((void *)output_image.channel[0]);
    hipFree((void *)output_image.channel[1]);
    rocJpegStreamDestroy(rocjpeg_stream_handle);
    rocJpegDestroy(handle);

    return EXIT_SUCCESS;
  }
