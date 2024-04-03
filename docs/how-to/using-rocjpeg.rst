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

3. Retrieve the image info
====================================================
``rocJpegGetImageInfo()`` retrieves the image info, including number of components, width and height of each component, and chroma subsampling.
For each image to be decoded, pass the JPEG data pointer and data length to the ``rocJpegGetImageInfo()`` function. This function is thread safe.

Below is the signature of ``rocJpegGetImageInfo()`` function:

.. code:: cpp

    RocJpegStatus rocJpegGetImageInfo(
      RocJpegHandle handle,
      const uint8_t *data,
      size_t length,
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

  The VCN hardware-accelerated JPEG decoder in AMD GPUs only supports decoding JPEG images with ``ROCJPEG_CSS_444``, ``ROCJPEG_CSS_422``,
  ``ROCJPEG_CSS_420``, and ``ROCJPEG_CSS_400`` chroma subsampling.

4. Decode a JPEG stream
====================================================
``rocJpegDecode()`` decodes single image based on the backend used to create the rocJpeg handle in rocJpegCreate API. For each image to be decoded,
pass the JPEG data pointer and data length to the ``rocJpegDecode()`` function. This function is thread safe.

See the signature of this function below:

.. code:: cpp

    RocJpegStatus rocJpegDecode(
      RocJpegHandle handle,
      const uint8_t *data,
      size_t length,
      RocJpegOutputFormat output_format,
      RocJpegImage *destination);

In the above ``rocJpegDecode()`` function, you can use the parameters ``RocJpegOutputFormat`` and ``RocJpegImage`` to set
the output behavior of the ``rocJpegDecode()`` function. The ``RocJpegImage`` structure is JPEG image descriptor used to
return the decoded output image. User must allocate device memories for each channel for this structure and pass it to the
``rocJpegDecode()`` API. This API then copies the decoded image to this struct based on the requested output format ``RocJpegOutputFormat``.
Below is the ``RocJpegImage`` structure.

.. code:: cpp

    typedef struct {
      uint8_t* channel[ROCJPEG_MAX_COMPONENT];
      uint32_t pitch[ROCJPEG_MAX_COMPONENT];
    } RocJpegImage;

You can set the ``RocJpegOutputFormat`` parameter to one of the ``output_format`` settings below:

.. csv-table::
  :header: "output_format", "Meaning"

  "ROCJPEG_OUTPUT_NATIVE", "Return native unchanged decoded YUV image from the VCN JPEG deocder."
  "ROCJPEG_OUTPUT_YUV_PLANAR", "Return in the YUV planar format."
  "ROCJPEG_OUTPUT_Y", "Return the Y component only."
  "ROCJPEG_OUTPUT_RGB", "Convert to interleaved RGB."

For example, if ``output_format`` is set to ``ROCJPEG_OUTPUT_NATIVE``, then based on the chroma subsampling of the input image, the
``rocJpegDecode()`` function does one of the following:

* For ``ROCJPEG_CSS_444`` write Y, U, and V to first, second, and third channels of ``RocJpegImage``.
* For ``ROCJPEG_CSS_422`` write YUYV (packed) to first channel of ``RocJpegImage``.
* For ``ROCJPEG_CSS_420`` write Y to first channel and UV (interleaved) to second channel of ``RocJpegImage``.
* For ``ROCJPEG_CSS_400`` write Y to first channel of ``RocJpegImage``.

if ``output_format`` is set to ``ROCJPEG_OUTPUT_Y`` or ``ROCJPEG_OUTPUT_RGB`` then ``rocJpegDecode()`` copies the output to first channel of ``RocJpegImage``.
Alternately, in the case of ``ROCJPEG_OUTPUT_YUV_PLANAR``, the data is written to the corresponding channels of the ``RocJpegImage`` destination structure.
The destination buffers should be large enough to be able to store output of specified format. These buffers should be
pre-allocted by the user in the device memories. For each color plane (channel), sizes could be retrieved for image using
``rocJpegGetImageInfo()`` API and minimum required memory buffer for each plane is plane_height * plane_pitch where
plane_pitch >= plane_width for planar output formats and plane_pitch >= plane_width * num_components for interleaved output format.

As mentioned above, you can use the retrieved parameters, ``num_components``, ``subsampling``, ``widths``, and ``heights`` from the ``rocJpegGetImageInfo()`` API to calculate
the required size for the output buffers for a single decode JPEG. To optimally set the destination parameter for the ``rocJpegDecode()`` function, use the following guidelines:

.. csv-table::
  :header: "output_format", "chroma subsampling", "destination.pitch[c] should be atleast:", "destination.channel[c] should be atleast:"

  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_444", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[c] = destination.pitch[c] * heights[0] for c = 0, 1, 2"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_422", "destination.pitch[0] = widths[0] * 2", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_420", "destination.pitch[1] = destination.pitch[0] = widths[0]", "destination.channel[0] = destination.pitch[0] * heights[0], destination.channel[1] = destination.pitch[1] * (heights[0] >> 1)"
  "ROCJPEG_OUTPUT_NATIVE", "ROCJPEG_CSS_400", "destination.pitch[0] = widhts[0]", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_YUV_PLANAR", "ROCJPEG_CSS_444, ROCJPEG_CSS_422, ROCJPEG_CSS_420", "destination.pitch[c] = widths[c] for c = 0, 1, 2", "destination.channel[c] = destination.pitch[c] * heights[c] for c = 0, 1, 2"
  "ROCJPEG_OUTPUT_YUV_PLANAR", "ROCJPEG_CSS_400", "destination.pitch[0] = widhts[0]", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_Y", "Any of the supported chroma subsampling", "destination.pitch[0] = widhts[0]", "destination.channel[0] = destination.pitch[0] * heights[0]"
  "ROCJPEG_OUTPUT_RGB", "Any of the supported chroma subsampling", "destination.pitch[0] = widhts[0] * 3", "destination.channel[0] = destination.pitch[0] * heights[0]"

5. Destroy the decoder
====================================================

You must call the ``rocJpegDestroy()`` to destroy the session and free up resources.

6. Get Error name
====================================================

You can call ``rocJpegGetErrorName`` to retrieve the name of the specified error code in text form returned from rocJPEG APIs.