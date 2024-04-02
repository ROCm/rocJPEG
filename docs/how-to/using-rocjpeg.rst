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

``rocJpegCreate()`` creates an instance of the hardware JPEG decoder object and provides you
with a handle upon successful creation. The decoder handle
returned by ``rocJpegCreate()`` must be retained for the entire decode session because the
handle is passed along with the other decoding APIs.

3. Retrieve the image info
====================================================
``rocJpegGetImageInfo()`` retrieves the image info, including channel, width and height of each component, and chroma subsampling.

4. Decode a JPEG stream
====================================================
``rocJpegDecode()`` decodes single image based on the backend used to create the rocJpeg handle in rocJpegCreate API.
Destination buffers should be large enough to be able to store output of specified format. These buffers should be pre-allocted by the user in the device memories.
For each color plane (channel) sizes could be retrieved for image using ``rocJpegGetImageInfo()`` API
and minimum required memory buffer for each plane is plane_height * plane_pitch where plane_pitch >= plane_width for
planar output formats and plane_pitch >= plane_width * num_components for interleaved output format.

5.  Destroy the decoder
====================================================

You must call the ``rocJpegDestroy()`` to destroy the session and free up resources.