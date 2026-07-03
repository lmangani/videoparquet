"""
Native video encoding/decoding using PyAV.
No external ffmpeg binary required - PyAV bundles libav libraries.
"""

import av
import numpy as np
import json
import os


def write_video(output_path, array, width, height, codec='ffv1', params=None,
                pix_fmt='gbrp16le', metadata=None):
    """
    Write a numpy array (T, H, W, C) as a video file.

    Parameters
    ----------
    output_path : str
        Path for the output video file.
    array : np.ndarray
        Video data with shape (frames, height, width, channels).
    width : int
        Frame width.
    height : int
        Frame height.
    codec : str
        Video codec ('ffv1' for lossless, 'libx264' for lossy).
    params : dict
        Additional codec parameters.
    pix_fmt : str
        Pixel format ('gbrp16le' for 16-bit planar, 'rgb24' for 8-bit packed).
    metadata : dict
        Metadata to store alongside the video.

    Returns
    -------
    str
        The actual pixel format used.
    """
    params = params or {}

    # Validate and convert input dtype
    if codec == 'ffv1' and pix_fmt == 'gbrp16le':
        if array.shape[-1] != 3:
            raise ValueError('gbrp16le requires exactly 3 channels')
        if array.dtype == np.float32:
            array = np.clip(array, 0, 65535).astype(np.uint16)
        elif array.dtype != np.uint16:
            array = array.astype(np.uint16)
        encode_pix_fmt = 'gbrp16le'
    else:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] != 3:
            raise ValueError('Only 3-channel arrays supported')
        pix_fmt = 'rgb24'
        encode_pix_fmt = 'yuv420p' if codec == 'libx264' else 'rgb24'

    # Generate metadata ID and save sidecar
    base = os.path.splitext(os.path.basename(output_path))[0]
    meta_id = f"VPARQUET_{base}"
    sidecar_path = f"{output_path}.meta.json"
    with open(sidecar_path, 'w') as f:
        json.dump(metadata or {}, f)

    # Determine container format
    ext = os.path.splitext(output_path)[1].lower()
    container_format = 'matroska' if ext == '.mkv' else 'mp4'

    # Create container and stream
    container = av.open(output_path, mode='w', format=container_format)
    stream = container.add_stream(codec, rate=30)
    stream.width = width
    stream.height = height
    stream.pix_fmt = encode_pix_fmt

    # Codec-specific options
    if codec == 'ffv1':
        stream.options = {'slicecrc': '1'}
    elif codec == 'libx264':
        crf = params.get('crf', 23)
        preset = params.get('preset', 'medium')
        stream.options = {'crf': str(crf), 'preset': preset}

    # Store metadata reference
    container.metadata['VPARQUET_ID'] = meta_id

    # Write frames
    for i in range(array.shape[0]):
        frame_data = array[i]  # (H, W, C)

        if pix_fmt == 'gbrp16le':
            frame = av.VideoFrame(width, height, 'gbrp16le')
            planar_data = np.transpose(frame_data, (2, 0, 1))  # (C, H, W)

            for plane_idx in range(3):
                plane = frame.planes[plane_idx]
                plane_array = planar_data[plane_idx]
                line_size_elements = plane.line_size // 2

                if line_size_elements > width:
                    padded = np.zeros((height, line_size_elements), dtype=np.uint16)
                    padded[:, :width] = plane_array
                    plane.update(padded.tobytes())
                else:
                    plane.update(np.ascontiguousarray(plane_array).tobytes())
        else:
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(frame_data), format='rgb24'
            )
            if encode_pix_fmt != 'rgb24':
                frame = frame.reformat(format=encode_pix_fmt)

        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return encode_pix_fmt


def read_video(input_path):
    """
    Read a video file into a numpy array (T, H, W, C).

    Parameters
    ----------
    input_path : str
        Path to the video file.

    Returns
    -------
    tuple
        (video_data, metadata) where video_data has shape (T, H, W, C).
    """
    container = av.open(input_path)

    # Load metadata from sidecar
    sidecar_path = f"{input_path}.meta.json"
    if not os.path.exists(sidecar_path):
        container.close()
        raise RuntimeError(f'Missing metadata file: {sidecar_path}')

    with open(sidecar_path, 'r') as f:
        metadata = json.load(f)

    stream = container.streams.video[0]
    width = stream.width
    height = stream.height
    actual_pix_fmt = stream.pix_fmt

    output_pix_fmt = metadata.get('OUT_PIX_FMT', 'rgb24')
    num_frames = int(metadata.get('FRAMES', 0))

    # Read frames
    frames = []
    for frame in container.decode(video=0):
        if output_pix_fmt == 'gbrp16le':
            planes_data = []
            for plane_idx in range(3):
                plane = frame.planes[plane_idx]
                plane_bytes = bytes(plane)
                plane_array = np.frombuffer(plane_bytes, dtype=np.uint16)

                line_size = plane.line_size // 2
                if line_size != frame.width:
                    plane_array = plane_array.reshape(frame.height, line_size)[:, :frame.width]
                else:
                    plane_array = plane_array.reshape(frame.height, frame.width)
                planes_data.append(plane_array.copy())

            frame_data = np.stack(planes_data, axis=0)
            frame_data = np.transpose(frame_data, (1, 2, 0))
        else:
            if actual_pix_fmt != output_pix_fmt:
                frame = frame.reformat(format=output_pix_fmt)
            frame_data = frame.to_ndarray(format=output_pix_fmt)
            if frame_data.ndim == 2:
                frame_data = frame_data[..., np.newaxis]

        frames.append(frame_data)

    container.close()

    video_data = np.stack(frames, axis=0)

    # Verify shape
    expected_shape = metadata.get('shape')
    if expected_shape and video_data.size != np.prod(expected_shape):
        raise ValueError(f"Size mismatch: got {video_data.size}, expected {np.prod(expected_shape)}")

    return video_data, metadata
