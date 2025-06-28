"""
FFmpeg wrapper functions for reading and writing video files in videoparquet.
Only supports ffv1 with 'gbrp' (planar RGB, 3 channels, no padding) for robust roundtrip.
"""

import ffmpeg
import numpy as np
import ast
import json
import os

def _ffmpeg_write(output_path, array, width, height, params, planar_in=True, input_pix_fmt='gbrp', loglevel='quiet', metadata=None):
    """
    Write a numpy array (T, H, W, C) as a video file using ffmpeg-python.
    For ffv1+RGB, supports float32 input and gbrp16le pixel format.
    """
    vcodec = params.get('c:v', 'libx264')
    # For ffv1+RGB, allow float32 and gbrp16le
    if vcodec == 'ffv1' and input_pix_fmt == 'gbrp16le':
        assert array.shape[-1] == 3, 'Only 3-channel (RGB) supported for ffv1+gbrp16le.'
        # Convert float32 to uint16 for writing
        if array.dtype == np.float32:
            array = np.clip(array, 0, 65535).astype(np.uint16)
    else:
        assert array.dtype == np.uint8, 'Only uint8 supported for non-ffv1+gbrp16le.'
        assert array.shape[-1] == 3, 'Only 3-channel (RGB) supported for now.'
    # Generate a unique XARRAY_ID for this video (e.g., based on output_path)
    base = os.path.splitext(os.path.basename(str(output_path)))[0]
    xarray_id = f"XARRAY_{base}"
    params = dict(params)
    params['metadata'] = f'XARRAY_ID={xarray_id}'
    # Store full metadata in a sidecar .json file
    sidecar_path = f"{output_path}_{xarray_id}.json"
    with open(sidecar_path, 'w') as f:
        json.dump(metadata, f)
    # Always use gbrp16le as input for ffv1+RGB
    if vcodec == 'ffv1' and input_pix_fmt == 'gbrp16le':
        input_pix_fmt = 'gbrp16le'
        channel_order = 'gbr'
    else:
        input_pix_fmt = 'rgb24'
        channel_order = 'rgb'
    # Define the input pipe
    input_pipe = ffmpeg.input('pipe:', format='rawvideo', pix_fmt=input_pix_fmt, s=f'{width}x{height}', framerate=30)
    process = (
        ffmpeg
        .output(input_pipe, str(output_path), loglevel=loglevel, **params)
        .overwrite_output()
        .run_async(pipe_stdin=True, overwrite_output=True)
    )
    # Convert to planar if needed
    if planar_in:
        array2 = np.transpose(array, (0, 3, 2, 1))  # (t, x, y, c) > (t, c, y, x)
    else:
        array2 = np.transpose(array, (0, 2, 1, 3))  # (t, x, y, c) > (t, y, x, c)
    for frame in array2:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()
    # Get actual pixel format
    try:
        actual_pix_fmt = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=pix_fmt',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(output_path)
        ]).decode().strip()
    except Exception:
        actual_pix_fmt = input_pix_fmt
    metadata['ACTUAL_PIX_FMT'] = actual_pix_fmt
    # Store metadata as XARRAY tag
    params['metadata'] = f'XARRAY={metadata}'
    # Overwrite metadata in file (ffmpeg-python doesn't update after run, so do nothing here)
    return actual_pix_fmt

def _ffmpeg_read(input_path, loglevel='quiet'):
    """
    Read a video file into a numpy array (T, H, W, C) using ffmpeg-python.
    For ffv1+gbrp16le, returns uint16 array (can be denormalized to float32 if needed).
    """
    probe = ffmpeg.probe(str(input_path))
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    tags = probe['format'].get('tags', {})
    # Always load full metadata from sidecar .json file using XARRAY_ID
    if 'XARRAY_ID' in tags:
        xarray_id = tags['XARRAY_ID']
        sidecar_path = f"{input_path}_{xarray_id}.json"
        if not os.path.exists(sidecar_path):
            raise RuntimeError(f'Missing sidecar metadata file: {sidecar_path}')
        with open(sidecar_path, 'r') as f:
            meta_info = json.load(f)
    else:
        raise RuntimeError('Missing XARRAY_ID metadata in video file.')
    width = int(video_info['width'])
    height = int(video_info['height'])
    actual_pix_fmt = video_info['pix_fmt']
    requested_pix_fmt = meta_info['REQ_PIX_FMT']
    output_pix_fmt = meta_info['OUT_PIX_FMT']
    planar_out = meta_info['PLANAR'] in [True, 'True']
    bits = int(meta_info['BITS'])
    channels = int(meta_info['CHANNELS'])
    num_frames = int(meta_info['FRAMES'])
    process = (
        ffmpeg
        .input(str(input_path))
        .output('pipe:', format='rawvideo', pix_fmt=output_pix_fmt, loglevel=loglevel)
        .run_async(pipe_stdout=True)
    )
    if planar_out:
        output_shape = [num_frames, channels, height, width]
    else:
        output_shape = [num_frames, height, width, channels]
    # For ffv1+gbrp16le, read as uint16
    if output_pix_fmt == 'gbrp16le':
        data = np.frombuffer(process.stdout.read(), np.uint16)
    else:
        data = np.frombuffer(process.stdout.read(), np.uint8)
    assert len(data) == np.prod(output_shape), f'Video {len(data)=} cannot be reshaped into {output_shape=}. Video data is {len(data)/np.prod(output_shape):.6f}x longer'
    video_data = data.reshape(output_shape)
    process.stdout.close()
    process.wait()
    # Convert back from planar if needed
    if planar_out:
        video_data = np.transpose(video_data, (0, 3, 2, 1))  # (t, c, y, x) > (t, x, y, c)
    else:
        video_data = np.transpose(video_data, (0, 2, 1, 3))  # (t, y, x, c) > (t, x, y, c)
    return video_data, meta_info 