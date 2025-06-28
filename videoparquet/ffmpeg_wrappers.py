"""
FFmpeg wrapper functions for reading and writing video files in videoparquet.
Only supports ffv1 with 'gbrp' (planar RGB, 3 channels, no padding) for robust roundtrip.
"""

import ffmpeg
import numpy as np

def _ffmpeg_write(output_path, array, width, height, params, loglevel='quiet', input_pix_fmt='rgb24'):
    """
    Write a numpy array (T, H, W, C) as a video file using ffmpeg-python.
    Assumes array is uint8 and C=3 (RGB).
    For ffv1, uses gbrp (planar RGB, 3 channels, no padding) for compatibility.
    Returns the output pixel format used.
    Raises an error if ffprobe reports a different pixel format.
    """
    assert array.dtype == np.uint8, 'Only uint8 supported for now.'
    assert array.shape[-1] == 3, 'Only 3-channel (RGB) supported for now.'
    vcodec = params.get('c:v', 'libx264')
    if vcodec == 'ffv1':
        output_pix_fmt = 'gbrp'
        planar_array = array
    else:
        output_pix_fmt = 'rgb24'
        planar_array = array
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt=input_pix_fmt, s=f'{width}x{height}')
        .output(str(output_path), vcodec=vcodec, pix_fmt=output_pix_fmt, r=30, loglevel=loglevel)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    # Write frames
    process.stdin.write(planar_array.tobytes())
    process.stdin.close()
    process.wait()
    # Check actual pixel format
    import subprocess
    try:
        actual_pix_fmt = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=pix_fmt',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(output_path)
        ]).decode().strip()
        if vcodec == 'ffv1':
            if actual_pix_fmt == 'gbrp':
                pass  # All good
            elif actual_pix_fmt == 'bgr0':
                print("WARNING: ffmpeg encoded ffv1 with bgr0 instead of gbrp. This is common on macOS/Homebrew and some Linux builds. bgr0 may have padding/alpha and is NOT guaranteed to be robust for scientific roundtrip. True lossless roundtrip is only guaranteed with gbrp. See README for details.")
            else:
                raise RuntimeError(f"ffv1 only supported with gbrp or bgr0 pixel format, got {actual_pix_fmt}. Try a different ffmpeg build or codec.")
    except Exception as e:
        raise RuntimeError(f"Could not verify pixel format with ffprobe: {e}")
    return output_pix_fmt

def _ffmpeg_read(input_path, width, height, num_frames, input_pix_fmt='rgb24', vcodec=None):
    """
    Read a video file into a numpy array (T, H, W, C) using ffmpeg-python.
    Robustly supports both gbrp and bgr0 for ffv1, converting bgr0 to RGB and dropping alpha if needed.
    Assumes output is uint8.
    Raises an error if the pixel format is not supported.
    """
    import subprocess
    try:
        # If vcodec is not provided, try to infer from file extension
        if vcodec is None and input_path.endswith('.mkv'):
            vcodec = 'ffv1'
        # Detect actual pixel format using ffprobe
        actual_pix_fmt = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=pix_fmt',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(input_path)
        ]).decode().strip()
        if vcodec == 'ffv1':
            if actual_pix_fmt == 'gbrp':
                pix_fmt = 'gbrp'
                out, err = (
                    ffmpeg
                    .input(str(input_path))
                    .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                    .run(capture_stdout=True, capture_stderr=True)
                )
                array = np.frombuffer(out, np.uint8)
                array = array.reshape((num_frames, height, width, 3))
                return array
            elif actual_pix_fmt == 'bgr0':
                pix_fmt = 'bgr0'
                out, err = (
                    ffmpeg
                    .input(str(input_path))
                    .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                    .run(capture_stdout=True, capture_stderr=True)
                )
                array = np.frombuffer(out, np.uint8)
                n_pixels = num_frames * height * width
                if array.size == n_pixels * 4:
                    # Tightly packed, no stride
                    array = array.reshape((num_frames, height, width, 4))
                    array = array[..., :3][..., ::-1]  # BGR0 (drop alpha, BGR->RGB)
                    return array
                # Calculate stride (row size in bytes, aligned to 16 bytes)
                row_stride = ((width * 4 + 15) // 16) * 16
                expected_size = num_frames * height * row_stride
                if array.size == expected_size:
                    # Extract only the first width*4 bytes per row
                    array_reshaped = array.reshape((num_frames, height, row_stride))
                    rgb_frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)
                    for t in range(num_frames):
                        for y in range(height):
                            row = array_reshaped[t, y, :width*4]
                            bgr0 = row.reshape((width, 4))
                            rgb_frames[t, y] = bgr0[:, :3][:, ::-1]  # BGR0 -> RGB
                    return rgb_frames
                elif array.size == n_pixels * 3:
                    array = array.reshape((num_frames, height, width, 3))
                    array = array[..., ::-1]  # BGR->RGB
                    return array
                else:
                    raise ValueError(f"Unexpected buffer size for bgr0: {array.size}, expected {n_pixels*4}, {expected_size} or {n_pixels*3}")
            else:
                out, err = (
                    ffmpeg
                    .input(str(input_path))
                    .output('pipe:', format='rawvideo', pix_fmt=input_pix_fmt)
                    .run(capture_stdout=True, capture_stderr=True)
                )
                array = np.frombuffer(out, np.uint8)
                array = array.reshape((num_frames, height, width, 3))
                return array
        else:
            out, err = (
                ffmpeg
                .input(str(input_path))
                .output('pipe:', format='rawvideo', pix_fmt=input_pix_fmt)
                .run(capture_stdout=True, capture_stderr=True)
            )
            array = np.frombuffer(out, np.uint8)
            array = array.reshape((num_frames, height, width, 3))
            return array
    except ffmpeg.Error as e:
        print('ffmpeg error:', e.stderr.decode())
        raise
    except Exception as e:
        print(f'Exception in _ffmpeg_read: {e}')
        raise 