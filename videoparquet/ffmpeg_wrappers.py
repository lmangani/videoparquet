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
        if vcodec == 'ffv1' and actual_pix_fmt != 'gbrp':
            raise RuntimeError(f"ffv1 only supported with gbrp pixel format, got {actual_pix_fmt}. "
                               f"Try a different ffmpeg build or codec.")
    except Exception as e:
        raise RuntimeError(f"Could not verify pixel format with ffprobe: {e}")
    return output_pix_fmt

def _ffmpeg_read(input_path, width, height, num_frames, input_pix_fmt='rgb24', vcodec=None):
    """
    Read a video file into a numpy array (T, H, W, C) using ffmpeg-python.
    Only supports gbrp (planar RGB, 3 channels, no padding) for ffv1.
    Assumes output is uint8.
    Raises an error if the pixel format is not supported.
    """
    try:
        # If vcodec is not provided, try to infer from file extension
        if vcodec is None and input_path.endswith('.mkv'):
            vcodec = 'ffv1'
        if vcodec == 'ffv1' or input_pix_fmt == 'gbrp':
            pix_fmt = 'gbrp'
            out, err = (
                ffmpeg
                .input(str(input_path))
                .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                .run(capture_stdout=True, capture_stderr=True)
            )
            array = np.frombuffer(out, np.uint8)
            array = array.reshape((num_frames, height, width, 3))
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