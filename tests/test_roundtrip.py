"""
Test lossless and lossy roundtrip conversion.
"""

import numpy as np
import os
import pytest
from videoparquet.utils import normalize, denormalize, reorder_coords_axis
from videoparquet.av_wrappers import write_video, read_video


def test_ffv1_lossless_roundtrip(tmp_path):
    """Test lossless roundtrip with ffv1 + gbrp16le."""
    # Create test data: 2 frames, 4x4, 3 channels
    arr = np.arange(2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3)
    minmax = np.stack([[arr[..., c].min(), arr[..., c].max()] for c in range(3)])
    bits = 16

    # Normalize to 16-bit
    arr_norm = normalize(arr, minmax, bits=bits)

    # Reorder RGB -> GBR for gbrp16le
    arr_gbr = reorder_coords_axis(arr_norm, ['r', 'g', 'b'], ['g', 'b', 'r'], axis=-1)

    # Write video
    video_path = str(tmp_path / 'test_ffv1.mkv')
    metadata = {
        'OUT_PIX_FMT': 'gbrp16le',
        'PLANAR': True,
        'BITS': bits,
        'CHANNELS': 3,
        'FRAMES': arr.shape[0],
        'CHANNEL_ORDER': 'gbr',
        'shape': list(arr.shape),
    }
    write_video(video_path, arr_gbr, width=4, height=4, codec='ffv1',
                pix_fmt='gbrp16le', metadata=metadata)

    # Read video
    arr_read, _ = read_video(video_path)

    # Reorder GBR -> RGB
    arr_rgb = reorder_coords_axis(arr_read, ['g', 'b', 'r'], ['r', 'g', 'b'], axis=-1)

    # Denormalize
    arr_denorm = denormalize(arr_rgb, minmax, bits=bits)

    # Compare
    max_err = np.max(np.abs(arr - arr_denorm), axis=(0, 1, 2))
    print(f"Max abs error per channel: {max_err}")
    assert np.all(max_err < 1e-3), f"Lossless roundtrip failed: max_err={max_err}"


def test_libx264_lossy_roundtrip(tmp_path):
    """Test lossy roundtrip with libx264."""
    arr = np.arange(2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3)
    minmax = np.stack([[arr[..., c].min(), arr[..., c].max()] for c in range(3)])
    bits = 8

    # Normalize to 8-bit
    arr_norm = normalize(arr, minmax, bits=bits)

    # Write video
    video_path = str(tmp_path / 'test_x264.mkv')
    metadata = {
        'OUT_PIX_FMT': 'rgb24',
        'PLANAR': False,
        'BITS': bits,
        'CHANNELS': 3,
        'FRAMES': arr.shape[0],
        'CHANNEL_ORDER': 'rgb',
        'shape': list(arr.shape),
    }
    write_video(video_path, arr_norm, width=4, height=4, codec='libx264',
                pix_fmt='rgb24', metadata=metadata)

    # Read video
    arr_read, _ = read_video(video_path)

    # Denormalize
    arr_denorm = denormalize(arr_read, minmax, bits=bits)

    # Compare - lossy codec will have some error
    max_err = np.max(np.abs(arr - arr_denorm), axis=(0, 1, 2))
    print(f"[libx264] Max abs error per channel: {max_err}")
    # Just verify it's reasonable (lossy compression)
    assert np.all(max_err < 10), f"Lossy error too high: max_err={max_err}"
