"""
Test roundtrip conversion: Parquet -> Video -> Parquet
"""

import pandas as pd
import numpy as np
import tempfile
import shutil
from videoparquet.parquet2video import parquet2video
from videoparquet.video2parquet import video2parquet
import subprocess
import pytest
from videoparquet.utils import normalize, denormalize, reorder_coords_axis
from videoparquet.ffmpeg_wrappers import _ffmpeg_write, _ffmpeg_read
import os

def test_ffv1_gbrp16le_minimal(tmp_path):
    # Minimal test data: 2 frames, 4x4, 3 channels, float32
    arr = np.arange(2*4*4*3, dtype=np.float32).reshape(2, 4, 4, 3)
    minmax = np.stack([[arr[...,c].min(), arr[...,c].max()] for c in range(3)])
    bits = 16
    # Normalize
    arr_norm = normalize(arr, minmax, bits=bits)
    print("Normalized array before writing:\n", arr_norm)
    # Reorder RGB->GBR for gbrp16le
    arr_gbr = reorder_coords_axis(arr_norm, ['r','g','b'], ['g','b','r'], axis=-1)
    # Write video
    outpath = os.path.join(tmp_path, 'test_ffv1.mkv')
    metadata = {
        'REQ_PIX_FMT': 'gbrp16le',
        'OUT_PIX_FMT': 'gbrp16le',
        'PLANAR': True,
        'BITS': bits,
        'CHANNELS': 3,
        'FRAMES': arr.shape[0],
        'RANGE': minmax.tolist(),
        'NORMALIZED': True,
        'CHANNEL_ORDER': 'gbr',
    }
    _ffmpeg_write(outpath, arr_gbr, 4, 4, {'c:v':'ffv1'}, planar_in=True, input_pix_fmt='gbrp16le', metadata=metadata)
    # Read video
    arr_read, meta = _ffmpeg_read(outpath)
    print("Array after reading, before denormalization:\n", arr_read)
    # Reorder GBR->RGB
    arr_rgb = reorder_coords_axis(arr_read, ['g','b','r'], ['r','g','b'], axis=-1)
    # Denormalize
    arr_denorm = denormalize(arr_rgb, minmax, bits=bits)
    print("Denormalized array:\n", arr_denorm)
    # Compare
    maxerr = np.max(np.abs(arr - arr_denorm), axis=(0,1,2))
    print("Max abs error per channel:", maxerr)
    assert np.all(maxerr < 1e-3), f"Lossless roundtrip failed: maxerr={maxerr}"

def test_libx264_lossy_minimal(tmp_path):
    # Same test data as the lossless test
    arr = np.arange(2*4*4*3, dtype=np.float32).reshape(2, 4, 4, 3)
    minmax = np.stack([[arr[...,c].min(), arr[...,c].max()] for c in range(3)])
    bits = 8
    # Normalize to 8 bits for libx264
    arr_norm = normalize(arr, minmax, bits=bits)
    print("[libx264] Normalized array before writing:\n", arr_norm)
    # No channel reordering needed for rgb24
    # Write video
    outpath = os.path.join(tmp_path, 'test_libx264.mkv')
    metadata = {
        'REQ_PIX_FMT': 'rgb24',
        'OUT_PIX_FMT': 'rgb24',
        'PLANAR': False,
        'BITS': bits,
        'CHANNELS': 3,
        'FRAMES': arr.shape[0],
        'RANGE': minmax.tolist(),
        'NORMALIZED': True,
        'CHANNEL_ORDER': 'rgb',
    }
    _ffmpeg_write(outpath, arr_norm, 4, 4, {'c:v':'libx264'}, planar_in=False, input_pix_fmt='rgb24', metadata=metadata)
    # Read video
    arr_read, meta = _ffmpeg_read(outpath)
    print("[libx264] Array after reading, before denormalization:\n", arr_read)
    # No channel reordering needed
    arr_rgb = arr_read
    # Denormalize
    arr_denorm = denormalize(arr_rgb, minmax, bits=bits)
    print("[libx264] Denormalized array:\n", arr_denorm)
    # Compare
    maxerr = np.max(np.abs(arr - arr_denorm), axis=(0,1,2))
    print("[libx264] Max abs error per channel:", maxerr)
    # No assertion, just print error for inspection 