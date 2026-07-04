"""
Test auto-shape detection functionality.
"""

import numpy as np
import pandas as pd
import pytest
from videoparquet import parquet2video, video2parquet, infer_video_shape


def test_infer_shape_row_per_frame():
    """Test shape inference for row-per-frame layout."""
    df = pd.DataFrame(np.random.rand(16, 64*64*3))
    shape = infer_video_shape(df)
    assert shape == (16, 64, 64, 3)


def test_infer_shape_row_per_pixel():
    """Test shape inference for row-per-pixel layout."""
    df = pd.DataFrame(np.random.rand(64*64, 3))
    shape = infer_video_shape(df)
    assert shape[1] * shape[2] == 64*64
    assert shape[3] == 3


def test_infer_shape_numpy_array():
    """Test shape inference from numpy array."""
    arr = np.random.rand(16, 32, 32, 3)
    shape = infer_video_shape(arr)
    # Should find some valid shape for the same number of elements
    assert shape[0] * shape[1] * shape[2] * shape[3] == arr.size


def test_auto_shape_roundtrip(tmp_path):
    """Test full roundtrip with auto-detected shape."""
    # Create test data
    arr = np.random.rand(8, 32, 32, 3).astype(np.float32)
    flat = arr.reshape(8, -1)
    df = pd.DataFrame(flat)

    parquet_path = tmp_path / 'test.parquet'
    df.to_parquet(parquet_path)

    # Convert with auto shape
    conversion_rules = {
        'auto_test': (
            list(df.columns),
            'auto',  # Auto-detect shape
            0,
            {'c:v': 'ffv1'},
            16,
            [arr.min(), arr.max()]
        )
    }

    parquet2video(
        str(parquet_path), 'test_id', conversion_rules,
        output_path=str(tmp_path), verbose=False
    )

    # Reconstruct
    recon_path = video2parquet(str(tmp_path), 'test_id', name='auto_test')
    df_recon = pd.read_parquet(recon_path)

    max_err = np.max(np.abs(df.values - df_recon.values))
    assert max_err < 0.001, f"Roundtrip error too high: {max_err}"
