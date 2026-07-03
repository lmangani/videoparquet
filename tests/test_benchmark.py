"""
Benchmark test for videoparquet compression.
"""

import pandas as pd
import numpy as np
import os
import time
import shutil
import pytest
from videoparquet import parquet2video, video2parquet


def test_parquet_video_benchmark(tmp_path):
    """Test compression and roundtrip with realistic data."""
    # Generate synthetic data
    num_frames, height, width, channels = 16, 64, 64, 3
    shape = (num_frames, height, width, channels)
    arr = (np.random.rand(*shape) * 255).astype(np.uint8)

    flat = arr.reshape(num_frames, -1)
    columns = [f'col{i}' for i in range(flat.shape[1])]
    df = pd.DataFrame(flat, columns=columns)
    minmax = [arr.min(), arr.max()]

    # Save test parquet
    parquet_path = tmp_path / 'data.parquet'
    df.to_parquet(parquet_path)
    parquet_size = os.stat(parquet_path).st_size / 2**20

    print(f'Parquet file size: {parquet_size:.2f} MB')

    # Convert to video
    conversion_rules = {
        'arr_lossless': (columns, shape, 0, {'c:v': 'ffv1'}, 8, minmax)
    }

    t0 = time.time()
    results = parquet2video(
        str(parquet_path), 'benchid', conversion_rules,
        compute_stats=True, output_path=str(tmp_path), save_dataset=False
    )
    t1 = time.time()

    print(f'Parquet -> Video time: {t1 - t0:.2f}s')

    for name, stats in results.items():
        print(f"Video '{name}': {stats['compressed_size_MB']:.2f} MB, "
              f"ratio={stats['compression_ratio']:.2f}")

    # Convert back to parquet
    t0 = time.time()
    recon_path = video2parquet(str(tmp_path), 'benchid', name='arr_lossless')
    t1 = time.time()

    print(f'Video -> Parquet time: {t1 - t0:.2f}s')

    # Verify roundtrip
    df_recon = pd.read_parquet(recon_path)
    max_err = np.max(np.abs(df.values - df_recon.values))
    print(f'Max absolute error: {max_err}')

    assert np.allclose(df.values, df_recon.values, atol=1), 'Roundtrip mismatch'

    # Verify compression
    video_size = os.stat(tmp_path / 'benchid' / 'arr_lossless.mkv').st_size / 2**20
    print(f'Video file size: {video_size:.2f} MB')
    assert video_size < parquet_size, 'Video not smaller than Parquet'
