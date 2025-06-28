import pandas as pd
import numpy as np
import tempfile
import os
import time
import subprocess
from videoparquet.parquet2video import parquet2video
from videoparquet.video2parquet import video2parquet

def test_parquet_video_benchmark():
    # Generate synthetic data as uint8 to match video output
    num_frames, height, width, channels = 16, 64, 64, 3
    shape = (num_frames, height, width, channels)
    arr = (np.random.rand(*shape) * 255).astype(np.uint8)
    flat = arr.reshape(num_frames, -1)
    columns = [f'col{i}' for i in range(flat.shape[1])]
    df = pd.DataFrame(flat, columns=columns)
    minmax = [arr.min(), arr.max()]

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, 'data.parquet')
        df.to_parquet(parquet_path)
        parquet_size = os.stat(parquet_path).st_size / 2**20  # MB
        print(f'Parquet file size: {parquet_size:.2f} MB')

        # Conversion rule: only lossless codec
        conversion_rules = {
            'arr_lossless': (columns, shape, 0, {'c:v': 'ffv1'}, 8, minmax)
        }
        # Parquet -> Video
        t0 = time.time()
        results = parquet2video(parquet_path, 'benchid', conversion_rules, compute_stats=True, output_path=tmpdir, save_dataset=False)
        t1 = time.time()
        print(f'Parquet -> Video total time: {t1-t0:.2f}s')
        for name, stats in results.items():
            print(f"Video '{name}': Compressed size = {stats['compressed_size_MB']:.2f} MB, "
                  f"Compression ratio = {stats['compression_ratio']:.2f}, "
                  f"bpppb = {stats['bpppb']:.4f}, Write time = {stats['write_time_s']:.2f}s")
            # ffprobe pixel format
            video_path = os.path.join(tmpdir, 'benchid', f'{name}.mkv')
            try:
                pix_fmt = subprocess.check_output([
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=pix_fmt',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    video_path
                ]).decode().strip()
                print(f"ffprobe pixel format for {name}: {pix_fmt}")
            except Exception as e:
                print(f"ffprobe failed: {e}")

        # Video -> Parquet (restore)
        for name in conversion_rules.keys():
            t0 = time.time()
            recon_path = video2parquet(tmpdir, 'benchid', name=name)
            t1 = time.time()
            recon_size = os.stat(recon_path).st_size / 2**20  # MB
            df_recon = pd.read_parquet(recon_path)
            print(f"Restored Parquet '{name}': Size = {recon_size:.2f} MB, Restore time = {t1-t0:.2f}s")
            assert df_recon.shape == df.shape, f"Restored DataFrame shape mismatch for {name}"
            assert df_recon.dtypes.equals(df.dtypes), f"Restored DataFrame dtype mismatch for {name}" 