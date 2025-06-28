import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
from .utils import normalize, denormalize, is_float, DRWrapper
from .ffmpeg_wrappers import _ffmpeg_write
from .metadata import save_metadata
import subprocess

def parquet2video(parquet_path, array_id, conversion_rules, compute_stats=False, include_data_in_stats=False,
                 output_path='./', fmt='auto', loglevel='quiet', exceptions='raise', 
                 verbose=True, nan_fill=None, all_zeros_is_nan=True, save_dataset=True,
                 metrics_value_range=None):
    '''
    Converts a Parquet file (containing array-like or tabular data) into video files, storing metadata for reconstruction.
    Parameters:
        parquet_path: Path to the input Parquet file.
        array_id: Identifier for the dataset.
        conversion_rules: Dict specifying which columns/arrays to convert and how.
        ... (other parameters as in xarray2video)
    '''
    print_fn = print if verbose else (lambda *a, **k: None)
    results = {}
    df = pd.read_parquet(parquet_path)
    output_path = Path(output_path)
    (output_path / array_id).mkdir(exist_ok=True, parents=True)

    for name, config in conversion_rules.items():
        # Unpack config: columns, shape, n_components, params, bits, value_range
        bits = 8
        n_components = 'all'
        value_range = None
        params = {'c:v': 'libx265', 'preset': 'medium', 'crf': 3}
        if len(config) == 6:
            columns, shape, n_components, params, bits, value_range = config
        elif len(config) == 5:
            columns, shape, n_components, params, bits = config
        elif len(config) == 4:
            columns, shape, n_components, params = config
        elif len(config) == 3:
            columns, shape, n_components = config
        else:
            raise AssertionError(f'Params: {config} should be: columns, shape, [n_components], [params], [bits], [min, max]')

        try:
            # Extract data as numpy array
            if isinstance(columns, str):
                columns = [columns]
            array = df[columns].values
            # Reshape if needed
            if shape is not None:
                array = array.reshape(shape)
            # Save a copy for stats
            if compute_stats:
                array_orig = array.copy()
            # NaN handling
            if nan_fill is not None:
                array_nans = np.isnan(array)
                if isinstance(nan_fill, int):
                    array[array_nans] = nan_fill
                elif nan_fill == 'mean':
                    array[array_nans] = np.nanmean(array)
                elif nan_fill == 'min':
                    array[array_nans] = np.nanmin(array)
                elif nan_fill == 'max':
                    array[array_nans] = np.nanmax(array)
                else:
                    raise AssertionError(f'{nan_fill=}?')
            # PCA
            use_pca = (isinstance(n_components, int) and n_components > 0)
            pca_params = None
            if use_pca:
                DR = DRWrapper(n_components=n_components)
                array = DR.fit_transform(array)
                pca_params = DR.get_params_str()
            # Normalization
            is_int_but_does_not_fit = (array.dtype.itemsize * 8 > bits) and np.nanmax(array) > (2**bits-1)
            normalized = is_float(array) or is_int_but_does_not_fit
            if normalized:
                if value_range is None:
                    value_range = [np.nanmin(array), np.nanmax(array)]
                array = normalize(array, minmax=value_range, bits=bits)
            # Pad to 3 channels if needed
            if array.shape[-1] < 3:
                pad_width = 3 - array.shape[-1]
                pad_shape = list(array.shape[:-1]) + [pad_width]
                pad = np.zeros(pad_shape, dtype=array.dtype)
                array = np.concatenate([array, pad], axis=-1)
            # Assume array is (T, H, W, C) and uint8 for now
            # Choose extension based on codec
            ext = '.mp4' if params.get('c:v', 'libx264') == 'libx264' else '.mkv'
            video_path = output_path / array_id / f'{name}{ext}'
            t0 = time.time()
            output_pix_fmt = _ffmpeg_write(str(video_path), array, array.shape[2], array.shape[1], params, loglevel=loglevel)
            # Get actual pixel format from ffprobe
            try:
                actual_pix_fmt = subprocess.check_output([
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=pix_fmt',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(video_path)
                ]).decode().strip()
                if params.get('c:v', 'libx264') == 'ffv1' and actual_pix_fmt != 'gbrp':
                    raise RuntimeError(f"ffv1 only supported with gbrp pixel format, got {actual_pix_fmt}. "
                                       f"Try a different ffmpeg build or codec.")
            except Exception as e:
                raise RuntimeError(f"Could not verify pixel format with ffprobe: {e}")
            t1 = time.time()
            # Save metadata (convert numpy types to native types)
            metadata = {
                'shape': list(map(int, array.shape)),
                'minmax': list(map(float, value_range)),
                'columns': list(map(str, columns)),
                'name': str(name),
                'bits': int(bits),
                'normalized': bool(normalized),
                'pca_params': pca_params,
                'ext': ext,
                'output_pix_fmt': output_pix_fmt,  # Requested
                'actual_pix_fmt': actual_pix_fmt   # Actual from ffprobe
            }
            meta_path = output_path / array_id / f'{name}.json'
            save_metadata(metadata, meta_path)
            # Stats
            original_size = array.size * array.itemsize / 2**20  # MB
            compressed_size = os.stat(video_path).st_size / 2**20  # MB
            compression = compressed_size / original_size if original_size > 0 else None
            bpppb = compressed_size * 2**20 * 8 / array.size if array.size > 0 else None
            results[name] = {
                'path': str(video_path),
                'metadata': str(meta_path),
                'original_size_MB': original_size,
                'compressed_size_MB': compressed_size,
                'compression_ratio': compression,
                'bpppb': bpppb,
                'write_time_s': t1 - t0
            }
            print_fn(f"Wrote video for '{name}': {video_path}, shape={array.shape}, dtype={array.dtype}, normalized={normalized}, PCA={use_pca}")
            if compute_stats:
                print_fn(f"Stats for '{name}': original_size={original_size:.2f}MB, compressed_size={compressed_size:.2f}MB, compression={compression:.2f}, bpppb={bpppb:.4f}, write_time={t1-t0:.2f}s")
        except Exception as e:
            print(f'Exception processing {array_id=} {name=}: {e}')
            if exceptions == 'raise':
                raise e
    # Optionally save the DataFrame
    if save_dataset:
        df.to_parquet(output_path / array_id / 'data.parquet')
    return results 