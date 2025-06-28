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
                 metrics_value_range=None, arrays=None):
    '''
    Converts a Parquet file (containing array-like or tabular data) into video files, storing metadata for reconstruction.
    Parameters:
        parquet_path: Path to the input Parquet file (or None if using arrays).
        array_id: Identifier for the dataset.
        conversion_rules: Dict specifying which columns/arrays to convert and how.
        arrays: Optional dict of numpy arrays to use directly if columns is None.
        ... (other parameters as in xarray2video)
    '''
    print_fn = print if verbose else (lambda *a, **k: None)
    results = {}
    if parquet_path is not None:
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
            if columns is None and arrays is not None:
                array = arrays[name]
            else:
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
            # For ffv1+RGB, upcast to float32 and use bits=16 for normalization
            vcodec = params.get('c:v', 'libx264')
            ordering = 'rgb'
            input_pix_fmt = 'rgb24'
            if vcodec == 'ffv1' and array.shape[-1] == 3:
                if array.dtype != np.float32:
                    array = array.astype(np.float32)
                bits = 16
                input_pix_fmt = 'gbrp16le'
                if 'gbr' in input_pix_fmt:
                    ordering = 'gbr'
            # Compute per-channel minmax for normalization
            if value_range is None:
                value_range = np.stack([
                    [np.nanmin(array[..., c]), np.nanmax(array[..., c])] for c in range(array.shape[-1])
                ], axis=0)
            else:
                value_range = np.array(value_range)
                # If shape is (2,), broadcast to (channels, 2)
                if value_range.ndim == 1 and value_range.shape[0] == 2:
                    value_range = np.tile(value_range, (array.shape[-1], 1))
                elif value_range.shape == (array.shape[-1],):
                    value_range = np.stack([value_range, value_range+1], axis=1)
            # Normalization
            is_int_but_does_not_fit = (array.dtype.itemsize * 8 > bits) and np.nanmax(array) > (2**bits-1)
            normalized = is_float(array) or is_int_but_does_not_fit
            if normalized:
                array = normalize(array, minmax=value_range, bits=bits)
            # For ffv1+gbrp16le, swap RGB to ordering after normalization
            if vcodec == 'ffv1' and array.shape[-1] == 3:
                from videoparquet.utils import reorder_coords_axis
                array = reorder_coords_axis(array, list('rgb'), list(ordering), axis=-1)
                # Also reorder value_range (minmax) to match channel order
                channel_idx = [list('rgb').index(c) for c in list(ordering)]
                value_range = value_range[channel_idx]
            # Store ordering in metadata
            metadata = {
                'shape': list(map(int, array.shape)),
                'minmax': value_range.tolist() if value_range is not None else None,
                'columns': list(map(str, columns)) if columns is not None else None,
                'name': str(name),
                'BITS': int(bits),
                'CHANNELS': int(array.shape[-1]),
                'FRAMES': int(array.shape[0]),
                'REQ_PIX_FMT': input_pix_fmt,
                'OUT_PIX_FMT': input_pix_fmt,
                'PLANAR': True if input_pix_fmt.startswith('gbrp') else False,
                'CODEC': vcodec,
                'ext': '.mkv',
                'CHANNEL_ORDER': ordering,
            }
            # Set video_path for all codecs
            ext = '.mkv'
            video_path = output_path / array_id / f'{name}{ext}'
            t0 = time.time()
            actual_pix_fmt = _ffmpeg_write(str(video_path), array, array.shape[2], array.shape[1], params, planar_in=(input_pix_fmt=='gbrp'), input_pix_fmt=input_pix_fmt, loglevel=loglevel, metadata=metadata)
            t1 = time.time()
            metadata['ACTUAL_PIX_FMT'] = actual_pix_fmt
            meta_path = output_path / array_id / f'{name}.json'
            save_metadata(metadata, meta_path)
            # Stats
            original_size = array.size * array.itemsize / 2**20
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
            # Save reconstructed array as .npy for direct roundtrip test
            np.save(output_path / array_id / f'reconstructed_{name}.npy', array)
            print_fn(f"Wrote video for '{name}': {video_path}, shape={array.shape}, dtype={array.dtype}, normalized={normalized}, PCA={use_pca}")
            if compute_stats:
                print_fn(f"Stats for '{name}': original_size={original_size:.2f}MB, compressed_size={compressed_size:.2f}MB, compression={compression:.2f}, bpppb={bpppb:.4f}, write_time={t1-t0:.2f}s")
        except Exception as e:
            print_fn(f"Exception processing array_id='{array_id}' name='{name}': {e}")
            if exceptions == 'raise':
                raise e
    # Optionally save the DataFrame
    if save_dataset:
        df.to_parquet(output_path / array_id / 'data.parquet')
    return results 