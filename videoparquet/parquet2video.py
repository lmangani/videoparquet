"""
Convert Parquet files to video format using native PyAV encoding.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
from .utils import normalize, is_float, DRWrapper, reorder_coords_axis, infer_video_shape
from .av_wrappers import write_video


def parquet2video(parquet_path, array_id, conversion_rules, compute_stats=False,
                  output_path='./', loglevel='quiet', exceptions='raise',
                  verbose=True, nan_fill=None, save_dataset=True, arrays=None):
    """
    Convert a Parquet file (containing array-like or tabular data) into video files.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the input Parquet file (or None if using arrays parameter).
    array_id : str
        Identifier for the dataset, used as subdirectory name.
    conversion_rules : dict
        Dict specifying which columns/arrays to convert and how.
        Format: {name: (columns, shape, n_components, params, bits, value_range)}

        params dict options:
        - 'c:v': codec ('ffv1' for lossless, 'libx264'/'libx265'/'libvpx-vp9' for lossy)
        - 'format': container format ('mkv', 'mp4', 'webm'). Default: 'mkv'
        - 'crf': quality for lossy codecs (0-51, lower=better). Default: 23
        - 'preset': encoding speed ('ultrafast' to 'veryslow'). Default: 'medium'
        - 'qr_metadata': if True, prepend QR code frame (survives re-encoding)
    compute_stats : bool
        Whether to compute and print compression statistics.
    output_path : str or Path
        Directory where output will be written.
    loglevel : str
        Logging level for video encoding.
    exceptions : str
        'raise' to raise exceptions, 'ignore' to suppress.
    verbose : bool
        Whether to print progress messages.
    nan_fill : int or str
        How to fill NaN values: int value, 'mean', 'min', or 'max'.
    save_dataset : bool
        Whether to save a copy of the DataFrame.
    arrays : dict
        Optional dict of numpy arrays to use directly instead of reading from Parquet.

    Returns
    -------
    dict
        Results dict with paths and compression statistics for each converted array.
    """
    print_fn = print if verbose else (lambda *a, **k: None)
    results = {}

    if parquet_path is not None:
        df = pd.read_parquet(parquet_path)

    output_path = Path(output_path)
    (output_path / array_id).mkdir(exist_ok=True, parents=True)

    for name, config in conversion_rules.items():
        # Unpack config with defaults
        bits = 8
        n_components = 'all'
        value_range = None
        params = {'c:v': 'ffv1'}

        if len(config) == 6:
            columns, shape, n_components, params, bits, value_range = config
        elif len(config) == 5:
            columns, shape, n_components, params, bits = config
        elif len(config) == 4:
            columns, shape, n_components, params = config
        elif len(config) == 3:
            columns, shape, n_components = config
        else:
            raise ValueError(f'Invalid config: {config}. Expected: (columns, shape, [n_components], [params], [bits], [value_range])')

        try:
            # Extract data as numpy array
            if columns is None and arrays is not None:
                array = arrays[name]
            else:
                if isinstance(columns, str):
                    columns = [columns]
                array = df[columns].values

                # Auto-detect shape if not provided or 'auto'
                if shape is None or shape == 'auto':
                    shape = infer_video_shape(array)
                    print_fn(f"Auto-detected shape: {shape}")

                array = array.reshape(shape)

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
                    raise ValueError(f'Unknown nan_fill: {nan_fill}')

            # PCA dimensionality reduction
            use_pca = isinstance(n_components, int) and n_components > 0
            pca_params = None
            if use_pca:
                DR = DRWrapper(n_components=n_components)
                array = DR.fit_transform(array)
                pca_params = DR.get_params_str()

            # Determine codec and pixel format
            vcodec = params.get('c:v', 'ffv1')
            ordering = 'rgb'
            input_pix_fmt = 'rgb24'

            # For ffv1 with 3 channels, use 16-bit planar RGB for lossless
            if vcodec == 'ffv1' and array.shape[-1] == 3:
                if array.dtype != np.float32:
                    array = array.astype(np.float32)
                bits = 16
                input_pix_fmt = 'gbrp16le'
                ordering = 'gbr'

            # Compute per-channel min/max for normalization
            if value_range is None:
                value_range = np.stack([
                    [np.nanmin(array[..., c]), np.nanmax(array[..., c])]
                    for c in range(array.shape[-1])
                ], axis=0)
            else:
                value_range = np.array(value_range)
                if value_range.ndim == 1 and value_range.shape[0] == 2:
                    value_range = np.tile(value_range, (array.shape[-1], 1))
                elif value_range.shape == (array.shape[-1],):
                    value_range = np.stack([value_range, value_range + 1], axis=1)

            # Normalize if needed
            is_int_but_does_not_fit = (array.dtype.itemsize * 8 > bits) and np.nanmax(array) > (2**bits - 1)
            normalized = is_float(array) or is_int_but_does_not_fit
            if normalized:
                array = normalize(array, minmax=value_range, bits=bits)

            # Reorder channels for planar format
            if vcodec == 'ffv1' and array.shape[-1] == 3:
                array = reorder_coords_axis(array, list('rgb'), list(ordering), axis=-1)
                channel_idx = [list('rgb').index(c) for c in list(ordering)]
                value_range = value_range[channel_idx]

            # Determine container format
            container_format = params.get('format', 'mkv')
            ext_map = {'mkv': '.mkv', 'mp4': '.mp4', 'webm': '.webm', 'avi': '.avi'}
            ext = ext_map.get(container_format, '.mkv')

            # Build metadata
            metadata = {
                'shape': list(map(int, array.shape)),
                'minmax': value_range.tolist(),
                'columns': list(map(str, columns)) if columns is not None else None,
                'name': str(name),
                'BITS': int(bits),
                'CHANNELS': int(array.shape[-1]),
                'FRAMES': int(array.shape[0]),
                'REQ_PIX_FMT': input_pix_fmt,
                'OUT_PIX_FMT': input_pix_fmt,
                'PLANAR': input_pix_fmt.startswith('gbrp'),
                'CODEC': vcodec,
                'ext': ext,
                'CHANNEL_ORDER': ordering,
                'pca_params': pca_params,
            }

            # Write video (metadata is embedded in the container)
            video_path = output_path / array_id / f'{name}{ext}'
            qr_metadata = params.get('qr_metadata', False)
            t0 = time.time()
            actual_pix_fmt = write_video(
                str(video_path), array,
                width=array.shape[2], height=array.shape[1],
                codec=vcodec, params=params,
                pix_fmt=input_pix_fmt, metadata=metadata,
                qr_metadata=qr_metadata
            )
            t1 = time.time()

            # Compute stats
            original_size = array.size * array.itemsize / 2**20
            compressed_size = os.stat(video_path).st_size / 2**20
            compression = compressed_size / original_size if original_size > 0 else None
            bpppb = compressed_size * 2**20 * 8 / array.size if array.size > 0 else None

            results[name] = {
                'path': str(video_path),
                'original_size_MB': original_size,
                'compressed_size_MB': compressed_size,
                'compression_ratio': compression,
                'bpppb': bpppb,
                'write_time_s': t1 - t0
            }

            print_fn(f"Wrote video '{name}': {video_path}, shape={array.shape}, normalized={normalized}, PCA={use_pca}")
            if compute_stats:
                print_fn(f"  Stats: {original_size:.2f}MB -> {compressed_size:.2f}MB, ratio={compression:.2f}, time={t1-t0:.2f}s")

        except Exception as e:
            print_fn(f"Exception processing '{name}': {e}")
            if exceptions == 'raise':
                raise

    # Save DataFrame copy
    if save_dataset and parquet_path is not None:
        df.to_parquet(output_path / array_id / 'data.parquet')

    return results
