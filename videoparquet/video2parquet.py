import pandas as pd
import numpy as np
from pathlib import Path
from .utils import denormalize, DRWrapper
from .ffmpeg_wrappers import _ffmpeg_read
from .metadata import load_metadata
import yaml

def video2parquet(input_path, array_id, name='test', exceptions='raise'):
    '''
    Reconstructs a Parquet file from video files and associated metadata.
    Loads shape, minmax, columns, etc. from metadata JSON.
    '''
    try:
        meta_path = Path(input_path) / array_id / f'{name}.json'
        metadata = load_metadata(meta_path)
        ext = metadata.get('ext', '.mp4')
        video_path = Path(input_path) / array_id / f'{name}{ext}'
        shape = metadata['shape']
        minmax = metadata['minmax']
        columns = metadata['columns']
        num_frames, height, width, channels = shape
        # Read with actual pixel format from metadata
        vcodec = metadata.get('CODEC', 'ffv1')
        if vcodec == 'ffv1':
            input_pix_fmt = metadata.get('OUT_PIX_FMT', 'gbrp')
        else:
            input_pix_fmt = 'rgb24'
        loglevel = 'quiet'
        array, meta_info = _ffmpeg_read(str(video_path), loglevel=loglevel)
        orig_shape = meta_info.get('shape', None)
        if orig_shape is not None:
            array = array[:orig_shape[0], :orig_shape[1], :orig_shape[2], :orig_shape[3]]
        expected_size = np.prod(orig_shape) if orig_shape is not None else array.size
        if array.size != expected_size:
            print(f"DEBUG: array.shape={array.shape}, expected={orig_shape}")
            print(f"DEBUG: array.dtype={array.dtype}, meta_info={meta_info}")
            print(f"DEBUG: metadata={metadata}")
            raise ValueError(f"Read buffer size {array.size} does not match expected shape {orig_shape}.")
        if metadata.get('normalized', False):
            array = denormalize(array, minmax)
        # PCA inverse
        pca_params = metadata.get('pca_params', None)
        if pca_params not in [None, 'None']:
            DR = DRWrapper(params=pca_params)
            n_components = DR.dr.n_components
            if array.shape[-1] > n_components:
                array = array[..., :n_components]
            array = DR.inverse_transform(array)
        # Always crop to (num_frames, height, width, channels)
        array = array[:num_frames, :height, :width, :channels]
        flat = array.reshape(num_frames, -1)
        # Robustly handle column mismatch due to padding
        if flat.shape[1] < len(columns):
            # Pad with zeros
            pad = np.zeros((num_frames, len(columns) - flat.shape[1]), dtype=flat.dtype)
            flat = np.concatenate([flat, pad], axis=1)
            print(f"WARNING: Padding reconstructed DataFrame from {flat.shape[1]} to {len(columns)} columns.")
        elif flat.shape[1] > len(columns):
            # Truncate
            flat = flat[:, :len(columns)]
            print(f"WARNING: Truncating reconstructed DataFrame from {flat.shape[1]} to {len(columns)} columns.")
        df = pd.DataFrame(flat, columns=columns)
        parquet_out = Path(input_path) / array_id / f'reconstructed_{name}.parquet'
        df.to_parquet(parquet_out)
        return parquet_out
    except Exception as e:
        print(f'Exception in video2parquet: {e}')
        if exceptions == 'raise':
            raise e 