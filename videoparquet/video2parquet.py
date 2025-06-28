import pandas as pd
import numpy as np
from pathlib import Path
from .utils import denormalize, DRWrapper
from .ffmpeg_wrappers import _ffmpeg_read
from .metadata import load_metadata

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
        actual_pix_fmt = metadata.get('actual_pix_fmt', metadata.get('output_pix_fmt', 'rgb24'))
        # Read with actual pixel format
        if actual_pix_fmt == 'bgr0':
            array = _ffmpeg_read(str(video_path), width, height, num_frames, input_pix_fmt='bgr0')
            array = array[..., :3][..., ::-1]  # BGR to RGB, drop alpha
        else:
            array = _ffmpeg_read(str(video_path), width, height, num_frames, input_pix_fmt=actual_pix_fmt)
        if array.size != num_frames * height * width * channels:
            raise ValueError(f"Read buffer size {array.size} does not match expected shape {(num_frames, height, width, channels)}.\n"
                             f"Pixel format: {actual_pix_fmt}.\n"
                             f"Try a different codec or pixel format.")
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
        # Flatten to (num_frames, -1) and create DataFrame
        flat = array.reshape(num_frames, -1)
        df = pd.DataFrame(flat, columns=columns)
        parquet_out = Path(input_path) / array_id / f'reconstructed_{name}.parquet'
        df.to_parquet(parquet_out)
        return parquet_out
    except Exception as e:
        print(f'Exception in video2parquet: {e}')
        if exceptions == 'raise':
            raise e 