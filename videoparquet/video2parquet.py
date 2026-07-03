"""
Convert video files back to Parquet format using native PyAV decoding.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from .utils import denormalize, DRWrapper, reorder_coords_axis
from .metadata import load_metadata
from .av_wrappers import read_video


def video2parquet(input_path, array_id, name='test', exceptions='raise'):
    """
    Reconstruct a Parquet file from video files and associated metadata.

    Parameters
    ----------
    input_path : str or Path
        Directory containing the video and metadata files.
    array_id : str
        Identifier for the dataset (subdirectory name).
    name : str
        Name of the array/video to reconstruct.
    exceptions : str
        'raise' to raise exceptions, 'ignore' to suppress.

    Returns
    -------
    Path or None
        Path to the reconstructed Parquet file, or None if no columns defined.
    """
    try:
        meta_path = Path(input_path) / array_id / f'{name}.json'
        metadata = load_metadata(meta_path)

        ext = metadata.get('ext', '.mkv')
        video_path = Path(input_path) / array_id / f'{name}{ext}'

        shape = metadata['shape']
        minmax = np.array(metadata['minmax'])
        columns = metadata.get('columns')
        num_frames, height, width, channels = shape

        vcodec = metadata.get('CODEC', 'ffv1')
        pix_fmt = metadata.get('OUT_PIX_FMT', 'gbrp16le')

        # Reorder minmax from storage order back to RGB
        if vcodec == 'ffv1' and pix_fmt == 'gbrp16le':
            ordering = metadata.get('CHANNEL_ORDER', 'gbr')
            channel_idx = [list(ordering).index(c) for c in 'rgb']
            minmax = minmax[channel_idx]

        # Read video
        array, _ = read_video(str(video_path))

        # Crop to expected shape
        array = array[:shape[0], :shape[1], :shape[2], :shape[3]]

        # Denormalize
        if vcodec == 'ffv1' and pix_fmt == 'gbrp16le':
            array = denormalize(array, minmax, bits=16).astype(np.float32)
            ordering = metadata.get('CHANNEL_ORDER', 'gbr')
            array = reorder_coords_axis(array, list(ordering), list('rgb'), axis=-1)
        elif metadata.get('normalized', False):
            array = denormalize(array, minmax)

        # Inverse PCA
        pca_params = metadata.get('pca_params')
        if pca_params not in [None, 'None']:
            DR = DRWrapper(params=pca_params)
            n_components = DR.dr.n_components
            if array.shape[-1] > n_components:
                array = array[..., :n_components]
            array = DR.inverse_transform(array)

        # Reshape to DataFrame
        array = array[:num_frames, :height, :width, :channels]
        flat = array.reshape(num_frames, -1)

        if columns is not None:
            # Pad if needed due to rounding
            if flat.shape[1] < len(columns):
                pad_width = len(columns) - flat.shape[1]
                flat = np.pad(flat, ((0, 0), (0, pad_width)), mode='constant')

            df_recon = pd.DataFrame(flat, columns=columns)
            out_path = Path(input_path) / array_id / f'reconstructed_{name}.parquet'
            df_recon.to_parquet(out_path)
            return out_path

        return None

    except Exception as e:
        print(f'Exception in video2parquet: {e}')
        if exceptions == 'raise':
            raise
        return None
