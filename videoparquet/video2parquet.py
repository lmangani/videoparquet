import pandas as pd
import numpy as np
from pathlib import Path
from .utils import denormalize, DRWrapper, reorder_coords_axis
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
        minmax = np.array(metadata['minmax'])
        columns = metadata['columns']
        num_frames, height, width, channels = shape
        vcodec = metadata.get('CODEC', 'ffv1')
        input_pix_fmt = metadata.get('OUT_PIX_FMT', 'gbrp')
        # For ffv1+gbrp16le, reorder minmax from ordering back to RGB before denormalization
        if vcodec == 'ffv1' and input_pix_fmt == 'gbrp16le':
            ordering = metadata.get('CHANNEL_ORDER', 'gbr')
            channel_idx = [list(ordering).index(c) for c in list('rgb')]
            minmax = minmax[channel_idx]
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
        # Denormalize for ffv1+gbrp16le as float32
        if vcodec == 'ffv1' and input_pix_fmt == 'gbrp16le':
            array = denormalize(array, minmax, bits=16).astype(np.float32)
            # Swap from ordering back to RGB
            ordering = metadata.get('CHANNEL_ORDER', 'gbr')
            array = reorder_coords_axis(array, list(ordering), list('rgb'), axis=-1)
        elif metadata.get('normalized', False):
            array = denormalize(array, minmax)
        # Save reconstructed array as .npy for direct roundtrip test
        np.save(Path(input_path) / array_id / f'reconstructed_{name}.npy', array)
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
        if columns is not None:
            if flat.shape[1] < len(columns):
                # Pad with zeros if needed
                pad_width = len(columns) - flat.shape[1]
                flat = np.pad(flat, ((0, 0), (0, pad_width)), mode='constant')
            df_recon = pd.DataFrame(flat, columns=columns)
            out_path = Path(input_path) / array_id / f'reconstructed_{name}.parquet'
            df_recon.to_parquet(out_path)
            return out_path
        # For direct array roundtrip, just save .npy and return
        return
    except Exception as e:
        print(f'Exception in video2parquet: {e}')
        if exceptions == 'raise':
            raise e 