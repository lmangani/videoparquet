"""
Utility functions for videoparquet (normalization, metadata handling, etc.)
"""

import numpy as np
from sklearn.decomposition import PCA

def normalize(array, minmax, bits=8):
    """
    If array is not uint8, clip array to `minmax` and rescale to [0, 2**bits-1].
    For bits=8, uint8 is used as output, for bits 9-16, uint16 is used.
    minmax must have shape (B,2), and array must have shape (...,B)
    """
    if array.dtype == np.uint8:
        return array
    assert bits >=8 and bits <=16, 'Only 8 to 16 bits supported'
    max_value = 2**bits - 1
    array_bands = []
    for c in range(array.shape[-1]):
        array_c = array[..., c].astype(np.float32)
        array_c = (array_c - minmax[c, 0]) / (minmax[c, 1] - minmax[c, 0]) * max_value
        array_c[array_c > max_value] = max_value
        array_c[array_c < 0] = 0
        array_c[np.isnan(array_c)] = 0
        array_bands.append(array_c)
    arr = np.round(np.stack(array_bands, axis=-1))
    return arr.astype(np.uint8 if bits == 8 else np.uint16)

def denormalize(array, minmax, bits=8):
    """
    Transform to float32, and undo the scaling done in `normalize`
    minmax must have shape (B,2), and array must have shape (...,B)
    """
    max_value = 2**bits - 1
    array_bands = []
    for c in range(array.shape[-1]):
        array_c = array[..., c].astype(np.float32)
        array_c = array_c / max_value * (minmax[c, 1] - minmax[c, 0]) + minmax[c, 0]
        array_bands.append(array_c)
    return np.stack(array_bands, axis=-1)

def is_float(array):
    """
    Check if array dtype is a float type.
    """
    return np.issubdtype(array.dtype, np.floating)

def reorder_coords_axis(array, coords_in, coords_out, axis=-1):
    """
    Permute the dimensions within a single axis of an array from coords_in into coords_out.
    E.g.: axis=-1, coords_in=('r','g','b'), coords_out=('g','b','r')
    """
    if coords_in == coords_out:
        return array
    new_order = [coords_in.index(i) for i in coords_out]
    # Move reorder axis to position 0, reorder, and then move it back
    array_swapped = np.swapaxes(array, axis, 0)[new_order]
    return np.swapaxes(array_swapped, 0, axis)

def infer_video_shape(df_or_array, target_frames=None):
    """
    Infer a video-compatible shape from a DataFrame or array.

    Parameters
    ----------
    df_or_array : DataFrame, ndarray, or int
        Input data or total element count.
    target_frames : int, optional
        Target number of frames. If None, auto-detect.

    Returns
    -------
    tuple
        (frames, height, width, 3) shape.

    Raises
    ------
    ValueError
        If no valid video shape can be found.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.random.rand(100, 12288))  # 100 frames, 64x64x3
    >>> infer_video_shape(df)
    (100, 64, 64, 3)

    >>> df = pd.DataFrame(np.random.rand(4096, 3))  # 4096 pixels, 3 channels
    >>> infer_video_shape(df)
    (1, 64, 64, 3)
    """
    import pandas as pd

    if isinstance(df_or_array, pd.DataFrame):
        n_rows, n_cols = df_or_array.shape
        n_elements = n_rows * n_cols
    elif isinstance(df_or_array, np.ndarray):
        n_elements = df_or_array.size
        n_cols = df_or_array.shape[-1] if df_or_array.ndim > 1 else 1
        n_rows = n_elements // n_cols
    else:
        n_elements = int(df_or_array)
        n_cols = None
        n_rows = None

    # Detect layout from column count
    # If n_cols == 3: each row is a pixel -> (n_rows, H, W, 3) where n_rows = frames*H*W
    # If n_cols == H*W*3: each row is a frame -> (n_rows, H, W, 3)

    if n_cols == 3:
        # Row-per-pixel layout: need to factor n_rows into frames*H*W
        shape, padding = auto_detect_shape(n_rows * 3, n_columns=3)
    elif n_cols is not None and n_cols % 3 == 0:
        # Row-per-frame layout: n_cols should be H*W*3
        pixels_per_frame = n_cols // 3
        # Find H, W
        h = w = int(pixels_per_frame ** 0.5)
        if h * w != pixels_per_frame:
            # Not square, find factors
            for h in range(int(pixels_per_frame ** 0.5), 0, -1):
                if pixels_per_frame % h == 0:
                    w = pixels_per_frame // h
                    break
        shape = (n_rows, h, w, 3)
        padding = 0
    else:
        shape, padding = auto_detect_shape(n_elements)

    if shape is None:
        raise ValueError(f"Cannot find valid video shape for {n_elements} elements")

    return shape


def auto_detect_shape(n_elements, n_columns=None, prefer_square=True):
    """
    Automatically detect a valid (frames, height, width, 3) shape for video encoding.

    Parameters
    ----------
    n_elements : int
        Total number of elements in the array.
    n_columns : int, optional
        Number of columns in the original DataFrame (helps determine layout).
    prefer_square : bool
        If True, prefer square-ish height/width dimensions.

    Returns
    -------
    tuple
        (frames, height, width, channels) shape, or None if no valid shape found.

    Notes
    -----
    Video encoding requires 3 channels (RGB). The function tries to find
    dimensions that work well with video codecs (multiples of 2 or 16).
    """
    channels = 3

    # Total pixels needed
    if n_elements % channels != 0:
        # Can't evenly divide into 3 channels
        # Try padding
        n_elements_padded = ((n_elements // channels) + 1) * channels
        padding_needed = n_elements_padded - n_elements
    else:
        n_elements_padded = n_elements
        padding_needed = 0

    total_pixels = n_elements_padded // channels

    # If n_columns == 3, each row is a pixel (frames*H*W rows, 3 cols)
    # If n_columns == H*W*3, each row is a frame (frames rows, H*W*3 cols)

    # Find factors
    def find_factors(n):
        factors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                factors.append((i, n // i))
        return factors

    # Try to find good video dimensions
    best_shape = None
    best_score = float('inf')

    # Try different frame counts
    for frames in range(1, min(total_pixels, 1000) + 1):
        if total_pixels % frames != 0:
            continue

        pixels_per_frame = total_pixels // frames

        # Find height x width factors
        for h, w in find_factors(pixels_per_frame):
            # Prefer dimensions that are multiples of 16 (codec-friendly)
            # and not too extreme in aspect ratio
            aspect_ratio = max(h, w) / max(min(h, w), 1)

            # Score: lower is better
            # Penalize extreme aspect ratios
            # Reward multiples of 16
            # Reward square-ish if preferred
            score = aspect_ratio
            if h % 16 == 0:
                score -= 0.5
            if w % 16 == 0:
                score -= 0.5
            if prefer_square and abs(h - w) < max(h, w) * 0.5:
                score -= 1

            # Minimum dimension should be at least 4
            if min(h, w) < 4:
                continue

            if score < best_score:
                best_score = score
                best_shape = (frames, h, w, channels)

    return best_shape, padding_needed


class DRWrapper:
    def __init__(self, n_components=None, params=None):
        if params is not None:
            import json
            self.dr = PCA()
            d = json.loads(params)
            # Convert lists back to numpy arrays for PCA attributes
            for k, v in d.items():
                if isinstance(v, list):
                    d[k] = np.array(v)
            self.dr.__dict__.update(d)
        else:
            self.dr = PCA(n_components=n_components)

    def fit_transform(self, array):
        orig_shape = array.shape
        flat = array.reshape(-1, orig_shape[-1])
        reduced = self.dr.fit_transform(flat)
        new_shape = orig_shape[:-1] + (reduced.shape[-1],)
        return reduced.reshape(new_shape)

    def inverse_transform(self, array):
        orig_shape = array.shape
        flat = array.reshape(-1, orig_shape[-1])
        restored = self.dr.inverse_transform(flat)
        new_shape = orig_shape[:-1] + (restored.shape[-1],)
        return restored.reshape(new_shape)

    def get_params_str(self):
        import json
        d = {}
        for k, v in self.dr.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, (np.generic,)):
                d[k] = v.item()
            else:
                d[k] = v
        return json.dumps(d) 