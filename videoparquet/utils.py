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