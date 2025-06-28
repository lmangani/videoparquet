"""
Utility functions for videoparquet (normalization, metadata handling, etc.)
"""

import numpy as np
from sklearn.decomposition import PCA

def normalize(array, minmax, bits=8):
    """
    Normalize array to [0, 2**bits-1] using minmax (min, max).
    """
    min_val, max_val = minmax
    scaled = (array - min_val) / (max_val - min_val)
    scaled = np.clip(scaled, 0, 1)
    return (scaled * (2**bits - 1)).astype(np.uint16 if bits > 8 else np.uint8)

def denormalize(array, minmax, bits=8):
    """
    Denormalize array from [0, 2**bits-1] back to original range using minmax.
    """
    min_val, max_val = minmax
    scaled = array.astype(np.float32) / (2**bits - 1)
    return scaled * (max_val - min_val) + min_val

def is_float(array):
    """
    Check if array dtype is a float type.
    """
    return np.issubdtype(array.dtype, np.floating)

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