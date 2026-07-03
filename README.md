<img src="https://github.com/user-attachments/assets/eca17ccf-4e9f-425d-b6a7-fbca8b47da94" width=150 />

# videoparquet

**Store tabular data as video. Get 5-10x compression. Keep it lossless.**

Inspired by [xarrayvideo](https://github.com/IPL-UV/xarrayvideo) and its accompanying paper[^1], videoparquet converts Parquet files to video and back—leveraging video codecs' spatial/temporal compression for scientific and tabular data.

```bash
pip install videoparquet
```

> **No external dependencies.** Uses [PyAV](https://github.com/PyAV-Org/PyAV) with bundled FFmpeg libraries. No `ffmpeg` binary required.

## Why?

Video codecs like FFV1 are designed to compress sequences of images with spatial and temporal patterns. Many scientific datasets (sensor arrays, time series, simulations) have similar structure. This library exploits that.

| Data Type | Compression | Use Case |
|-----------|-------------|----------|
| **Random noise** | 4-6x | Baseline |
| **Structured/scientific** | 7-10x | Sensor data, simulations |
| **Highly correlated** | 10-25x | Satellite imagery, time series |

## Benchmark

```
Dataset          Parquet    Video     Ratio    Encode    Decode
─────────────────────────────────────────────────────────────────
Random 64³×3     9,991 KB   1,685 KB  5.9x     302 ms    512 ms
Structured 64³×3 9,991 KB   1,390 KB  7.2x     361 ms    508 ms
```

All roundtrips are **lossless** (max error < 0.001 per channel).

## Quick Start

```python
from videoparquet import parquet2video, video2parquet
import pandas as pd
import numpy as np

# Create some data
arr = np.random.randn(16, 64, 64, 3).astype(np.float32)
df = pd.DataFrame(arr.reshape(16, -1))
df.to_parquet('data.parquet')

# Define what to convert
conversion_rules = {
    'myarray': (
        list(df.columns),           # columns to use
        (16, 64, 64, 3),            # reshape to (frames, H, W, channels)
        0,                          # PCA components (0 = none)
        {'c:v': 'ffv1'},            # codec (ffv1 = lossless)
        16,                         # bit depth
        [arr.min(), arr.max()]      # value range for normalization
    )
}

# Convert Parquet → Video
parquet2video('data.parquet', 'dataset_id', conversion_rules, output_path='./output')
# Creates: ./output/dataset_id/myarray.mkv

# Convert Video → Parquet
video2parquet('./output', 'dataset_id', name='myarray')
# Creates: ./output/dataset_id/reconstructed_myarray.parquet
```

## Supported Codecs

| Codec | Type | Format | Best For |
|-------|------|--------|----------|
| `ffv1` | Lossless | `gbrp16le` (16-bit) | Scientific data, exact roundtrip |
| `libx264` | Lossy | `yuv420p` (8-bit) | Preview, smaller files |

## How It Works

1. **Reshape**: Tabular data → 4D array `(frames, height, width, channels)`
2. **Normalize**: Scale values to 16-bit range, track min/max per channel
3. **Encode**: Write as video using FFV1 codec (lossless, planar RGB)
4. **Decode**: Read video, denormalize using stored metadata
5. **Reconstruct**: Reshape back to original DataFrame

Metadata (shape, normalization params, column names) is stored in a sidecar JSON file.

## Installation

```bash
pip install videoparquet
```

**Requirements:**
- Python 3.8+
- NumPy, Pandas, PyArrow
- PyAV (bundled FFmpeg, no system install needed)
- scikit-learn (for optional PCA)

## API Reference

### `parquet2video()`

```python
parquet2video(
    parquet_path,       # Path to input Parquet file
    array_id,           # Identifier (becomes subdirectory name)
    conversion_rules,   # Dict of {name: (columns, shape, pca, params, bits, range)}
    output_path='./',   # Output directory
    compute_stats=False,# Print compression statistics
    verbose=True,       # Print progress
    nan_fill=None,      # Handle NaN: int, 'mean', 'min', 'max'
)
```

### `video2parquet()`

```python
video2parquet(
    input_path,         # Directory containing video files
    array_id,           # Dataset identifier
    name='test',        # Name of the array to reconstruct
)
```

## Testing

```bash
pytest tests/ -v
```

## Limitations

- Only 3-channel arrays supported (maps to RGB video planes)
- FFV1 lossless requires 16-bit planar format (`gbrp16le`)
- Data must be reshapeable to `(frames, height, width, 3)`

## License

MIT

## Citation

If you use this in research, please cite the xarrayvideo paper:

```bibtex
@article{pellicer2025video,
  title={Video compression for spatiotemporal Earth system data},
  author={Pellicer-Valero, Oscar J and Aybar, Cesar and Camps-Valls, Gustau},
  journal={arXiv preprint arXiv:2506.19656},
  year={2025}
}
```

[^1]: Pellicer-Valero, O. J., Aybar, C., & Camps-Valls, G. (2025). Video compression for spatiotemporal Earth system data. arXiv. https://doi.org/10.48550/arXiv.2506.19656
