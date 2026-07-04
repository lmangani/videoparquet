<img src="https://github.com/user-attachments/assets/eca17ccf-4e9f-425d-b6a7-fbca8b47da94" width=150 />

# videoparquet

**Back up your Parquet files to YouTube. Up to 7x compression.** 🎬📊

Convert Parquet → Video → Upload to any video site → Download → Video → Parquet. Your data survives!

```bash
pip install videoparquet
```

> ⚠️ **This is a novelty project!** A fun experiment in using video platforms as data storage. Not for production use. Inspired by [xarrayvideo](https://github.com/IPL-UV/xarrayvideo)[^1].

> **No external dependencies.** Uses [PyAV](https://github.com/PyAV-Org/PyAV) with bundled FFmpeg libraries. No `ffmpeg` binary required.

## Why?

Because you can upload videos anywhere. YouTube, Vimeo, Google Drive, iCloud, that random video hosting site from 2008 that's somehow still running.

Your data becomes a video. Videos are forever. QED. 🎉

*(Also: video codecs are surprisingly good at compressing structured numerical data.)*

## Benchmark

| Data Type | Compression | Notes |
|-----------|-------------|-------|
| Smooth sensor grids | **up to 7x** | Low noise, high spatial correlation |
| Typical sensor data | **3-4x** | Environmental monitoring, thermal grids |
| Random/noisy data | **2-3x** | Still beats Parquet for floats |
| Integer/categorical | **0.5-1x** | Parquet wins here, don't use videoparquet |

All roundtrips are **lossless** (max error < 0.001).

### Best results with

- Continuous float data (not integers)
- Spatial grids (2D/3D sensor arrays)
- Temporal correlation between frames
- Low sensor noise

## The Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Parquet   │ ──▶ │    Video    │ ──▶ │   YouTube   │
│   (data)    │     │   (.mkv)    │     │  (backup!)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Download   │ ──▶ │   Parquet   │
                    │   (.mkv)    │     │ (restored!) │
                    └─────────────┘     └─────────────┘
```

## Quick Start

```python
from videoparquet import parquet2video, video2parquet, infer_video_shape
import pandas as pd
import numpy as np

# Create some data
arr = np.random.randn(16, 64, 64, 3).astype(np.float32)
df = pd.DataFrame(arr.reshape(16, -1))
df.to_parquet('data.parquet')

# Auto-detect the video shape from your data
print(infer_video_shape(df))  # → (16, 64, 64, 3)

# Define what to convert
conversion_rules = {
    'myarray': (
        list(df.columns),           # columns to use
        'auto',                     # auto-detect shape (or specify manually)
        0,                          # PCA components (0 = none)
        {'c:v': 'ffv1'},            # codec (ffv1 = lossless)
        16,                         # bit depth
        [arr.min(), arr.max()]      # value range for normalization
    )
}

# Convert Parquet → Video
parquet2video('data.parquet', 'dataset_id', conversion_rules, output_path='./output')
# Creates: ./output/dataset_id/myarray.mkv (single file, all metadata embedded!)

# Convert Video → Parquet (just pass the .mkv file directly)
video2parquet('./output/dataset_id/myarray.mkv')
# Creates: ./output/dataset_id/reconstructed_myarray.parquet
```

## Format Options

**Defaults**: MKV container + FFV1 codec (lossless, recommended for local/cloud storage)

```python
# Lossless (default)
params = {'c:v': 'ffv1'}

# Lossy H.264 for YouTube/sharing (smaller files)
params = {'c:v': 'libx264', 'format': 'mp4', 'crf': 18}

# Lossy H.265 (better compression, less compatible)
params = {'c:v': 'libx265', 'format': 'mp4', 'crf': 20}

# VP9 for WebM
params = {'c:v': 'libvpx-vp9', 'format': 'webm', 'crf': 20}
```

| Codec | Type | Container | Best For |
|-------|------|-----------|----------|
| `ffv1` | Lossless | MKV | Local storage, exact roundtrip |
| `libx264` | Lossy | MP4 | YouTube, sharing, preview |
| `libx265` | Lossy | MP4 | Smaller files, modern players |
| `libvpx-vp9` | Lossy | WebM | Web, open format |

> ⚠️ **YouTube note**: YouTube re-encodes all uploads. Use MP4/H.264 for best compatibility. Metadata is preserved in the file but may be stripped by some platforms.

## How It Works

1. **Reshape**: Tabular data → 4D array `(frames, height, width, channels)`
2. **Normalize**: Scale values to 16-bit range, track min/max per channel
3. **Encode**: Write as video using FFV1 codec (lossless, planar RGB)
4. **Decode**: Read video, denormalize using embedded metadata
5. **Reconstruct**: Reshape back to original DataFrame

**All-in-one file**: Metadata (shape, normalization params, column names) is embedded directly in the MKV container. No sidecar files needed - just upload/download the single `.mkv` file.

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
