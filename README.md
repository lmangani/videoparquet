<img src="https://github.com/user-attachments/assets/eca17ccf-4e9f-425d-b6a7-fbca8b47da94" width=150 />

# videoparquet

Inspired by xarrayvideo, **videoparquet** is a Python library for converting Parquet files (containing array-like or tabular data) to video files and back, using ffmpeg and advanced data handling techniques.

> THIS IS JUST A FUN EXPERIMENT! DO NOT TAKE IT TOO SERIOUSLY ⚠️

## Features
- Convert Parquet files to video (mp4/mkv) using ffmpeg codecs (lossy or lossless)
- Store and recover all necessary metadata for roundtrip conversion
- Support for normalization, denormalization, and PCA (dimensionality reduction)
- Multi-array and multi-video support per Parquet file
- Flexible codec and bit-depth selection
- Automated recipe generation for batch processing
- Test suite for roundtrip and lossy/lossless scenarios

## Installation

```bash
pip install -r requirements.txt
# or, for development:
# pip install -e .
```

## Usage

### Basic: Parquet to Video and Back
```python
from videoparquet.parquet2video import parquet2video
from videoparquet.video2parquet import video2parquet
import pandas as pd
import numpy as np

# Create synthetic data and save as Parquet
arr = np.random.rand(4, 4, 3)  # (frames, pixels, channels)
df = pd.DataFrame(arr.reshape(4, -1))
df.to_parquet('data.parquet')

# Define conversion rules (see below for details)
conversion_rules = {
    'arr1': (list(df.columns), arr.shape, 0, {'c:v': 'libx264'}, 8, [arr.min(), arr.max()])
}

# Parquet -> Video
parquet2video('data.parquet', 'exampleid', conversion_rules, output_path='.')

# Video -> Parquet
video2parquet('.', 'exampleid', name='arr1')
```

### Advanced: Using PCA and Lossless Codecs
```python
conversion_rules = {
    # Use PCA to reduce to 2 components (lossy)
    'arr_pca': (list(df.columns), arr.shape, 2, {'c:v': 'libx264'}, 8, [arr.min(), arr.max()]),
    # Use lossless codec (ffv1)
    'arr_lossless': (list(df.columns), arr.shape, 0, {'c:v': 'ffv1'}, 8, [arr.min(), arr.max()])
}
parquet2video('data.parquet', 'exampleid', conversion_rules, output_path='.')
video2parquet('.', 'exampleid', name='arr_pca')
video2parquet('.', 'exampleid', name='arr_lossless')
```

### Automated Recipe Generation
```python
from videoparquet.get_recipe import get_recipe
import pandas as pd
# df = pd.read_parquet('data.parquet')
recipe = get_recipe(df)  # Returns a dict of conversion rules
```

## Testing
Run the test suite to verify roundtrip and PCA scenarios:
```bash
pytest tests/test_roundtrip.py
```

## Motivation
This project enables efficient storage, compression, and sharing of large datasets by leveraging video codecs, while maintaining the ability to recover the original data using Parquet as the canonical format.

## Codec and Pixel Format Restrictions

**Important:** For robust, lossless roundtrip, only the `ffv1` codec with the `gbrp` pixel format (planar RGB, 3 channels, no padding) is supported. If your ffmpeg build does not support this combination, the library will raise a clear error. This ensures that timeseries/tabular data can be reliably converted to and from video without data loss or row padding issues.

Other codecs (e.g., `libx264`) may be used for lossy compression, but roundtrip is not guaranteed.

## Benchmarking

See [BENCHMARK.md](BENCHMARK.md) for a summary of benchmark results comparing Parquet and video-based storage for timeseries/tabular data. This includes size, compression ratio, and performance metrics for scientific reproducibility. 
