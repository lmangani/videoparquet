<img src="https://github.com/user-attachments/assets/eca17ccf-4e9f-425d-b6a7-fbca8b47da94" width=150 />

# videoparquet

Inspired by [xarrayvideo](https://github.com/IPL-UV/xarrayvideo), **videoparquet** is a Python library for converting Parquet files (containing array-like or tabular data) to video files and back, using ffmpeg and advanced data handling techniques.

> THIS IS JUST A FUN EXPERIMENT! DO NOT TAKE IT TOO SERIOUSLY ⚠️

## Features
- Convert Parquet files to video (mp4/mkv) using ffmpeg codecs (lossy or lossless)
- Store and recover all necessary metadata for roundtrip conversion
- Support for normalization, denormalization, and PCA (dimensionality reduction)
- Multi-array and multi-video support per Parquet file
- Flexible codec and bit-depth selection
- Automated recipe generation for batch processing
- **Strict, xarrayvideo-style test suite for lossless and lossy roundtrip**

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

### Advanced: Using Lossless Codecs
```python
conversion_rules = {
    # Use lossless codec (ffv1, 3-channel RGB only)
    'arr_lossless': (list(df.columns), arr.shape, 0, {'c:v': 'ffv1'}, 16, [arr.min(), arr.max()])
}
parquet2video('data.parquet', 'exampleid', conversion_rules, output_path='.')
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
Run the test suite to verify strict roundtrip and lossy scenarios:
```bash
pytest tests/test_roundtrip.py
```

### What is tested?
- **Lossless roundtrip:** Only 3-channel ffv1+gbrp16le is supported and tested. Max error is <0.001 per channel.
- **Lossy roundtrip:** A test with libx264 (rgb24) is included for comparison. Max error is typically 2–3 per channel (on a 0–95 range).

Example output:
```
Max abs error per channel: [0.00068665 0.00068665 0.00068665]  # ffv1+gbrp16le (lossless)
[libx264] Max abs error per channel: [2.588234 2.588234 2.588234]  # libx264 (lossy)
```

## Motivation
This project enables efficient storage, compression, and sharing of large datasets by leveraging video codecs, while maintaining the ability to recover the original data using Parquet as the canonical format.

## Codec and Pixel Format Restrictions

**Important:** For robust, lossless roundtrip, only the `ffv1` codec with the `gbrp16le` pixel format (planar RGB, 3 channels, no padding) is supported and tested. If your ffmpeg build does not support this combination, the library will raise a clear error. This ensures that timeseries/tabular data can be reliably converted to and from video without data loss or row padding issues.

Other codecs (e.g., `libx264`) may be used for lossy compression, but roundtrip is not guaranteed.

## 🚀 Benchmark Highlights
- Achieve up to **25x compression** over Parquet for timeseries/tabular data using video codecs (ffv1/gbrp16le)
- **Fast encoding/decoding**: Video roundtrip in under a second for typical scientific arrays
- **Lossless roundtrip** supported (with ffv1/gbrp16le and compatible ffmpeg)

## Benchmarking

See [BENCHMARK.md](BENCHMARK.md) for a summary of benchmark results comparing Parquet and video-based storage for timeseries/tabular data. This includes size, compression ratio, and performance metrics for scientific reproducibility.

## ⚠️ ffmpeg, ffv1, and Pixel Format Limitations

**IMPORTANT:**

- For true lossless roundtrip and compression, `videoparquet` requires ffmpeg to encode `ffv1` videos with the `gbrp16le` (planar RGB) pixel format.
- On macOS (Homebrew) and many Linux builds, ffmpeg may encode `ffv1` as `bgr0` instead of `gbrp16le`, even though `gbrp16le` is listed as supported. This is a known limitation/quirk of many ffmpeg builds.
- `bgr0` is not true planar RGB and may have padding/alpha issues. It is **not guaranteed to be robust for scientific roundtrip**.
- The test suite will **skip strict roundtrip and compression tests** if `gbrp16le` is not available, and will warn the user. Only platforms with `ffv1/gbrp16le` will run and require these tests to pass.
- To check your ffmpeg's pixel format support for ffv1, run:

  ```sh
  ffmpeg -h encoder=ffv1 | grep gbrp
  ```

- For scientific reproducibility, use a Docker image or reference ffmpeg build known to support `ffv1/gbrp16le`.

### How to Check Your ffmpeg

Run this command:

```sh
ffmpeg -f lavfi -i testsrc2=duration=1:size=2x2:rate=1 -pix_fmt gbrp16le -c:v ffv1 -y test_ffv1_gbrp16le.mkv && ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt -of default=noprint_wrappers=1:nokey=1 test_ffv1_gbrp16le.mkv
```

- If the output is `gbrp16le`, your ffmpeg is suitable for scientific roundtrip.
- If the output is `bgr0`, your ffmpeg will not guarantee true lossless roundtrip.

### For Scientific Reproducibility & CI

- Use a reference ffmpeg build (e.g., static Linux build from https://johnvansickle.com/ffmpeg/) or a Docker container with a known-good ffmpeg.
- The test suite is designed to run in CI (GitHub Actions) as long as the correct ffmpeg build is available.
- See the code and error messages for more details. 
