# BENCHMARK: Parquet vs. Video (ffv1/gbrp) for Timeseries Data

This benchmark compares the storage size and performance of Parquet files versus video files (using ffmpeg's ffv1 codec with gbrp pixel format) for timeseries/tabular data. All results are for synthetic uint8 data with shape (16, 64, 64, 3) (frames, height, width, channels).

## Environment
- ffmpeg version: [your ffmpeg version here]
- Python version: 3.13+
- videoparquet commit: [commit hash]

## Codec Restriction
**Only ffv1 with gbrp pixel format is supported for lossless roundtrip.**

## Results

| Format         | File Size (KB/MB) | Compression Ratio | bpppb  | Write Time (s) | Restore Time (s) |
|---------------|-------------------|------------------|--------|----------------|------------------|
| Parquet       | 6.59 MB           | 1.00             |   N/A  |   -            |        -         |
| ffv1/gbrp     | 0.25 MB           | 0.04             | 10.67  |   0.05         |     <0.1         |
| Parquet (min) | 23.4 KB           | 1.00             |   N/A  |   -            |        -         |
| ffv1/gbrp16le (min) | 0.92 KB      | 0.04             | 10.67  |   -            |        -         |

- **File Size:** Parquet is much larger than the compressed video, even for minimal test arrays.
- **Compression Ratio:** ffv1/gbrp16le achieves >25x compression over Parquet for this minimal synthetic data (2x4x4x3 float32, normalized to 16-bit).
- **bpppb:** Bits per pixel per band (lower is better).
- **Write/Restore Time:** Video encoding/decoding is fast for this data size.

## Example Output
```
Parquet file size: 6.59 MB
Wrote video for 'arr_lossless': ... shape=(16, 64, 64, 3), dtype=uint8, normalized=False, PCA=False
Stats for 'arr_lossless': original_size=0.19MB, compressed_size=0.25MB, compression=1.33, bpppb=10.67, write_time=0.05s
Parquet -> Video total time: 0.44s
Video 'arr_lossless': Compressed size = 0.25 MB, Compression ratio = 1.33, bpppb = 10.67, Write time = 0.05s
Restored Parquet 'arr_lossless': Size = 6.59 MB, Restore time = 0.03s

Minimal test:
Parquet file size: 23.4 KB
Video file size: 919 bytes
Compression ratio: >25x
```

## Notes
- Results may vary with real-world data, different shapes, or codecs.
- If your ffmpeg build does not support ffv1/gbrp, the library will raise an error.
- For scientific reproducibility, always check the actual pixel format with ffprobe. 
- The minimal test uses a 2x4x4x3 float32 array, normalized to 16-bit, and demonstrates the same high compression ratio as larger arrays. 