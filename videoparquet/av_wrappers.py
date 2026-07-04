"""
Native video encoding/decoding using PyAV.
No external ffmpeg binary required - PyAV bundles libav libraries.
All metadata is embedded directly in the MKV container - no sidecar files.
Optional QR code metadata for platforms that strip container metadata.
"""

import av
import numpy as np
import json
import os
import base64
import zlib

# Marker to identify videoparquet files
VPARQUET_MARKER = 'VPARQUET_V2'

# QR code support (optional)
try:
    import qrcode
    import cv2
    HAS_QR = True
except ImportError:
    HAS_QR = False


def _create_qr_frame(metadata, frame_size):
    """
    Create a QR code frame containing compressed metadata.

    Parameters
    ----------
    metadata : dict
        Metadata to encode.
    frame_size : tuple
        (width, height) of the video frame.

    Returns
    -------
    np.ndarray
        RGB frame with QR code centered, or None if QR not available.
    """
    if not HAS_QR:
        return None

    encoded = _encode_metadata(metadata)

    # Create QR with high error correction
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=max(4, min(frame_size) // 50),
        border=4,
    )
    qr.add_data(f'{VPARQUET_MARKER}:{encoded}')
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_array = np.array(qr_img.convert('RGB'))

    # Create frame with QR centered
    w, h = frame_size
    frame = np.full((h, w, 3), 128, dtype=np.uint8)

    qh, qw = qr_array.shape[:2]
    if qw <= w and qh <= h:
        ox, oy = (w - qw) // 2, (h - qh) // 2
        frame[oy:oy+qh, ox:ox+qw] = qr_array

    return frame


def _decode_qr_frame(frame):
    """
    Decode metadata from a QR code frame.

    Parameters
    ----------
    frame : np.ndarray
        RGB frame that may contain a QR code.

    Returns
    -------
    dict or None
        Decoded metadata, or None if no valid QR found.
    """
    if not HAS_QR:
        return None

    try:
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(frame)

        if data and data.startswith(VPARQUET_MARKER + ':'):
            encoded = data[len(VPARQUET_MARKER) + 1:]
            return _decode_metadata(encoded)
    except Exception:
        pass

    return None


def _compress_columns(columns):
    """
    Compress column list to a compact representation.

    Detects patterns:
    - Numeric: [0, 1, 2, ...] -> {"_t": "range", "n": count}
    - Prefixed: ["col0", "col1", ...] -> {"_t": "prefix", "p": "col", "n": count}
    - Otherwise: store as-is
    """
    if columns is None:
        return None

    n = len(columns)
    if n == 0:
        return []

    # Check if all numeric (0, 1, 2, ...)
    try:
        if all(int(c) == i for i, c in enumerate(columns)):
            return {'_t': 'range', 'n': n}
    except (ValueError, TypeError):
        pass

    # Check for common prefix pattern
    first = str(columns[0])
    for prefix_len in range(len(first), 0, -1):
        prefix = first[:prefix_len]
        try:
            if all(str(c).startswith(prefix) and int(str(c)[prefix_len:]) == i
                   for i, c in enumerate(columns)):
                return {'_t': 'prefix', 'p': prefix, 'n': n}
        except (ValueError, TypeError):
            continue

    # No pattern found, store as-is (but only first 100 + count for very long lists)
    if n > 100:
        return {'_t': 'sample', 'cols': [str(c) for c in columns[:100]], 'n': n}

    return [str(c) for c in columns]


def _expand_columns(compressed):
    """Expand compressed column representation back to list."""
    if compressed is None:
        return None

    if isinstance(compressed, list):
        return compressed

    if isinstance(compressed, dict):
        t = compressed.get('_t')
        n = compressed.get('n', 0)

        if t == 'range':
            return [str(i) for i in range(n)]
        elif t == 'prefix':
            prefix = compressed.get('p', '')
            return [f"{prefix}{i}" for i in range(n)]
        elif t == 'sample':
            # Best effort: use sample if matches, else generate numeric
            return [str(i) for i in range(n)]

    return None


def _encode_metadata(metadata):
    """Compress and encode metadata for embedding in container."""
    # Optimize columns storage
    optimized = metadata.copy()
    if 'columns' in optimized:
        optimized['columns'] = _compress_columns(optimized['columns'])

    # Convert numpy arrays to lists
    for key, value in optimized.items():
        if isinstance(value, np.ndarray):
            optimized[key] = value.tolist()

    json_bytes = json.dumps(optimized, separators=(',', ':')).encode('utf-8')
    compressed = zlib.compress(json_bytes, level=9)
    encoded = base64.b64encode(compressed).decode('ascii')
    return encoded


def _decode_metadata(encoded):
    """Decode and decompress metadata from container."""
    compressed = base64.b64decode(encoded)
    json_bytes = zlib.decompress(compressed)
    metadata = json.loads(json_bytes.decode('utf-8'))

    # Expand columns
    if 'columns' in metadata:
        metadata['columns'] = _expand_columns(metadata['columns'])

    return metadata


def write_video(output_path, array, width, height, codec='ffv1', params=None,
                pix_fmt='gbrp16le', metadata=None, qr_metadata=False):
    """
    Write a numpy array (T, H, W, C) as a video file with embedded metadata.

    Parameters
    ----------
    output_path : str
        Path for the output video file.
    array : np.ndarray
        Video data with shape (frames, height, width, channels).
    width : int
        Frame width.
    height : int
        Frame height.
    codec : str
        Video codec ('ffv1' for lossless, 'libx264' for lossy).
    params : dict
        Additional codec parameters.
    pix_fmt : str
        Pixel format ('gbrp16le' for 16-bit planar, 'rgb24' for 8-bit packed).
    metadata : dict
        Metadata to embed in the video container.
    qr_metadata : bool
        If True, prepend a QR code frame with metadata (survives re-encoding).

    Returns
    -------
    str
        The actual pixel format used.
    """
    params = params or {}
    metadata = metadata or {}

    # Generate QR frame if requested (before dtype conversion)
    qr_frame = None
    if qr_metadata and HAS_QR:
        qr_frame = _create_qr_frame(metadata, (width, height))

    # Validate and convert input dtype
    if codec == 'ffv1' and pix_fmt == 'gbrp16le':
        if array.shape[-1] != 3:
            raise ValueError('gbrp16le requires exactly 3 channels')
        if array.dtype == np.float32:
            array = np.clip(array, 0, 65535).astype(np.uint16)
        elif array.dtype != np.uint16:
            array = array.astype(np.uint16)
        encode_pix_fmt = 'gbrp16le'
    else:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] != 3:
            raise ValueError('Only 3-channel arrays supported')
        pix_fmt = 'rgb24'
        encode_pix_fmt = 'yuv420p' if codec == 'libx264' else 'rgb24'

    # Determine container format
    ext = os.path.splitext(output_path)[1].lower()
    format_map = {'.mkv': 'matroska', '.mp4': 'mp4', '.webm': 'webm', '.avi': 'avi'}
    container_format = format_map.get(ext, 'matroska')

    # Create container and stream
    container = av.open(output_path, mode='w', format=container_format)
    stream = container.add_stream(codec, rate=30)
    stream.width = width
    stream.height = height
    stream.pix_fmt = encode_pix_fmt

    # Codec-specific options
    if codec == 'ffv1':
        stream.options = {'slicecrc': '1'}
    elif codec == 'libx264':
        crf = params.get('crf', 23)
        preset = params.get('preset', 'medium')
        stream.options = {'crf': str(crf), 'preset': preset}
    elif codec == 'libx265':
        crf = params.get('crf', 23)
        preset = params.get('preset', 'medium')
        stream.options = {'crf': str(crf), 'preset': preset}
    elif codec == 'libvpx-vp9':
        crf = params.get('crf', 30)
        stream.options = {'crf': str(crf)}

    # Mark QR header before encoding metadata
    if qr_frame is not None:
        metadata['HAS_QR_HEADER'] = True

    # Embed metadata in container
    encoded_meta = _encode_metadata(metadata)
    if container_format == 'matroska':
        # MKV supports arbitrary custom tags
        container.metadata['VPARQUET'] = VPARQUET_MARKER
        container.metadata['VPARQUET_META'] = encoded_meta
    else:
        # MP4/WebM/AVI: use 'comment' field with marker prefix
        container.metadata['comment'] = f'{VPARQUET_MARKER}:{encoded_meta}'

    # Track frame index for pts
    frame_idx = 0

    # Write QR frame first if requested
    if qr_frame is not None:
        qr_av_frame = av.VideoFrame.from_ndarray(qr_frame, format='rgb24')
        if encode_pix_fmt != 'rgb24':
            qr_av_frame = qr_av_frame.reformat(format=encode_pix_fmt)
        qr_av_frame.pts = frame_idx
        for packet in stream.encode(qr_av_frame):
            container.mux(packet)
        frame_idx += 1

    # Write data frames
    for i in range(array.shape[0]):
        frame_data = array[i]  # (H, W, C)

        if pix_fmt == 'gbrp16le':
            frame = av.VideoFrame(width, height, 'gbrp16le')
            planar_data = np.transpose(frame_data, (2, 0, 1))  # (C, H, W)

            for plane_idx in range(3):
                plane = frame.planes[plane_idx]
                plane_array = planar_data[plane_idx]
                line_size_elements = plane.line_size // 2

                if line_size_elements > width:
                    padded = np.zeros((height, line_size_elements), dtype=np.uint16)
                    padded[:, :width] = plane_array
                    plane.update(padded.tobytes())
                else:
                    plane.update(np.ascontiguousarray(plane_array).tobytes())
        else:
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(frame_data), format='rgb24'
            )
            if encode_pix_fmt != 'rgb24':
                frame = frame.reformat(format=encode_pix_fmt)

        frame.pts = frame_idx
        frame_idx += 1
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return encode_pix_fmt


def read_video(input_path):
    """
    Read a video file into a numpy array (T, H, W, C).

    Metadata is extracted from the MKV container - no sidecar files needed.

    Parameters
    ----------
    input_path : str
        Path to the video file.

    Returns
    -------
    tuple
        (video_data, metadata) where video_data has shape (T, H, W, C).
    """
    container = av.open(input_path)

    # Extract embedded metadata from container
    container_meta = dict(container.metadata)
    metadata = None

    # Try MKV custom tags first
    if 'VPARQUET' in container_meta:
        encoded_meta = container_meta.get('VPARQUET_META', '')
        if encoded_meta:
            metadata = _decode_metadata(encoded_meta)

    # Try comment field (MP4/WebM/AVI)
    if metadata is None and 'comment' in container_meta:
        comment = container_meta['comment']
        if comment.startswith(VPARQUET_MARKER + ':'):
            encoded_meta = comment[len(VPARQUET_MARKER) + 1:]
            metadata = _decode_metadata(encoded_meta)

    # Try QR code in first frame (for re-encoded videos)
    qr_checked = False
    if metadata is None and HAS_QR:
        # Read first frame and check for QR
        for frame in container.decode(video=0):
            first_frame = frame.to_ndarray(format='rgb24')
            metadata = _decode_qr_frame(first_frame)
            qr_checked = True
            break
        # Reset container to read from start
        container.close()
        container = av.open(input_path)

    # Legacy support: try sidecar file
    if metadata is None:
        sidecar_path = f"{input_path}.meta.json"
        if os.path.exists(sidecar_path):
            with open(sidecar_path, 'r') as f:
                metadata = json.load(f)
        else:
            container.close()
            raise RuntimeError(f'Not a videoparquet file (no embedded metadata): {input_path}')

    stream = container.streams.video[0]
    width = stream.width
    height = stream.height
    actual_pix_fmt = stream.pix_fmt

    output_pix_fmt = metadata.get('OUT_PIX_FMT', 'rgb24')
    has_qr_header = metadata.get('HAS_QR_HEADER', False)

    # Read frames
    frames = []
    frame_count = 0
    for frame in container.decode(video=0):
        # Skip QR header frame if present
        if has_qr_header and frame_count == 0:
            frame_count += 1
            continue
        frame_count += 1
        if output_pix_fmt == 'gbrp16le':
            planes_data = []
            for plane_idx in range(3):
                plane = frame.planes[plane_idx]
                plane_bytes = bytes(plane)
                plane_array = np.frombuffer(plane_bytes, dtype=np.uint16)

                line_size = plane.line_size // 2
                if line_size != frame.width:
                    plane_array = plane_array.reshape(frame.height, line_size)[:, :frame.width]
                else:
                    plane_array = plane_array.reshape(frame.height, frame.width)
                planes_data.append(plane_array.copy())

            frame_data = np.stack(planes_data, axis=0)
            frame_data = np.transpose(frame_data, (1, 2, 0))
        else:
            if actual_pix_fmt != output_pix_fmt:
                frame = frame.reformat(format=output_pix_fmt)
            frame_data = frame.to_ndarray(format=output_pix_fmt)
            if frame_data.ndim == 2:
                frame_data = frame_data[..., np.newaxis]

        frames.append(frame_data)

    container.close()

    video_data = np.stack(frames, axis=0)

    # Verify shape
    expected_shape = metadata.get('shape')
    if expected_shape and video_data.size != np.prod(expected_shape):
        raise ValueError(f"Size mismatch: got {video_data.size}, expected {np.prod(expected_shape)}")

    return video_data, metadata


def is_videoparquet(input_path):
    """
    Check if a video file is a videoparquet file with embedded metadata.

    Parameters
    ----------
    input_path : str
        Path to the video file.

    Returns
    -------
    bool
        True if the file has videoparquet metadata embedded.
    """
    try:
        container = av.open(input_path)
        meta = dict(container.metadata)
        container.close()

        # Check MKV custom tag
        if 'VPARQUET' in meta:
            return True

        # Check comment field (MP4/WebM/AVI)
        comment = meta.get('comment', '')
        if comment.startswith(VPARQUET_MARKER + ':'):
            return True

        return False
    except Exception:
        return False


def get_embedded_metadata(input_path):
    """
    Extract metadata from a videoparquet file without decoding video.

    Parameters
    ----------
    input_path : str
        Path to the video file.

    Returns
    -------
    dict
        The embedded metadata, or None if not a videoparquet file.
    """
    try:
        container = av.open(input_path)
        container_meta = dict(container.metadata)
        container.close()

        # Try MKV custom tag
        if 'VPARQUET' in container_meta:
            encoded_meta = container_meta.get('VPARQUET_META', '')
            if encoded_meta:
                return _decode_metadata(encoded_meta)

        # Try comment field (MP4/WebM/AVI)
        comment = container_meta.get('comment', '')
        if comment.startswith(VPARQUET_MARKER + ':'):
            encoded_meta = comment[len(VPARQUET_MARKER) + 1:]
            return _decode_metadata(encoded_meta)

        return None
    except Exception:
        return None
