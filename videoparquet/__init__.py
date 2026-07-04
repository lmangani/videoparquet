"""videoparquet: Parquet <-> Video conversion library (inspired by xarrayvideo)"""

__version__ = '0.3.0'

from .parquet2video import parquet2video
from .video2parquet import video2parquet
from .get_recipe import get_recipe
from .utils import infer_video_shape
from .av_wrappers import is_videoparquet, get_embedded_metadata

__all__ = [
    'parquet2video',
    'video2parquet',
    'get_recipe',
    'infer_video_shape',
    'is_videoparquet',
    'get_embedded_metadata',
] 