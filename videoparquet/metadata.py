"""
Metadata handling utilities for videoparquet.
"""

import json
from pathlib import Path

def save_metadata(metadata, path):
    """Save metadata dict as JSON to the given path."""
    with open(path, 'w') as f:
        json.dump(metadata, f)

def load_metadata(path):
    """Load metadata dict from JSON file at the given path."""
    with open(path, 'r') as f:
        return json.load(f) 