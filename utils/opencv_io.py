import os

import cv2
import numpy as np


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Read an image from a Unicode or UNC path while preserving cv2.imread semantics."""
    try:
        encoded = np.fromfile(os.fspath(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if encoded.size == 0:
        return None
    try:
        return cv2.imdecode(encoded, flags)
    except cv2.error:
        return None


def imread_required(path, flags=cv2.IMREAD_COLOR):
    """Read an image and raise a path-specific error when it cannot be decoded."""
    image = imread_unicode(path, flags)
    if image is None:
        raise RuntimeError(f'Failed to open or decode image: {os.fspath(path)}')
    return image
