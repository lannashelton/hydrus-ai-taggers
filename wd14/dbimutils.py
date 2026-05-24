import cv2
import numpy as np

def make_square(img, size):
    """
    Pad the image with white pixels so it becomes a square.
    The square size will be max(width, height) – the 'size' parameter
    is kept for compatibility but not used.
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    delta_w = max_dim - w
    delta_h = max_dim - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [255, 255, 255]                     # white BGR
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


def smart_resize(img, size):
    """
    Resize the image so the larger side fits into `size`, preserving aspect ratio,
    then pad with white to produce a square of `size x size`.
    """
    h, w = img.shape[:2]
    if h > size or w > size:
        ratio = size / max(h, w)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    delta_w = size - img.shape[1]
    delta_h = size - img.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [255, 255, 255]
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
