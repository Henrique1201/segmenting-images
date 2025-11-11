import numpy as np
import cv2
from skimage.filters import threshold_otsu

def otsu_segment(img_gray):
    if img_gray.dtype != np.uint8:
        img = (img_gray*255).astype(np.uint8)
    else:
        img = img_gray
    thresh = threshold_otsu(img)
    mask = (img > thresh).astype(np.uint8)
    return mask

def adaptive_segment(img_gray, block_size=35, C=5):
    if block_size % 2 == 0:
        block_size += 1
    img = img_gray if img_gray.dtype == np.uint8 else (img_gray*255).astype(np.uint8)
    mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, C)
    return (mask//255).astype(np.uint8)
