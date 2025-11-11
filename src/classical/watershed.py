import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import color

def watershed_segment(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)
    dist = ndi.distance_transform_edt(closing)
    coords = peak_local_max(dist, footprint=np.ones((3,3)), labels=closing)
    mask = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-dist, markers, mask=closing)
    return (labels > 0).astype(np.uint8)
