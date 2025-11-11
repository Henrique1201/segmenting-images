import cv2
import numpy as np

def kmeans_segment(img_rgb, k=3):
    data = img_rgb.reshape((-1, 3)).astype('float32')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype('uint8')
    segmented = centers[labels.flatten()].reshape(img_rgb.shape)
    return segmented
