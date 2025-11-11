import numpy as np

def compute_iou(gt, pred, eps=1e-7):
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return inter / (union + eps)

def compute_dice(gt, pred, eps=1e-7):
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    inter = np.logical_and(gt, pred).sum()
    return 2 * inter / (gt.sum() + pred.sum() + eps)
