import numpy as np

def calculate_iou(pred_mask: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates Intersection over Union (IoU) for a predicted mask vs ground truth.
    Both arrays are expected to be 2D arrays of floats [0, 1] or booleans.
    """
    pred_bin = (pred_mask > threshold).astype(bool)
    true_bin = (true_mask > threshold).astype(bool)
    
    intersection = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return intersection / union

def calculate_dice(pred_mask: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates Dice coefficient for a predicted mask vs ground truth.
    """
    pred_bin = (pred_mask > threshold).astype(bool)
    true_bin = (true_mask > threshold).astype(bool)
    
    intersection = np.logical_and(pred_bin, true_bin).sum()
    total_pixels = pred_bin.sum() + true_bin.sum()
    
    if total_pixels == 0:
        return 1.0
        
    return (2. * intersection) / total_pixels
