import numpy as np
from scipy.spatial.distance import directed_hausdorff

def dice_coefficient(prediction, target, smooth=1e-6):
    pred_flat = prediction.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    return (2.0 * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)

def mean_absolute_difference(prediction, target):
    return np.mean(np.abs(target.flatten() - prediction.flatten()))

def hausdorff_distance(prediction, target):
    pred_points = np.argwhere(prediction > 0.5)
    target_points = np.argwhere(target > 0.5)
    if len(pred_points) == 0 or len(target_points) == 0: return float('inf')
    forward_hd = directed_hausdorff(pred_points, target_points)[0]
    backward_hd = directed_hausdorff(target_points, pred_points)[0]
    return max(forward_hd, backward_hd)

def calculate_volume_simpson(mask_2d, pixel_spacing=1.0):
    height = mask_2d.shape[0]
    disc_thickness = pixel_spacing
    volume = 0.0
    for row in range(height):
        row_pixels = np.sum(mask_2d[row] > 0.5)
        if row_pixels > 0:
            diameter = row_pixels * pixel_spacing
            area = np.pi * (diameter / 2.0) ** 2
            volume += area * disc_thickness
    return volume / 1000.0

def ejection_fraction(ed_volume, es_volume):
    if ed_volume == 0: return 0.0
    return (ed_volume - es_volume) / ed_volume

def ef_correlation(ef_predicted, ef_ground_truth):
    if len(ef_predicted) < 2: return 0.0
    return np.corrcoef(ef_ground_truth, ef_predicted)[0, 1]

def ef_mae(ef_predicted, ef_ground_truth):
    return np.mean(np.abs(np.array(ef_ground_truth) - np.array(ef_predicted)))
