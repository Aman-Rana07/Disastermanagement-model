# ═══════════════════════════════════════════════
# predict.py — Inference on new satellite data
# ═══════════════════════════════════════════════

import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE, MODEL_SAVE_PATH, PATCH_SIZE, STRIDE
from src.utils import save_prediction_geotiff


@torch.no_grad()
def predict_full_scene(model, features_norm, edge_index,
                       district_id_map=None, device=None,
                       threshold=0.5):
    """
    Run inference on a full normalized feature stack.
    Reconstructs full flood probability map by averaging
    overlapping patch predictions.

    Args:
        model:         trained AssamFloodModel
        features_norm: (C, H, W) normalized feature array
        edge_index:    district graph edge index
        district_id_map: (H, W) int array of district IDs per pixel
        threshold:     flood probability threshold (default 0.5)
    Returns:
        flood_prob_map: (H, W) float — probability per pixel
        flood_mask_map: (H, W) binary — 1=flooded, 0=not
    """
    if device is None:
        device = torch.device(
            DEVICE if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)
    edge_index = edge_index.to(device)

    C, H, W = features_norm.shape
    prob_map  = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for row in range(0, H - PATCH_SIZE, STRIDE):
        for col in range(0, W - PATCH_SIZE, STRIDE):
            patch = features_norm[:,
                                   row:row + PATCH_SIZE,
                                   col:col + PATCH_SIZE]

            # Skip mostly-empty patches
            if not np.isfinite(patch).mean() > 0.7:
                continue

            patch_tensor = torch.from_numpy(
                patch[np.newaxis]).float().to(device)

            # Get district ID for this patch location
            if district_id_map is not None:
                cy = row + PATCH_SIZE // 2
                cx = col + PATCH_SIZE // 2
                dist_id = int(district_id_map[cy, cx])
            else:
                dist_id = 0

            dist_tensor = torch.tensor(
                [dist_id], dtype=torch.long, device=device)

            # Predict flood probability for this patch
            prob = model(patch_tensor,
                         dist_tensor,
                         edge_index).item()

            # Add to probability map (averaging overlaps)
            prob_map[row:row + PATCH_SIZE,
                     col:col + PATCH_SIZE] += prob
            count_map[row:row + PATCH_SIZE,
                      col:col + PATCH_SIZE] += 1

    # Average overlapping predictions
    count_map = np.maximum(count_map, 1)
    prob_map  = prob_map / count_map
    mask_map  = (prob_map > threshold).astype(np.float32)

    flooded_km2 = mask_map.sum() * (10 * 10) / 1e6
    print(f"Predicted flooded area: {flooded_km2:.1f} km²")
    print(f"  (Actual 2022: ~33,902 km²)")

    return prob_map, mask_map


def load_model(model_class, checkpoint_path=None):
    """Load trained model from checkpoint"""
    if checkpoint_path is None:
        checkpoint_path = MODEL_SAVE_PATH

    device     = torch.device(
        DEVICE if torch.cuda.is_available() else 'cpu')
    model      = model_class().to(device)
    checkpoint = torch.load(checkpoint_path,
                             map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(F1={checkpoint['f1']:.4f})")
    return model