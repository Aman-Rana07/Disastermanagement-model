# ═══════════════════════════════════════════════
# evaluate.py — Metrics, confusion matrix, plots
# ═══════════════════════════════════════════════

import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE


def compute_metrics(preds, labels, threshold=0.5):
    """
    Compute all flood prediction metrics.
    WHY these specific metrics:
    - Accuracy: overall but misleading for imbalanced data
    - Precision: don't raise false flood alarms
    - Recall: MOST IMPORTANT — never miss an actual flood
    - F1: balance of precision and recall
    - IoU: standard for segmentation tasks
    """
    preds  = (preds > threshold).float()
    labels = labels.float()

    TP = ((preds == 1) & (labels == 1)).sum().float()
    FP = ((preds == 1) & (labels == 0)).sum().float()
    FN = ((preds == 0) & (labels == 1)).sum().float()
    TN = ((preds == 0) & (labels == 0)).sum().float()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)
    accuracy  = (TP + TN) / (TP + FP + FN + TN + 1e-8)

    return {
        'accuracy':  accuracy.item(),
        'precision': precision.item(),
        'recall':    recall.item(),
        'f1':        f1.item(),
        'iou':       iou.item(),
        'TP': int(TP), 'FP': int(FP),
        'FN': int(FN), 'TN': int(TN)
    }


@torch.no_grad()
def evaluate_model(model, loader, edge_index, device=None):
    """
    Run evaluation on a DataLoader.
    Returns averaged metrics across all batches.
    """
    if device is None:
        device = torch.device(DEVICE)

    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels, dist_ids in loader:
        images   = images.to(device)
        dist_ids = dist_ids.to(device)

        preds = model(images, dist_ids, edge_index)
        all_preds.append(preds.squeeze().cpu())
        all_labels.append(labels.squeeze().cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_preds, all_labels)
    return metrics, all_preds, all_labels


def print_metrics(metrics, split='Val'):
    """Pretty print metrics"""
    print(f"\n{'─'*40}")
    print(f"  {split} Results")
    print(f"{'─'*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}  ← most important")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  TP:{metrics['TP']}  FP:{metrics['FP']}  "
          f"FN:{metrics['FN']}  TN:{metrics['TN']}")
    print(f"{'─'*40}\n")