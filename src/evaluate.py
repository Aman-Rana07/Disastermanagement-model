"""Pixel-level binary segmentation metrics."""

import torch

from config import DEVICE


def compute_metrics(probabilities, labels, valid_mask=None, threshold=0.5):
    """Compute thresholded metrics over valid pixels only."""
    probabilities = probabilities.float().reshape(-1)
    labels = labels.float().reshape(-1)
    if probabilities.shape != labels.shape:
        raise ValueError("Prediction and label arrays must have equal size.")
    if valid_mask is not None:
        valid_mask = valid_mask.bool().reshape(-1)
        if valid_mask.shape != labels.shape:
            raise ValueError("Validity mask and labels must have equal size.")
        probabilities = probabilities[valid_mask]
        labels = labels[valid_mask]
    if probabilities.numel() == 0:
        raise ValueError("Cannot evaluate a set with no valid labeled pixels.")

    predicted = probabilities >= threshold
    actual = labels >= 0.5
    tp = (predicted & actual).sum().float()
    fp = (predicted & ~actual).sum().float()
    fn = (~predicted & actual).sum().float()
    tn = (~predicted & ~actual).sum().float()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    brier = ((probabilities - labels) ** 2).mean()
    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "brier": brier.item(),
        "TP": int(tp.item()),
        "FP": int(fp.item()),
        "FN": int(fn.item()),
        "TN": int(tn.item()),
        "valid_pixels": int(probabilities.numel()),
    }


@torch.no_grad()
def evaluate_model(model, loader, device=None, threshold=0.5):
    if device is None:
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model.eval()
    totals = torch.zeros(6, dtype=torch.float64)
    for images, labels, valid in loader:
        probabilities = torch.sigmoid(model(images.to(device))).cpu()
        labels = labels.cpu()
        valid = valid.cpu().bool()
        if not valid.any():
            continue
        predicted = probabilities[valid] >= threshold
        actual = labels[valid] >= 0.5
        totals += torch.tensor(
            [
                (predicted & actual).sum().item(),
                (predicted & ~actual).sum().item(),
                (~predicted & actual).sum().item(),
                (~predicted & ~actual).sum().item(),
                ((probabilities[valid] - labels[valid]) ** 2).sum().item(),
                valid.sum().item(),
            ],
            dtype=torch.float64,
        )
    if totals[5].item() == 0:
        raise ValueError("Evaluation loader contains no batches.")
    tp, fp, fn, tn, brier_sum, valid_pixels = totals.tolist()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "accuracy": (tp + tn) / (tp + fp + fn + tn + 1e-8),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
        "brier": brier_sum / valid_pixels,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "valid_pixels": int(valid_pixels),
    }


def print_metrics(metrics, split="Validation"):
    print(f"\n{split} metrics (valid pixels: {metrics['valid_pixels']:,})")
    for name in ("accuracy", "precision", "recall", "f1", "brier"):
        print(f"  {name.title():10s}: {metrics[name]:.4f}")
    print(
        f"  TP:{metrics['TP']} FP:{metrics['FP']} "
        f"FN:{metrics['FN']} TN:{metrics['TN']}"
    )
