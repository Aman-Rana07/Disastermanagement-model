"""Masked losses for imbalanced binary flood segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FOCAL_ALPHA, FOCAL_GAMMA


class CombinedLoss(nn.Module):
    """Focal BCE plus soft Dice, ignoring pixels marked invalid."""

    def __init__(self, weight_focal=0.6, weight_dice=0.4,
                 alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, smooth=1.0):
        super().__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target, valid_mask=None):
        if logits.shape != target.shape:
            raise ValueError(
                f"Logit and target shapes differ: {logits.shape} vs {target.shape}."
            )
        target = target.float()
        if valid_mask is None:
            valid_mask = torch.ones_like(target, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()
        if valid_mask.shape != target.shape:
            raise ValueError("Validity mask must have the same shape as target.")
        if not valid_mask.any():
            raise ValueError("Batch contains no valid labeled pixels.")

        valid_float = valid_mask.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = torch.exp(-bce)
        alpha_factor = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = alpha_factor * (1.0 - pt).pow(self.gamma) * bce
        focal_loss = (focal * valid_float).sum() / valid_float.sum()

        probabilities = probabilities * valid_float
        target = target * valid_float
        intersection = (probabilities * target).sum()
        dice = (2 * intersection + self.smooth) / (
            probabilities.sum() + target.sum() + self.smooth
        )
        return self.weight_focal * focal_loss + self.weight_dice * (1.0 - dice)
