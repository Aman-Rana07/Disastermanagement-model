# ═══════════════════════════════════════════════
# losses.py — Focal Loss for flood prediction
# ═══════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import FOCAL_ALPHA, FOCAL_GAMMA


class FocalLoss(nn.Module):
    """
    Focal Loss for binary flood segmentation.

    WHY Focal Loss over BCE:
    Flood pixels are spatially rare vs non-flood pixels.
    Standard BCE treats all pixels equally — the model learns
    to predict "not flooded" everywhere and still gets 80%+ accuracy.
    Focal Loss down-weights easy correct predictions (non-flood pixels
    the model is confident about) and focuses training on hard cases
    (flood boundaries, partially submerged areas).

    Formula: FL = -α(1-pt)^γ * log(pt)
    γ=2 means a well-classified pixel contributes 4× less to loss
    than a misclassified pixel.
    """
    def __init__(self, alpha=None, gamma=None):
        super().__init__()
        self.alpha = alpha if alpha is not None else FOCAL_ALPHA
        self.gamma = gamma if gamma is not None else FOCAL_GAMMA

    def forward(self, pred, target):
        """
        Args:
            pred:   (B,) or (B,1) float — predicted flood probability
            target: (B,) float — ground truth (0 or 1)
        """
        pred   = pred.squeeze()
        target = target.float().squeeze()

        # Clamp for numerical stability
        pred   = torch.clamp(pred, 1e-7, 1 - 1e-7)
        bce    = F.binary_cross_entropy(pred, target, reduction='none')
        pt     = torch.exp(-bce)
        focal  = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for pixel-level segmentation.
    Measures overlap between prediction and ground truth.
    Particularly effective for imbalanced classes.

    Formula: Dice = 1 - (2|A∩B|) / (|A| + |B|)
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = pred.squeeze().float()
        target = target.squeeze().float()

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss.
    Focal handles class imbalance at pixel level.
    Dice handles spatial overlap between predicted and actual flood extent.
    Together they optimize both precision and spatial accuracy.

    weight_focal + weight_dice should sum to 1.0
    """
    def __init__(self, weight_focal=0.6, weight_dice=0.4):
        super().__init__()
        self.focal        = FocalLoss()
        self.dice         = DiceLoss()
        self.weight_focal = weight_focal
        self.weight_dice  = weight_dice

    def forward(self, pred, target):
        l_focal = self.focal(pred, target)
        l_dice  = self.dice(pred, target)
        return (self.weight_focal * l_focal +
                self.weight_dice  * l_dice)