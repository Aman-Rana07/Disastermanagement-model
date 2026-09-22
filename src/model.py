# ═══════════════════════════════════════════════
# model.py — CNN + GCN flood prediction model
# ═══════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import timm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (IN_CHANNELS, CNN_OUT_DIM, GCN_HIDDEN,
                    GCN_OUT_DIM, N_DISTRICTS, DROPOUT)


class CNNEncoder(nn.Module):
    """EfficientNet-B3 backbone for satellite patch features"""
    def __init__(self, in_channels=IN_CHANNELS,
                 out_features=CNN_OUT_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        orig = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            in_channels,
            orig.out_channels,
            orig.kernel_size,
            orig.stride,
            orig.padding,
            bias=False
        )
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.num_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.proj(self.backbone(x))   # (B, 512)


class SpatialGCN(nn.Module):
    """GCN over Assam district adjacency graph"""
    def __init__(self, in_ch=CNN_OUT_DIM,
                 hid=GCN_HIDDEN, out=GCN_OUT_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid)
        self.bn1   = nn.BatchNorm1d(hid)
        self.conv2 = GCNConv(hid, out)
        self.drop  = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.drop(x)
        x = F.relu(self.conv2(x, edge_index))
        return x   # (N_districts, 128)


class FloodHead(nn.Module):
    """Final prediction head — outputs flood probability per patch"""
    def __init__(self, in_features=CNN_OUT_DIM + GCN_OUT_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)   # (B, 1)


class AssamFloodModel(nn.Module):
    """
    Full CNN + GCN model for Assam 2022 flood prediction.

    Flow:
    1. CNN reads each satellite patch → 512-dim embedding
    2. Embeddings are mean-pooled per district
    3. GCN propagates across Assam district adjacency graph
    4. GCN embeddings broadcast back to patch level
    5. CNN + GCN features concatenated → flood probability
    """
    def __init__(self, n_districts=N_DISTRICTS):
        super().__init__()
        self.cnn        = CNNEncoder()
        self.gcn        = SpatialGCN()
        self.head       = FloodHead()
        self.n_districts = n_districts

    def forward(self, images, district_ids, edge_index):
        """
        Args:
            images:       (B, C, H, W)
            district_ids: (B,) int — district index per patch
            edge_index:   (2, E) — graph connectivity
        Returns:
            flood_prob:   (B, 1) float in [0, 1]
        """
        # 1. CNN features per patch
        patch_feats = self.cnn(images)   # (B, 512)

        # 2. Aggregate to district level
        device = images.device
        dist_feats = torch.zeros(
            self.n_districts, patch_feats.shape[1],
            device=device
        )
        counts = torch.zeros(
            self.n_districts, 1, device=device)

        for d in range(self.n_districts):
            mask = (district_ids == d)
            if mask.any():
                dist_feats[d] = patch_feats[mask].mean(0)
                counts[d]     = mask.sum().float()

        # 3. GCN: spatial propagation across districts
        gcn_feats = self.gcn(dist_feats, edge_index)  # (35, 128)

        # 4. Broadcast back to patch level
        gcn_per_patch = gcn_feats[district_ids]   # (B, 128)

        # 5. Concatenate + predict
        combined   = torch.cat([patch_feats, gcn_per_patch], dim=1)
        flood_prob = self.head(combined)          # (B, 1)

        return flood_prob


def count_parameters(model):
    """Print number of trainable parameters"""
    total = sum(p.numel() for p in model.parameters()
                if p.requires_grad)
    print(f"Trainable parameters: {total:,} "
          f"({total/1e6:.2f}M)")
    return total