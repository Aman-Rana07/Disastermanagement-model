"""U-Net style pixel-wise flood segmentation model."""

import torch
import torch.nn as nn


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AssamFloodModel(nn.Module):
    """Predict one flood logit per input pixel; use sigmoid only at inference."""

    def __init__(self, in_channels, base_channels=16, dropout=0.1):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8, dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(
            base_channels * 8, base_channels * 16, dropout
        )
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.decoder4 = ConvBlock(base_channels * 16, base_channels * 8, dropout)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.decoder3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    @staticmethod
    def _match_size(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=reference.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input shaped (B,{self.in_channels},H,W), "
                f"got {tuple(images.shape)}."
            )
        e1 = self.encoder1(images)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.decoder4(torch.cat((self._match_size(self.up4(x), e4), e4), 1))
        x = self.decoder3(torch.cat((self._match_size(self.up3(x), e3), e3), 1))
        x = self.decoder2(torch.cat((self._match_size(self.up2(x), e2), e2), 1))
        x = self.decoder1(torch.cat((self._match_size(self.up1(x), e1), e1), 1))
        return self.output(x)


def count_parameters(model):
    """Return and print the number of trainable model parameters."""
    total = sum(parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad)
    print(f"Trainable parameters: {total:,} ({total / 1e6:.2f}M)")
    return total
