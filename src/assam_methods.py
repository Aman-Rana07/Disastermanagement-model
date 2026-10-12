"""Spatial graph and attention models for the Assam 2022 event."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import AssamFloodModel


def build_patch_adjacency(grid_size, device=None):
    """Build a symmetric-normalized four-neighbor grid adjacency matrix."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive.")

    node_count = grid_size * grid_size
    adjacency = torch.eye(node_count, dtype=torch.float32)
    for row in range(grid_size):
        for col in range(grid_size):
            node = row * grid_size + col
            for delta_row, delta_col in ((1, 0), (0, 1)):
                neighbor_row = row + delta_row
                neighbor_col = col + delta_col
                if neighbor_row < grid_size and neighbor_col < grid_size:
                    neighbor = neighbor_row * grid_size + neighbor_col
                    adjacency[node, neighbor] = 1.0
                    adjacency[neighbor, node] = 1.0

    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inverse_sqrt_degree = degree.rsqrt()
    adjacency = (
        inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )
    if device is not None:
        adjacency = adjacency.to(device)
    return adjacency


class SpatialPatchGCN(nn.Module):
    """Predict coarse water scores on an 8-pixel spatial graph grid."""

    def __init__(
        self,
        in_channels,
        patch_size=128,
        cell_size=8,
        hidden_channels=32,
        zero_output=False,
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if patch_size < 1 or cell_size < 1 or patch_size % cell_size:
            raise ValueError("patch_size must be divisible by cell_size.")
        self.cell_size = cell_size
        self.grid_size = patch_size // cell_size
        self.graph_conv1 = nn.Linear(in_channels, hidden_channels)
        self.graph_conv2 = nn.Linear(hidden_channels, 1)
        self.register_buffer(
            "adjacency", build_patch_adjacency(self.grid_size)
        )
        if zero_output:
            nn.init.zeros_(self.graph_conv2.weight)
            nn.init.zeros_(self.graph_conv2.bias)

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError("Expected graph features shaped (B,C,H,W).")
        if features.shape[-2] % self.cell_size or features.shape[-1] % self.cell_size:
            raise ValueError("Feature height and width must align to cell_size.")

        height, width = features.shape[-2:]
        pooled = F.avg_pool2d(
            features, kernel_size=self.cell_size, stride=self.cell_size
        )
        grid_height, grid_width = pooled.shape[-2:]
        if (grid_height, grid_width) != (self.grid_size, self.grid_size):
            raise ValueError(
                f"Expected a {self.grid_size}x{self.grid_size} graph grid, "
                f"got {grid_height}x{grid_width}."
            )
        nodes = pooled.flatten(2).transpose(1, 2)
        hidden = torch.matmul(self.adjacency, nodes)
        hidden = F.gelu(self.graph_conv1(hidden))
        hidden = F.dropout(hidden, p=0.1, training=self.training)
        hidden = torch.matmul(self.adjacency, hidden)
        node_logits = self.graph_conv2(hidden)
        grid_logits = node_logits.transpose(1, 2).reshape(
            features.shape[0], 1, grid_height, grid_width
        )
        return F.interpolate(grid_logits, size=(height, width), mode="nearest")


class GCNOnly(nn.Module):
    """A graph-only water mapper operating on VV/VH patch-cell means."""

    def __init__(self, patch_size=128, cell_size=8, hidden_channels=32):
        super().__init__()
        self.graph = SpatialPatchGCN(
            in_channels=2,
            patch_size=patch_size,
            cell_size=cell_size,
            hidden_channels=hidden_channels,
        )

    def forward(self, images):
        return self.graph(images)


class CNNGCNModel(nn.Module):
    """U-Net pixel logits refined by message passing on a spatial patch graph."""

    def __init__(
        self,
        in_channels=2,
        base_channels=8,
        patch_size=128,
        cell_size=8,
        hidden_channels=32,
    ):
        super().__init__()
        self.cnn = AssamFloodModel(
            in_channels=in_channels, base_channels=base_channels
        )
        self.graph = SpatialPatchGCN(
            in_channels=in_channels + 1,
            patch_size=patch_size,
            cell_size=cell_size,
            hidden_channels=hidden_channels,
            zero_output=True,
        )

    def forward(self, images):
        cnn_logits = self.cnn(images)
        graph_features = torch.cat((images, cnn_logits), dim=1)
        return cnn_logits + self.graph(graph_features)


class CNNGCNTransformerModel(nn.Module):
    """Refine U-Net logits with graph message passing and global attention."""

    def __init__(
        self,
        in_channels=2,
        base_channels=8,
        patch_size=128,
        cell_size=8,
        hidden_channels=32,
        attention_heads=4,
        transformer_layers=2,
    ):
        super().__init__()
        if hidden_channels % attention_heads:
            raise ValueError(
                "hidden_channels must be divisible by attention_heads."
            )
        if transformer_layers < 1:
            raise ValueError("transformer_layers must be positive.")
        if patch_size < 1 or cell_size < 1 or patch_size % cell_size:
            raise ValueError("patch_size must be divisible by cell_size.")

        self.cell_size = cell_size
        self.grid_size = patch_size // cell_size
        node_count = self.grid_size * self.grid_size
        self.cnn = AssamFloodModel(
            in_channels=in_channels, base_channels=base_channels
        )
        self.node_projection = nn.Linear(in_channels + 1, hidden_channels)
        self.graph_projection = nn.Linear(hidden_channels, hidden_channels)
        self.register_buffer(
            "adjacency", build_patch_adjacency(self.grid_size)
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, node_count, hidden_channels)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=attention_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.node_classifier = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.node_classifier.weight)
        nn.init.zeros_(self.node_classifier.bias)

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("Expected images shaped (B,C,H,W).")
        height, width = images.shape[-2:]
        if height % self.cell_size or width % self.cell_size:
            raise ValueError("Image height and width must align to cell_size.")

        cnn_logits = self.cnn(images)
        graph_features = torch.cat((images, cnn_logits), dim=1)
        pooled = F.avg_pool2d(
            graph_features, kernel_size=self.cell_size, stride=self.cell_size
        )
        grid_height, grid_width = pooled.shape[-2:]
        if (grid_height, grid_width) != (self.grid_size, self.grid_size):
            raise ValueError(
                f"Expected a {self.grid_size}x{self.grid_size} graph grid, "
                f"got {grid_height}x{grid_width}."
            )

        nodes = pooled.flatten(2).transpose(1, 2)
        nodes = F.gelu(self.node_projection(nodes))
        nodes = F.gelu(
            nodes + self.graph_projection(
                torch.matmul(self.adjacency, nodes)
            )
        )
        nodes = self.transformer(nodes + self.position_embedding)
        node_logits = self.node_classifier(nodes).transpose(1, 2).reshape(
            images.shape[0], 1, grid_height, grid_width
        )
        residual = F.interpolate(
            node_logits, size=(height, width), mode="nearest"
        )
        return cnn_logits + residual
