import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PurePyTorchGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        support = torch.mm(x, self.weight)
        return torch.mm(adj_norm, support)


def build_normalized_adj(adj_matrix):
    A_tilde = adj_matrix + torch.eye(adj_matrix.size(0), device=adj_matrix.device)
    deg = torch.sum(A_tilde, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_mat = torch.diag(deg_inv_sqrt)
    return torch.mm(torch.mm(D_mat, A_tilde), D_mat)


class SpatialDecoder(nn.Module):
    def __init__(self, in_features=640):
        super().__init__()
        self.fc = nn.Linear(in_features, 512 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 512, 8, 8)
        return self.deconv(x)


class HybridCNN_GCN_Model(nn.Module):
    def __init__(self, in_channels, n_districts=35):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        orig_stem = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            in_channels, orig_stem.out_channels, orig_stem.kernel_size,
            orig_stem.stride, orig_stem.padding, bias=False
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3)
        )
        self.gcn1 = PurePyTorchGCNLayer(512, 256)
        self.gcn2 = PurePyTorchGCNLayer(256, 128)
        self.decoder = SpatialDecoder(in_features=512 + 128)
        self.n_districts = n_districts

    def forward(self, images, district_ids, adj_norm):
        patch_feats = self.cnn_fc(self.backbone(images))
        district_feats = torch.zeros(self.n_districts, 512, device=images.device)
        for d in range(self.n_districts):
            mask = (district_ids == d)
            if mask.any():
                district_feats[d] = patch_feats[mask].mean(0)

        g1 = F.relu(self.gcn1(district_feats, adj_norm))
        gcn_feats = F.relu(self.gcn2(g1, adj_norm))
        gcn_per_patch = gcn_feats[district_ids]
        combined = torch.cat([patch_feats, gcn_per_patch], dim=1)
        return self.decoder(combined)


class SpatialFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * ((1 - pt) ** self.gamma) * bce
        return loss.mean()