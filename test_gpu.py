import torch
from src.model import HybridCNN_GCN_Model, SpatialFocalLoss, build_normalized_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Testing hybrid pipeline on: {torch.cuda.get_device_name(0)}")

# 1. Create synthetic dummy batch (4 patches, 4 bands, 256x256 resolution)
images = torch.randn(4, 4, 256, 256, device=device)
labels = torch.randint(0, 2, (4, 1, 256, 256), dtype=torch.float32, device=device)
district_ids = torch.tensor([0, 1, 2, 3], device=device)

# 2. Build 35-node graph adjacency matrix
adj_matrix = torch.eye(35, device=device)
adj_norm = build_normalized_adj(adj_matrix)

# 3. Instantiate model & loss function
model = HybridCNN_GCN_Model(in_channels=4, n_districts=35).to(device)
criterion = SpatialFocalLoss()

# 4. Test Forward & Backward passes
output = model(images, district_ids, adj_norm)
loss = criterion(output, labels)
loss.backward()

print(f"  ✓ Model Output Shape: {output.shape}")
print(f"  ✓ Spatial Focal Loss: {loss.item():.4f}")
print("🚀 Hybrid CNN+GCN architecture and CUDA memory allocation verified on RTX 4050!")