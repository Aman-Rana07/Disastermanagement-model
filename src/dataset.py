# ═══════════════════════════════════════════════
# dataset.py — PyTorch Dataset for flood patches
# ═══════════════════════════════════════════════

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY


class AssamFloodDataset(Dataset):
    """
    PyTorch Dataset for Assam 2022 flood patches.

    Each item returns:
        image:       (C, H, W) float32 tensor — satellite features
        label:       (H, W)    float32 tensor — flood mask (0/1)
        district_id: int       — which Assam district this patch is in
    """
    def __init__(self, images, labels, district_ids=None,
                 augment=False):
        """
        Args:
            images:       numpy (N, C, H, W)
            labels:       numpy (N, H, W)
            district_ids: numpy (N,) int — district index per patch
            augment:      apply augmentation (train only)
        """
        self.images       = images
        self.labels       = labels
        self.augment      = augment
        self.district_ids = (district_ids if district_ids is not None
                             else np.zeros(len(images), dtype=np.int64))

        # Augmentation pipeline — only spatial transforms
        # WHY no color/brightness: satellite bands are physically
        # calibrated — distorting them breaks spectral indices
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                p=0.3
            ),
        ]) if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # (C, H, W) → (H, W, C) for albumentations
        image = self.images[idx].transpose(1, 2, 0)
        label = self.labels[idx]

        if self.transform is not None:
            augmented = self.transform(
                image=image,
                mask=label
            )
            image = augmented['image']
            label = augmented['mask']

        # Back to (C, H, W) for PyTorch
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(
                image.transpose(2, 0, 1)).float()
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).float()

        dist_id = int(self.district_ids[idx])

        return image, label, dist_id


def assign_district_ids(coords, assam_shapefile_path=None,
                        n_districts=35, total_rows=None,
                        total_cols=None):
    """
    Assign a district ID (0–34) to each patch based on
    its geographic position.

    Simple approach: divide Assam into a 5×7 grid (35 cells)
    matching the 35 districts — each grid cell = one district.
    For production: use GeoPandas spatial join with actual shapefile.

    Args:
        coords:     list of (row, col) patch top-left corners
        total_rows: total rows in the full raster
        total_cols: total cols in the full raster
    Returns:
        district_ids: numpy (N,) int array
    """
    if total_rows is None or total_cols is None:
        # Default: uniform grid assignment
        return np.zeros(len(coords), dtype=np.int64)

    n_row_blocks = 5   # 5 north-south strips
    n_col_blocks = 7   # 7 east-west strips → 5×7 = 35 districts

    row_size = total_rows // n_row_blocks
    col_size = total_cols // n_col_blocks

    district_ids = []
    for row, col in coords:
        r_block = min(row // row_size, n_row_blocks - 1)
        c_block = min(col // col_size, n_col_blocks - 1)
        dist_id = r_block * n_col_blocks + c_block
        district_ids.append(dist_id)

    return np.array(district_ids, dtype=np.int64)


def get_dataloaders(splits, district_ids_train=None,
                    district_ids_val=None,
                    district_ids_test=None):
    """
    Build PyTorch DataLoaders from train/val/test splits.

    Args:
        splits: dict from spatial_block_split()
        district_ids_*: numpy arrays of district IDs per split
    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = AssamFloodDataset(
        splits['X_train'],
        splits['y_train'],
        district_ids=district_ids_train,
        augment=True      # augment training only
    )
    val_ds = AssamFloodDataset(
        splits['X_val'],
        splits['y_val'],
        district_ids=district_ids_val,
        augment=False
    )
    test_ds = AssamFloodDataset(
        splits['X_test'],
        splits['y_test'],
        district_ids=district_ids_test,
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,           # shuffle training
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True          # avoid single-sample batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print(f"DataLoaders → "
          f"Train: {len(train_loader)} batches | "
          f"Val: {len(val_loader)} batches | "
          f"Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader