import os
import glob
import torch
import rasterio
import numpy as np
import geopandas as gpd
from torch.utils.data import Dataset

class AssamFloodDataset(Dataset):
    def __init__(self, raster_dir, mask_dir, shapefile_path=None, transform=None):
        """
        Custom PyTorch Dataset for loading multi-channel GeoTIFF rasters and flood masks.
        """
        self.raster_paths = sorted(glob.glob(os.path.join(raster_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        self.transform = transform

        # Load District Boundaries Shapefile if available
        self.district_gdf = gpd.read_file(shapefile_path) if shapefile_path and os.path.exists(shapefile_path) else None

    def __len__(self):
        return len(self.raster_paths)

    def __getitem__(self, idx):
        raster_path = self.raster_paths[idx]
        mask_path = self.mask_paths[idx]

        # 1. Read Multi-spectral GeoTIFF (Channels, Height, Width)
        with rasterio.open(raster_path) as src:
            image = src.read()  # Float32 array
            # Convert NaN/Inf values to zero
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Read Binary Label Mask (1, Height, Width)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Read single target channel
            mask = np.nan_to_num(mask, nan=0.0)
            mask = np.expand_dims(mask, axis=0)

        # 3. Extract District ID (Assigned sequentially or parsed from filename)
        district_id = idx % 35  # Fallback 0-34 indexing for GCN batch alignment

        # 4. Normalize image channel values
        image = (image - np.mean(image)) / (np.std(image) + 1e-6)

        # 5. Convert to PyTorch Tensors
        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        district_tensor = torch.tensor(district_id, dtype=torch.long)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor, district_tensor