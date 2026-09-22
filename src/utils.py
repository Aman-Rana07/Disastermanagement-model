# ═══════════════════════════════════════════════
# utils.py — Normalization, patching, helpers
# ═══════════════════════════════════════════════

import numpy as np
import rasterio
from rasterio.enums import Resampling
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CHANNEL_NAMES, CHANNEL_BOUNDS, PATCH_SIZE, STRIDE, MIN_VALID_RATIO


def load_geotiff(path):
    """
    Load a GeoTIFF file as numpy array.
    Returns: (data array, metadata dict, transform)
    """
    with rasterio.open(path) as src:
        data      = src.read().astype(np.float32)
        meta      = src.meta.copy()
        transform = src.transform
        crs       = src.crs
    print(f"Loaded: {os.path.basename(path)} | "
          f"Shape: {data.shape} | "
          f"Dtype: {data.dtype}")
    return data, meta, transform, crs


def resample_to_match(source_path, reference_path, output_path):
    """
    Resample source raster to match reference raster resolution.
    Used to bring 30m terrain (TWI/TRI) to 10m Sentinel resolution.
    Uses bilinear interpolation — correct for continuous terrain data.
    """
    with rasterio.open(reference_path) as ref:
        target_transform = ref.transform
        target_crs       = ref.crs
        target_height    = ref.height
        target_width     = ref.width
        target_meta      = ref.meta.copy()

    with rasterio.open(source_path) as src:
        data = src.read(
            out_shape=(
                src.count,
                target_height,
                target_width
            ),
            resampling=Resampling.bilinear
        ).astype(np.float32)

    target_meta.update({
        'count': data.shape[0],
        'dtype': 'float32'
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **target_meta) as dst:
        dst.write(data)

    print(f"Resampled {os.path.basename(source_path)} → "
          f"{target_height}×{target_width} px")
    return data


def normalize_features(features, channel_names=None, bounds=None):
    """
    Normalize each channel to [0, 1] using physics-based bounds.
    Handles NoData (-9999, nan, inf) by replacing with channel mean.

    Args:
        features: numpy array (C, H, W)
        channel_names: list of channel names matching CHANNEL_NAMES
        bounds: dict of {name: (min, max)} — uses config if None
    Returns:
        normalized array (C, H, W) float32
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    if bounds is None:
        bounds = CHANNEL_BOUNDS

    normalized = np.zeros_like(features, dtype=np.float32)

    for i, name in enumerate(channel_names):
        if i >= features.shape[0]:
            break

        band = features[i].astype(np.float32).copy()

        # Replace NoData values with valid pixel mean
        valid_mask = np.isfinite(band) & (band > -9990)
        if valid_mask.sum() > 0:
            fill_val = float(np.nanmean(band[valid_mask]))
        else:
            fill_val = 0.0
        band[~valid_mask] = fill_val

        # Clip and normalize to [0, 1]
        lo, hi = bounds.get(name, (band.min(), band.max()))
        band   = np.clip(band, lo, hi)
        band   = (band - lo) / (hi - lo + 1e-8)

        normalized[i] = band

    print(f"Normalized {features.shape[0]} channels | "
          f"Range: {normalized.min():.3f} – {normalized.max():.3f}")
    return normalized


def extract_patches(features, labels, patch_size=None,
                    stride=None, min_valid=None):
    """
    Extract overlapping patches from full-scene rasters.
    Skips NoData-heavy patches and zero-variance patches (clouds).

    Args:
        features: (C, H, W) normalized feature array
        labels:   (1, H, W) or (H, W) binary flood mask
        patch_size: size of square patches (default from config)
        stride:   step between patches (default from config)
        min_valid: minimum fraction of valid pixels (default from config)

    Returns:
        patches_x: (N, C, patch_size, patch_size)
        patches_y: (N, patch_size, patch_size)
        coords:    list of (row, col) top-left corners
    """
    if patch_size is None: patch_size = PATCH_SIZE
    if stride     is None: stride     = STRIDE
    if min_valid  is None: min_valid  = MIN_VALID_RATIO

    # Ensure labels are 2D
    if labels.ndim == 3:
        labels = labels[0]

    _, H, W  = features.shape
    patches_x = []
    patches_y = []
    coords    = []

    total   = 0
    skipped = 0

    for row in range(0, H - patch_size, stride):
        for col in range(0, W - patch_size, stride):
            total += 1

            img = features[:,
                           row:row + patch_size,
                           col:col + patch_size]
            lbl = labels[row:row + patch_size,
                         col:col + patch_size]

            # Skip mostly-NoData patches
            valid_ratio = np.isfinite(img).mean()
            if valid_ratio < min_valid:
                skipped += 1
                continue

            # Skip zero-variance patches (fully clouded/masked)
            if img.std() < 1e-3:
                skipped += 1
                continue

            patches_x.append(img.astype(np.float32))
            patches_y.append(lbl.astype(np.float32))
            coords.append((row, col))

    patches_x = np.array(patches_x)
    patches_y = np.array(patches_y)

    flood_count = (patches_y.mean(axis=(1, 2)) > 0.05).sum()
    print(f"Patches: {len(patches_x)} extracted, "
          f"{skipped} skipped ({total} total) | "
          f"Flood patches (>5% flooded): {flood_count}")
    return patches_x, patches_y, coords


def spatial_block_split(patches_x, patches_y, coords,
                        total_rows, n_blocks=5,
                        train_blocks=None,
                        val_blocks=None,
                        test_blocks=None):
    """
    Split patches into train/val/test by geographic N-S blocks.
    Prevents spatial autocorrelation leakage from random splitting.

    Args:
        patches_x:    (N, C, H, W) image patches
        patches_y:    (N, H, W) label patches
        coords:       list of (row, col) top-left corners
        total_rows:   total number of rows in original raster
        n_blocks:     number of geographic blocks to divide Assam into
    """
    if train_blocks is None: train_blocks = [0, 1, 2]
    if val_blocks   is None: val_blocks   = [3]
    if test_blocks  is None: test_blocks  = [4]

    block_size = total_rows // n_blocks
    blocks     = np.array([r // block_size for r, c in coords])

    train_idx = np.where(np.isin(blocks, train_blocks))[0]
    val_idx   = np.where(np.isin(blocks, val_blocks))[0]
    test_idx  = np.where(np.isin(blocks, test_blocks))[0]

    print(f"Spatial split → "
          f"Train: {len(train_idx)} | "
          f"Val: {len(val_idx)} | "
          f"Test: {len(test_idx)}")

    return {
        'X_train': patches_x[train_idx],
        'y_train': patches_y[train_idx],
        'X_val':   patches_x[val_idx],
        'y_val':   patches_y[val_idx],
        'X_test':  patches_x[test_idx],
        'y_test':  patches_y[test_idx],
    }


def save_prediction_geotiff(pred_array, reference_path,
                             output_path, dtype='float32'):
    """
    Save a prediction array as GeoTIFF with correct
    georeference from the reference raster.
    """
    with rasterio.open(reference_path) as ref:
        meta = ref.meta.copy()

    meta.update({
        'count': 1,
        'dtype': dtype
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(pred_array[np.newaxis, :, :].astype(dtype))

    print(f"Saved prediction → {output_path}")