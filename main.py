# ═══════════════════════════════════════════════
# main.py — Entry point, runs full pipeline
# ═══════════════════════════════════════════════

import torch
import numpy as np
import os

from config import (FEATURE_STACK, FLOOD_MASK, TERRAIN_STACK,
                    PROCESSED_DIR, EXPORTS_DIR, DEVICE,
                    IN_CHANNELS, N_DISTRICTS)

from src.utils      import (load_geotiff, resample_to_match,
                             normalize_features, extract_patches,
                             spatial_block_split)
from src.dataset    import (get_dataloaders, assign_district_ids)
from src.graph      import build_edge_index
from src.model      import AssamFloodModel, count_parameters
from src.train      import train
from src.evaluate   import evaluate_model, print_metrics
from src.predict    import predict_full_scene, load_model
from src.utils      import save_prediction_geotiff


def main():
    device = torch.device(
        DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\n{'═'*50}")
    print(f"  EarthSense AI — Assam 2022 Flood Prediction")
    print(f"  Device: {device}")
    print(f"{'═'*50}\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(EXPORTS_DIR,   exist_ok=True)

    # ── 1. Load Data ────────────────────────────
    print("STEP 1: Loading satellite data...")
    features, feat_meta, transform, crs = load_geotiff(
        FEATURE_STACK)
    labels,   _,         _,         _   = load_geotiff(
        FLOOD_MASK)

    # ── 2. Load & Resample Terrain (TWI, TRI) ───
    print("\nSTEP 2: Loading terrain indices...")
    terrain_resampled_path = os.path.join(
        PROCESSED_DIR, 'terrain_10m.tif')

    if not os.path.exists(terrain_resampled_path):
        terrain_10m = resample_to_match(
            TERRAIN_STACK,
            FEATURE_STACK,
            terrain_resampled_path
        )
    else:
        terrain_data, _, _, _ = load_geotiff(
            terrain_resampled_path)
        terrain_10m = terrain_data

    # Add TWI + TRI as channels 21 & 22
    features = np.concatenate(
        [features, terrain_10m[:2]], axis=0)
    print(f"Final feature channels: {features.shape[0]}")

    # ── 3. Normalize ────────────────────────────
    print("\nSTEP 3: Normalizing features...")
    features_norm = normalize_features(features)

    # ── 4. Extract Patches ──────────────────────
    print("\nSTEP 4: Extracting patches...")
    patches_x, patches_y, coords = extract_patches(
        features_norm, labels)

    # ── 5. Assign District IDs ──────────────────
    print("\nSTEP 5: Assigning district IDs...")
    total_rows = features.shape[1]
    total_cols = features.shape[2]
    dist_ids   = assign_district_ids(
        coords, total_rows=total_rows, total_cols=total_cols)

    # ── 6. Spatial Split ────────────────────────
    print("\nSTEP 6: Spatial block split...")
    splits = spatial_block_split(
        patches_x, patches_y, coords, total_rows)

    # Split district IDs the same way
    block_size  = total_rows // 5
    blocks      = np.array([r // block_size for r, c in coords])
    train_idx   = np.where(np.isin(blocks, [0,1,2]))[0]
    val_idx     = np.where(blocks == 3)[0]
    test_idx    = np.where(blocks == 4)[0]

    train_loader, val_loader, test_loader = get_dataloaders(
        splits,
        district_ids_train = dist_ids[train_idx],
        district_ids_val   = dist_ids[val_idx],
        district_ids_test  = dist_ids[test_idx]
    )

    # ── 7. Build Graph ──────────────────────────
    print("\nSTEP 7: Building Assam district graph...")
    edge_index = build_edge_index(device=str(device))

    # ── 8. Build Model ──────────────────────────
    print("\nSTEP 8: Initializing model...")
    model = AssamFloodModel(n_districts=N_DISTRICTS)
    count_parameters(model)

    # ── 9. Train ────────────────────────────────
    print("\nSTEP 9: Training...")
    history = train(model, train_loader, val_loader,
                    edge_index, device)

    # ── 10. Test Evaluation ─────────────────────
    print("\nSTEP 10: Final test evaluation...")
    model = load_model(AssamFloodModel)
    test_metrics, _, _ = evaluate_model(
        model, test_loader, edge_index.to(device), device)
    print_metrics(test_metrics, split='Test')

    # ── 11. Predict Full Scene ──────────────────
    print("\nSTEP 11: Predicting full Assam flood map...")
    prob_map, mask_map = predict_full_scene(
        model, features_norm, edge_index.to(device))

    save_prediction_geotiff(
        prob_map, FEATURE_STACK,
        os.path.join(EXPORTS_DIR, 'flood_probability_map.tif'))
    save_prediction_geotiff(
        mask_map, FEATURE_STACK,
        os.path.join(EXPORTS_DIR, 'flood_mask_predicted.tif'))

    print(f"\n{'═'*50}")
    print(f"  Pipeline complete!")
    print(f"  Outputs saved to: {EXPORTS_DIR}")
    print(f"{'═'*50}\n")


if __name__ == '__main__':
    main()