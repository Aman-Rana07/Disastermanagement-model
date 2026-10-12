"""Train and score Cachar 2022 water models against a provisional NRSC map proxy."""

import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

from src.assam_methods import (
    CNNGCNModel,
    CNNGCNTransformerModel,
    GCNOnly,
)
from src.assam_plotting import save_assam_metric_comparison
from src.evaluate import compute_metrics
from src.losses import CombinedLoss
from src.model import AssamFloodModel


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILENAME = "assam_cachar_s1_vv_vh_20220710_20220720.tif"
INPUT_CANDIDATES = (
    os.path.join(
        ROOT_DIR, "data", "public_assam_2022", "sentinel1", INPUT_FILENAME
    ),
)
REFERENCE_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "exports",
    "assam_2022",
    "reference",
    "nrsc_cachar_flood_proxy.tif",
)
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "exports", "assam_2022", "trained")
PATCH_SIZE = 128
SPATIAL_BLOCK_TILES = 4
BATCH_SIZE = 8
MAX_EPOCHS = 12
PATIENCE = 3
BASE_CHANNELS = 8
TRANSFORMER_HIDDEN_CHANNELS = 32
TRANSFORMER_ATTENTION_HEADS = 4
TRANSFORMER_LAYERS = 2
MIN_VALID_FRACTION = 0.20
MIN_EPOCH_PATCHES = 64
SPLIT_STRATA = 12
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_THRESHOLDS = {
    "vh_threshold": 0.60,
    "cnn": 0.50,
    "gcn": 0.50,
    "cnn_gcn": 0.50,
    "cnn_gcn_transformer": 0.50,
}
MODEL_LABELS = {
    "vh_threshold": "VH threshold",
    "cnn": "CNN (U-Net)",
    "gcn": "GCN",
    "cnn_gcn": "CNN + GCN",
    "cnn_gcn_transformer": "CNN + GCN + Transformer",
}


def _input_path():
    for candidate in INPUT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "2022 Sentinel-1 VV/VH GeoTIFF not found. Checked: "
        + ", ".join(INPUT_CANDIDATES)
    )


def _normalise(raw, valid):
    raw = raw.astype(np.float32, copy=False)
    raw[0] = np.nan_to_num(
        raw[0], nan=-30.0, posinf=-30.0, neginf=-30.0
    )
    raw[1] = np.nan_to_num(
        raw[1], nan=-35.0, posinf=-35.0, neginf=-35.0
    )
    raw[0] = np.clip((raw[0] + 30.0) / 35.0, 0.0, 1.0)
    raw[1] = np.clip((raw[1] + 35.0) / 35.0, 0.0, 1.0)
    raw[:, ~valid] = 0.0
    return raw


def _align_reference(input_path, reference_path, aligned_path):
    with rasterio.open(input_path) as image, rasterio.open(
        reference_path
    ) as reference:
        if image.crs is None or reference.crs is None:
            raise ValueError("Both the image and reference need a CRS.")
        if image.count != 2:
            raise ValueError(
                f"Expected two Sentinel-1 VV/VH bands, found {image.count}."
            )
        descriptions = tuple(
            (item or "").strip().upper() for item in image.descriptions
        )
        if descriptions != ("VV", "VH"):
            raise ValueError(
                f"Expected bands described as ('VV', 'VH'), got {descriptions}."
            )
        if os.path.isfile(aligned_path):
            with rasterio.open(aligned_path) as aligned:
                if (
                    aligned.width == image.width
                    and aligned.height == image.height
                    and aligned.crs == image.crs
                    and aligned.transform == image.transform
                ):
                    return aligned_path

        profile = image.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
            compress="DEFLATE",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        temporary_path = aligned_path + ".tmp"
        with rasterio.open(temporary_path, "w", **profile) as aligned:
            reproject(
                source=rasterio.band(reference, 1),
                destination=rasterio.band(aligned, 1),
                src_transform=reference.transform,
                src_crs=reference.crs,
                src_nodata=reference.nodata,
                dst_transform=image.transform,
                dst_crs=image.crs,
                dst_nodata=255,
                resampling=Resampling.nearest,
                init_dest_nodata=True,
            )
            aligned.update_tags(
                source=reference_path,
                resampling="nearest",
                warning=(
                    "NRSC false-colour map-derived proxy, not ground truth."
                ),
            )
        os.replace(temporary_path, aligned_path)
    return aligned_path


class Cachar2022Patches(Dataset):
    def __init__(self, input_path, label_path, coordinates, augment=False):
        self.input_path = input_path
        self.label_path = label_path
        self.coordinates = coordinates
        self.augment = augment
        self.image = rasterio.open(input_path)
        self.labels = rasterio.open(label_path)
        self._labels_cache = self.labels.read(1)

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        row, col = self.coordinates[index]
        window = Window(
            col,
            row,
            min(PATCH_SIZE, self.image.width - col),
            min(PATCH_SIZE, self.image.height - row),
        )
        raw = self.image.read(window=window, out_dtype="float32")
        label = self.labels.read(1, window=window)
        valid = (
            (label != 255)
            & (self.image.read_masks(window=window) > 0).all(axis=0)
            & np.isfinite(raw).all(axis=0)
            & (raw != -9999.0).all(axis=0)
        )
        target = (label == 1).astype(np.float32)
        image = torch.from_numpy(_normalise(raw, valid).copy())
        target_tensor = torch.from_numpy(target).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid).unsqueeze(0)
        pad_h = PATCH_SIZE - image.shape[-2]
        pad_w = PATCH_SIZE - image.shape[-1]
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            image = torch.nn.functional.pad(image, padding)
            target_tensor = torch.nn.functional.pad(
                target_tensor, padding
            )
            valid_tensor = torch.nn.functional.pad(
                valid_tensor, padding, value=False
            )
        if self.augment:
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
                target_tensor = target_tensor.flip(-1)
                valid_tensor = valid_tensor.flip(-1)
            if torch.rand(()) < 0.5:
                image = image.flip(-2)
                target_tensor = target_tensor.flip(-2)
                valid_tensor = valid_tensor.flip(-2)
        return image, target_tensor, valid_tensor.bool()

    def close(self):
        self.image.close()
        self.labels.close()


def _patch_inventory(input_path, label_path):
    coordinates = []
    block_stats = {}
    with rasterio.open(input_path) as image, rasterio.open(
        label_path
    ) as labels:
        label_array = labels.read(1)
        for row in range(0, image.height, PATCH_SIZE):
            for col in range(0, image.width, PATCH_SIZE):
                height = min(PATCH_SIZE, image.height - row)
                width = min(PATCH_SIZE, image.width - col)
                region = label_array[row:row + height, col:col + width]
                valid = region != 255
                if float(valid.mean()) < MIN_VALID_FRACTION:
                    continue
                if not (
                    image.read_masks(
                        1,
                        window=Window(col, row, width, height),
                    ) > 0
                ).mean() >= MIN_VALID_FRACTION:
                    continue
                coordinates.append((row, col))
                block = (
                    row // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
                    col // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
                )
                positive, labeled = block_stats.get(block, (0, 0))
                block_stats[block] = (
                    positive + int((region == 1).sum()),
                    labeled + int(valid.sum()),
                )
    return coordinates, block_stats


def _split_by_spatial_blocks(coordinates, block_stats):
    groups = {}
    for coordinate in coordinates:
        row, col = coordinate
        block = (
            row // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
            col // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
        )
        groups.setdefault(block, []).append(coordinate)
    blocks = list(groups)
    prevalence = {
        block: (
            block_stats[block][0] / block_stats[block][1]
            if block in block_stats and block_stats[block][1]
            else 0.0
        )
        for block in blocks
    }
    positive_rates = [
        prevalence[block] for block in blocks if prevalence[block] > 0
    ]
    cut_points = (
        np.quantile(
            positive_rates,
            np.linspace(0.0, 1.0, SPLIT_STRATA + 1)[1:-1],
        ).tolist()
        if positive_rates
        else []
    )
    strata = {}
    for block in blocks:
        rate = prevalence[block]
        stratum = (
            0
            if rate == 0.0
            else 1 + int(np.digitize(rate, cut_points, right=True))
        )
        strata.setdefault(stratum, []).append(block)

    split_blocks = {"train": set(), "valid": set(), "test": set()}
    rng = random.Random(SEED)
    for stratum_blocks in strata.values():
        rng.shuffle(stratum_blocks)
        count = len(stratum_blocks)
        if count < 3:
            train_count = count - 1
            valid_count = 0
        else:
            train_count = max(1, int(count * 0.70))
            valid_count = max(1, int(count * 0.15))
            if train_count + valid_count >= count:
                valid_count = count - train_count - 1
        split_blocks["train"].update(stratum_blocks[:train_count])
        split_blocks["valid"].update(
            stratum_blocks[train_count:train_count + valid_count]
        )
        split_blocks["test"].update(
            stratum_blocks[train_count + valid_count:]
        )
    splits = {
        name: [
            coordinate
            for coordinate in coordinates
            if (
                coordinate[0] // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
                coordinate[1] // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
            ) in assigned
        ]
        for name, assigned in split_blocks.items()
    }
    for name, items in splits.items():
        assigned_blocks = split_blocks[name]
        positive_count = sum(block_stats.get(block, (0, 0))[0]
                             for block in assigned_blocks)
        labeled_count = sum(block_stats.get(block, (0, 0))[1]
                            for block in assigned_blocks)
        print(
            f"{name}: {len(items)} 128-pixel patches, "
            f"{len(split_blocks[name])} spatial blocks, "
            f"{positive_count:,}/{labeled_count:,} proxy-positive/labeled "
            f"pixels ({positive_count / max(1, labeled_count):.1%})"
        )
    if min(len(splits[name]) for name in ("train", "valid", "test")) == 0:
        raise ValueError(
            "Spatial split is empty; more georeferenced reference coverage "
            "is needed."
        )
    return splits, split_blocks


def _predict(model, loader):
    model.eval()
    probabilities = []
    labels = []
    valid_masks = []
    with torch.inference_mode():
        for images, targets, valid in loader:
            scores = torch.sigmoid(model(images.to(DEVICE))).cpu()
            probabilities.append(scores)
            labels.append(targets)
            valid_masks.append(valid)
    if not probabilities:
        raise ValueError("No patches available to evaluate.")
    return (
        torch.cat(probabilities),
        torch.cat(labels),
        torch.cat(valid_masks),
    )


def _best_threshold(probabilities, labels, valid):
    candidates = np.arange(0.1, 0.91, 0.05)
    return max(
        candidates,
        key=lambda threshold: (
            compute_metrics(
                probabilities,
                labels,
                valid,
                threshold=float(threshold),
            )["f1"],
            -abs(float(threshold) - 0.5),
        ),
    ).item()


def _reported_metrics(probabilities, labels, valid, threshold):
    return compute_metrics(
        probabilities,
        labels,
        valid,
        threshold=threshold,
    )


def _block_bootstrap_intervals(
    probabilities,
    labels,
    valid,
    coordinates,
    threshold,
    iterations=1000,
):
    """Estimate metric intervals by resampling independent spatial blocks."""
    grouped_counts = {}
    probability_array = probabilities[:, 0].numpy()
    label_array = labels[:, 0].numpy() >= 0.5
    valid_array = valid[:, 0].numpy().astype(bool)
    for index, (row, col) in enumerate(coordinates):
        block = (
            row // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
            col // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
        )
        selected = valid_array[index]
        predicted = probability_array[index] >= threshold
        actual = label_array[index]
        counts = np.asarray(
            [
                np.count_nonzero(predicted & actual & selected),
                np.count_nonzero(predicted & ~actual & selected),
                np.count_nonzero(~predicted & actual & selected),
                np.count_nonzero(~predicted & ~actual & selected),
            ],
            dtype=np.int64,
        )
        grouped_counts[block] = grouped_counts.get(
            block, np.zeros(4, dtype=np.int64)
        ) + counts
    block_counts = np.stack(list(grouped_counts.values()))
    if len(block_counts) < 2:
        raise ValueError(
            "At least two test spatial blocks are required for bootstrap."
        )
    generator = np.random.default_rng(SEED + 1)
    sampled = generator.integers(
        0, len(block_counts), size=(iterations, len(block_counts))
    )
    totals = block_counts[sampled].sum(axis=1)
    tp, fp, fn, tn = totals.T
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    metrics = {
        "accuracy": (tp + tn) / np.maximum(tp + fp + fn + tn, 1),
        "f1": f1,
    }
    return {
        name: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for name, values in metrics.items()
    }


def _train_one(name, model, train_loader, valid_loader):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )
    criterion = CombinedLoss()
    checkpoint = os.path.join(OUTPUT_DIR, f"best_{name}.pth")
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for images, labels, valid in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            valid = valid.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        valid_probabilities, valid_labels, valid_mask = _predict(
            model, valid_loader
        )
        metrics = _reported_metrics(
            valid_probabilities,
            valid_labels,
            valid_mask,
            threshold=0.5,
        )
        print(
            f"{name} epoch {epoch}/{MAX_EPOCHS}: "
            f"loss={np.mean(losses):.4f}, "
            f"validation proxy F1@0.50={metrics['f1']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"{name}: early stopping after {PATIENCE} epochs.")
                break
    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    )
    probabilities, labels, valid = _predict(model, valid_loader)
    threshold = _best_threshold(probabilities, labels, valid)
    return model, {
        "best_epoch": best_epoch,
        "threshold": float(threshold),
        "checkpoint": checkpoint,
        "validation_metrics": _reported_metrics(
            probabilities, labels, valid, threshold=float(threshold)
        ),
    }


def _build_models():
    return {
        "cnn": AssamFloodModel(
            in_channels=2, base_channels=BASE_CHANNELS
        ).to(DEVICE),
        "gcn": GCNOnly(patch_size=PATCH_SIZE, cell_size=8).to(DEVICE),
        "cnn_gcn": CNNGCNModel(
            in_channels=2,
            base_channels=BASE_CHANNELS,
            patch_size=PATCH_SIZE,
            cell_size=8,
        ).to(DEVICE),
        "cnn_gcn_transformer": CNNGCNTransformerModel(
            in_channels=2,
            base_channels=BASE_CHANNELS,
            patch_size=PATCH_SIZE,
            cell_size=8,
            hidden_channels=TRANSFORMER_HIDDEN_CHANNELS,
            attention_heads=TRANSFORMER_ATTENTION_HEADS,
            transformer_layers=TRANSFORMER_LAYERS,
        ).to(DEVICE),
    }


def _save_full_predictions(input_path, label_path, models, thresholds):
    prediction_dir = os.path.join(OUTPUT_DIR, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    paths = {
        name: os.path.join(prediction_dir, f"{name}_prediction.tif")
        for name in MODEL_THRESHOLDS
    }
    with rasterio.open(input_path) as image, rasterio.open(
        label_path
    ) as labels:
        profile = image.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
            compress="DEFLATE",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        writers = {
            name: rasterio.open(path, "w", **profile)
            for name, path in paths.items()
        }
        label_array = labels.read(1)
        try:
            windows = [
                Window(
                    col,
                    row,
                    min(PATCH_SIZE, image.width - col),
                    min(PATCH_SIZE, image.height - row),
                )
                for row in range(0, image.height, PATCH_SIZE)
                for col in range(0, image.width, PATCH_SIZE)
            ]
            for start in range(0, len(windows), BATCH_SIZE):
                batch_windows = windows[start:start + BATCH_SIZE]
                batch_tensors = []
                valid_masks = []
                labels_for_windows = []
                for window in batch_windows:
                    row, col = int(window.row_off), int(window.col_off)
                    height, width = int(window.height), int(window.width)
                    raw = image.read(window=window, out_dtype="float32")
                    label = label_array[
                        row:row + height, col:col + width
                    ]
                    valid = (
                        (label != 255)
                        & (image.read_masks(window=window) > 0).all(axis=0)
                        & np.isfinite(raw).all(axis=0)
                        & (raw != -9999.0).all(axis=0)
                    )
                    patch = torch.from_numpy(_normalise(raw, valid).copy())
                    pad_h, pad_w = PATCH_SIZE - height, PATCH_SIZE - width
                    if pad_h or pad_w:
                        patch = torch.nn.functional.pad(
                            patch, (0, pad_w, 0, pad_h)
                        )
                    batch_tensors.append(patch)
                    valid_masks.append(valid)
                    labels_for_windows.append(label)
                batch = torch.stack(batch_tensors).to(DEVICE)
                with torch.inference_mode():
                    probability = {
                        "vh_threshold": 1.0 - batch[:, 1:2],
                        **{
                            name: torch.sigmoid(model(batch))
                            for name, model in models.items()
                        },
                    }
                for name, scores in probability.items():
                    for index, window in enumerate(batch_windows):
                        height, width = int(window.height), int(window.width)
                        prediction = (
                            scores[index, 0, :height, :width].cpu().numpy()
                            >= thresholds[name]
                        ).astype(np.uint8)
                        label = labels_for_windows[index]
                        valid = valid_masks[index]
                        prediction[~valid] = 255
                        writers[name].write(prediction, 1, window=window)
        finally:
            for writer in writers.values():
                writer.close()
    return paths


def _save_reference_overlay(input_path, label_path, predictions):
    figure, axes = plt.subplots(2, 5, figsize=(23, 10))
    with rasterio.open(input_path) as image:
        shape = (700, 700)
        vv = image.read(
            1, out_shape=shape, resampling=Resampling.average
        )
        vh = image.read(
            2, out_shape=shape, resampling=Resampling.average
        )
        imagery_valid = (
            image.dataset_mask(
                out_shape=shape, resampling=Resampling.nearest
            )
            > 0
        )
    axes[0, 0].imshow(
        np.ma.masked_where(~imagery_valid, vv),
        cmap="gray",
        vmin=-25,
        vmax=0,
    )
    axes[0, 0].set_title("2022 Sentinel-1 VV")
    axes[0, 1].imshow(
        np.ma.masked_where(~imagery_valid, vh),
        cmap="gray",
        vmin=-35,
        vmax=-5,
    )
    axes[0, 1].set_title("2022 Sentinel-1 VH")
    with rasterio.open(label_path) as reference:
        labels = reference.read(
            1, out_shape=shape, resampling=Resampling.nearest
        )
    axes[0, 2].imshow(
        np.ma.masked_where(labels == 255, labels),
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[0, 2].set_title("Provisional NRSC map-derived proxy")
    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.03,
        0.92,
        "Study area: Cachar district, Assam\n"
        "Input: 10–20 July 2022 Sentinel-1 median\n"
        "Reference: NRSC false-colour map proxy\n"
        "Metrics: spatial holdout agreement only\n"
        "Not field-validated flood accuracy",
        transform=axes[0, 3].transAxes,
        va="top",
        fontsize=12,
        wrap=True,
    )
    axes[0, 4].axis("off")
    for axis, method in zip(axes[1], MODEL_THRESHOLDS):
        with rasterio.open(predictions[method]) as prediction:
            values = prediction.read(
                1, out_shape=shape, resampling=Resampling.nearest
            )
        axis.imshow(
            np.ma.masked_where(values == 255, values),
            cmap="Blues",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.set_title(MODEL_LABELS[method])
    figure.suptitle(
        "Cachar 2022 models vs NRSC-derived proxy — not ground truth",
        fontsize=14,
    )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    path = os.path.join(OUTPUT_DIR, "assam_2022_model_maps.png")
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if not os.path.isfile(REFERENCE_PATH):
        raise FileNotFoundError(
            "NRSC map-derived proxy missing. Run "
            "prepare_nrsc_2022_reference.py first."
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_path = _input_path()
    label_path = os.path.join(OUTPUT_DIR, "reference_on_sentinel_grid.tif")
    _align_reference(input_path, REFERENCE_PATH, label_path)
    coordinates, block_stats = _patch_inventory(input_path, label_path)
    if len(coordinates) < MIN_EPOCH_PATCHES:
        raise ValueError(
            f"Only {len(coordinates)} labeled patches found; "
            f"at least {MIN_EPOCH_PATCHES} are required for training."
        )
    splits, split_blocks = _split_by_spatial_blocks(
        coordinates, block_stats
    )
    if len(split_blocks["test"]) < 4 or len(split_blocks["valid"]) < 4:
        raise ValueError(
            "The spatial holdout has too few independent map blocks. "
            "Use a larger reference area or a smaller spatial block size."
        )
    datasets = {
        name: Cachar2022Patches(
            input_path,
            label_path,
            items,
            augment=(name == "train"),
        )
        for name, items in splits.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(name == "train"),
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
        )
        for name, dataset in datasets.items()
    }
    print(f"Training device: {DEVICE}")
    models = _build_models()
    reports = {}
    thresholds = {
        "vh_threshold": 0.60,
    }
    for name, model in models.items():
        print(f"\nTraining 2022 Cachar {name} model...")
        trained_model, training_report = _train_one(
            name, model, loaders["train"], loaders["valid"]
        )
        models[name] = trained_model
        reports[name] = training_report
        thresholds[name] = training_report["threshold"]

    baseline_valid, valid_labels, valid_mask = _predict_baseline(
        loaders["valid"]
    )
    thresholds["vh_threshold"] = float(
        _best_threshold(baseline_valid, valid_labels, valid_mask)
    )

    test_probabilities = {}
    test_labels = None
    test_valid = None
    for name, model in models.items():
        probability, labels, valid = _predict(model, loaders["test"])
        test_probabilities[name] = probability
        test_labels = labels
        test_valid = valid
    baseline_test, test_labels, test_valid = _predict_baseline(loaders["test"])
    test_probabilities["vh_threshold"] = baseline_test

    results = {}
    for name in MODEL_THRESHOLDS:
        intervals = _block_bootstrap_intervals(
            test_probabilities[name],
            test_labels,
            test_valid,
            splits["test"],
            thresholds[name],
        )
        results[name] = {
            "threshold": thresholds[name],
            "test_metrics": _reported_metrics(
                test_probabilities[name],
                test_labels,
                test_valid,
                threshold=thresholds[name],
            ),
            "spatial_block_bootstrap_95ci": intervals,
        }
        metrics = results[name]["test_metrics"]
        results[name]["prediction_collapsed_to_all_positive"] = (
            metrics["TN"] == 0 and metrics["FN"] == 0
        )
        if name in reports:
            results[name]["validation_metrics"] = reports[name][
                "validation_metrics"
            ]
            results[name]["best_epoch"] = reports[name]["best_epoch"]

    paths = _save_full_predictions(
        input_path, label_path, models, thresholds
    )
    report = {
        "study": "2022 Cachar flood-event mapping within Assam, India",
        "input": input_path,
        "input_sensor": "Copernicus Sentinel-1 VV/VH",
        "input_acquisition_window": (
            "2022-07-10 through 2022-07-20, median composite"
        ),
        "reference": {
            "source": "NRSC/ISRO Cachar flood map issued 2022-07-17",
            "pre_flood_image": "IRS Resourcesat-2 AWiFS acquired 2022-03-02",
            "post_flood_image": "IRS Resourcesat-2 AWiFS acquired 2022-07-14",
            "label_method": (
                "Digitized post-panel cyan pixels minus pre-panel cyan pixels "
                "from the false-colour map PDF, then nearest-neighbor aligned "
                "to Sentinel-1."
            ),
            "quality": (
                "Provisional map-derived proxy; not an official vector "
                "flood mask, field survey, or independent ground truth."
            ),
        },
        "spatial_split": {
            "seed": SEED,
            "split_unit": (
                f"{PATCH_SIZE * SPATIAL_BLOCK_TILES} x "
                f"{PATCH_SIZE * SPATIAL_BLOCK_TILES} Sentinel-1 pixels "
                "(non-overlapping spatial blocks)"
            ),
            "patch_counts": {
                name: len(items) for name, items in splits.items()
            },
            "block_counts": {
                name: len(items) for name, items in split_blocks.items()
            },
            "proxy_positive_fraction_by_split": {
                name: round(
                    sum(block_stats.get(block, (0, 0))[0]
                        for block in items)
                    / max(
                        1,
                        sum(block_stats.get(block, (0, 0))[1]
                            for block in items),
                    ),
                    4,
                )
                for name, items in split_blocks.items()
            },
            "test_scope": "held-out spatial blocks from Cachar NRSC map area",
        },
        "model_settings": {
            "patch_size": PATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "transformer": {
                "hidden_channels": TRANSFORMER_HIDDEN_CHANNELS,
                "attention_heads": TRANSFORMER_ATTENTION_HEADS,
                "layers": TRANSFORMER_LAYERS,
                "tokens_per_patch": (PATCH_SIZE // 8) ** 2,
            },
            "device": str(DEVICE),
            "classes": "0=map non-cyan, 1=post-only cyan, 255=unknown",
        },
        "threshold_source": (
            "Selected on held-out validation spatial blocks; test blocks "
            "were not used for threshold selection."
        ),
        "uncertainty_estimation": {
            "method": (
                "Percentile bootstrap resampling of held-out spatial blocks "
                "with replacement; 95% intervals for accuracy and F1."
            ),
            "iterations": 1000,
            "test_spatial_blocks": len(split_blocks["test"]),
            "interpretation": (
                "Intervals quantify spatial-block sampling variability for "
                "agreement with this proxy; they do not correct proxy-label "
                "errors or represent field-validated flood accuracy."
            ),
        },
        "results": results,
        "prediction_paths": paths,
        "metrics_interpretation": (
            "These are spatial holdout agreement scores against a provisional "
            "false-colour NRSC-map-derived proxy. They are not field-validated "
            "flood accuracy and may reflect PDF color-digitization errors."
        ),
    }
    report_path = os.path.join(
        OUTPUT_DIR, "assam_2022_spatial_holdout_metrics.json"
    )
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    plot_path = os.path.join(
        OUTPUT_DIR, "assam_2022_proxy_agreement_comparison.png"
    )
    save_assam_metric_comparison(
        results,
        plot_path,
        (
            "2022 Cachar | NRSC false-colour map-derived proxy | "
            "spatial holdout with 95% block-bootstrap intervals | "
            "NOT ground-truth accuracy"
        ),
        title="2022 Cachar spatial-holdout model comparison",
    )
    map_path = _save_reference_overlay(input_path, label_path, paths)
    print(f"\nMetrics report: {report_path}")
    print(f"Proxy agreement comparison: {plot_path}")
    print(f"Model maps: {map_path}")
    for name in MODEL_THRESHOLDS:
        print(
            f"{MODEL_LABELS[name]}: "
            f"accuracy={results[name]['test_metrics']['accuracy']:.3f}, "
            f"F1={results[name]['test_metrics']['f1']:.3f}, "
            f"F1 95% CI="
            f"{results[name]['spatial_block_bootstrap_95ci']['f1'][0]:.3f}-"
            f"{results[name]['spatial_block_bootstrap_95ci']['f1'][1]:.3f}"
        )
        if results[name]["prediction_collapsed_to_all_positive"]:
            print(
                f"WARNING: {MODEL_LABELS[name]} predicted every valid "
                "test pixel as proxy-positive."
            )
    print(
        "IMPORTANT: These figures measure agreement with the provisional "
        "NRSC map-derived proxy, not field-validated flood accuracy."
    )
    for dataset in datasets.values():
        dataset.close()


def _predict_baseline(loader):
    probabilities = []
    labels = []
    valid_masks = []
    with torch.inference_mode():
        for images, targets, valid in loader:
            probabilities.append(1.0 - images[:, 1:2])
            labels.append(targets)
            valid_masks.append(valid)
    return torch.cat(probabilities), torch.cat(labels), torch.cat(valid_masks)


if __name__ == "__main__":
    main()
