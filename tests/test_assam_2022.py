import unittest

import numpy as np
import torch

from prepare_nrsc_2022_reference import _water_and_validity
from train_assam_2022 import (
    PATCH_SIZE,
    SPATIAL_BLOCK_TILES,
    _block_bootstrap_intervals,
    _normalise,
    _reported_metrics,
    _split_by_spatial_blocks,
)


class Assam2022WorkflowTests(unittest.TestCase):
    def test_nrsc_color_rule_separates_cyan_red_and_white(self):
        pixels = np.array(
            [[[50, 150, 160], [180, 60, 50], [255, 255, 255]]],
            dtype=np.uint8,
        )
        water_like, valid = _water_and_validity(pixels)
        np.testing.assert_array_equal(water_like, [[True, False, False]])
        np.testing.assert_array_equal(valid, [[True, True, False]])

    def test_normalization_masks_invalid_pixels(self):
        raw = np.array(
            [
                [[-10.0, np.nan]],
                [[-20.0, -15.0]],
            ],
            dtype=np.float32,
        )
        valid = np.array([[True, False]])
        normalized = _normalise(raw, valid)
        self.assertEqual(normalized.shape, (2, 1, 2))
        self.assertTrue(np.all(normalized[:, 0, 0] > 0))
        np.testing.assert_array_equal(normalized[:, 0, 1], [0.0, 0.0])

    def test_spatial_split_never_separates_tiles_in_same_block(self):
        spacing = PATCH_SIZE
        coordinates = [
            (row * spacing, col * spacing)
            for row in range(12)
            for col in range(12)
        ]
        block_positive_pixels = {}
        splits, _ = _split_by_spatial_blocks(
            coordinates, block_positive_pixels
        )
        split_for_coordinate = {
            coordinate: split
            for split, items in splits.items()
            for coordinate in items
        }
        owners = {}
        for row, col in coordinates:
            block = (
                row // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
                col // (PATCH_SIZE * SPATIAL_BLOCK_TILES),
            )
            owner = split_for_coordinate[(row, col)]
            if block in owners:
                self.assertEqual(owners[block], owner)
            owners[block] = owner
        self.assertEqual(set(split_for_coordinate), set(coordinates))

    def test_block_bootstrap_returns_spatial_metric_intervals(self):
        probabilities = torch.tensor(
            [[[[0.9, 0.1], [0.9, 0.1]]], [[[0.9, 0.9], [0.9, 0.9]]]]
        )
        labels = torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]
        )
        valid = torch.ones_like(labels, dtype=torch.bool)
        intervals = _block_bootstrap_intervals(
            probabilities,
            labels,
            valid,
            [(0, 0), (PATCH_SIZE * SPATIAL_BLOCK_TILES, 0)],
            threshold=0.5,
            iterations=50,
        )
        self.assertEqual(set(intervals), {"accuracy", "f1"})
        for lower, upper in intervals.values():
            self.assertGreaterEqual(lower, 0.0)
            self.assertLessEqual(upper, 1.0)
            self.assertLessEqual(lower, upper)

    def test_reported_2022_metrics_omit_iou(self):
        probabilities = torch.tensor([[[[0.9, 0.1]]]])
        labels = torch.tensor([[[[1.0, 0.0]]]])
        valid = torch.ones_like(labels, dtype=torch.bool)
        metrics = _reported_metrics(
            probabilities, labels, valid, threshold=0.5
        )
        self.assertEqual(metrics["f1"], 1.0)
        self.assertNotIn("iou", metrics)


if __name__ == "__main__":
    unittest.main()
