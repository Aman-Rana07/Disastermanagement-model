import unittest

import torch

from src.assam_methods import (
    CNNGCNModel,
    CNNGCNTransformerModel,
    GCNOnly,
    SpatialPatchGCN,
    build_patch_adjacency,
)


class AssamGraphModelTests(unittest.TestCase):
    def test_patch_adjacency_is_symmetric_with_self_loops(self):
        adjacency = build_patch_adjacency(4)
        self.assertTrue(torch.allclose(adjacency, adjacency.T))
        self.assertTrue(torch.all(adjacency.diagonal() > 0))
        self.assertEqual(adjacency.shape, (16, 16))

    def test_gcn_and_combined_models_return_trainable_pixel_logits(self):
        torch.manual_seed(3)
        images = torch.rand(2, 2, 128, 128)
        for model in (
            GCNOnly(),
            CNNGCNModel(base_channels=2),
            CNNGCNTransformerModel(
                base_channels=2,
                hidden_channels=16,
                transformer_layers=1,
            ),
        ):
            logits = model(images)
            self.assertEqual(logits.shape, (2, 1, 128, 128))
            self.assertTrue(torch.isfinite(logits).all())
            logits.mean().backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.parameters())
            )

    def test_transformer_rejects_incompatible_attention_width(self):
        with self.assertRaisesRegex(
            ValueError, "divisible by attention_heads"
        ):
            CNNGCNTransformerModel(hidden_channels=30, attention_heads=4)

    def test_graph_rejects_unaligned_input_size(self):
        model = SpatialPatchGCN(in_channels=2)
        with self.assertRaisesRegex(ValueError, "align to cell_size"):
            model(torch.rand(1, 2, 127, 128))


if __name__ == "__main__":
    unittest.main()
