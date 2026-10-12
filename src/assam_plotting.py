"""Matplotlib plots for the held-out Assam-method comparison."""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("vh_threshold", "VH threshold"),
    ("cnn", "CNN (U-Net)"),
    ("gcn", "GCN"),
    ("cnn_gcn", "CNN + GCN"),
    ("cnn_gcn_transformer", "CNN + GCN + Transformer"),
)
METRICS = (
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
    ("precision", "Precision"),
    ("recall", "Recall"),
)


def save_assam_metric_comparison(
    results,
    output_path,
    subtitle,
    title="Held-out 2022 Cachar model comparison",
):
    """Save held-out test accuracy and complementary metrics as a bar chart."""
    scores = {}
    for method, _ in METHODS:
        if method not in results or "test_metrics" not in results[method]:
            raise ValueError(f"Missing test metrics for method: {method}")
        scores[method] = []
        for metric, _ in METRICS:
            value = results[method]["test_metrics"].get(metric)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"Invalid {metric} test score for {method}: {value!r}"
                )
            scores[method].append(float(value))

    metric_positions = np.arange(len(METRICS))
    bar_width = 0.15
    figure, axis = plt.subplots(figsize=(12, 6.5))
    for method_index, (method, label) in enumerate(METHODS):
        positions = metric_positions + (
            method_index - (len(METHODS) - 1) / 2
        ) * bar_width
        error_bars = []
        for metric, _ in METRICS:
            bounds = results[method].get(
                "spatial_block_bootstrap_95ci", {}
            ).get(metric)
            score = results[method]["test_metrics"][metric]
            if bounds is None:
                error_bars.append((0.0, 0.0))
            else:
                error_bars.append(
                    (max(0.0, score - bounds[0]), max(0.0, bounds[1] - score))
                )
        yerr = np.asarray(error_bars).T
        bars = axis.bar(
            positions,
            scores[method],
            bar_width,
            label=label,
            yerr=yerr if np.any(yerr) else None,
            capsize=2,
        )
        axis.bar_label(
            bars,
            labels=[f"{score:.1%}" for score in scores[method]],
            padding=2,
            fontsize=8,
            rotation=90,
        )

    axis.set_xticks(
        metric_positions,
        [label for _, label in METRICS],
    )
    axis.set_ylim(0, 1.18)
    axis.set_ylabel("Score (%)")
    figure.suptitle(
        title,
        fontsize=15,
        y=0.99,
    )
    figure.text(
        0.5,
        0.92,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        ncols=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path
