"""Regenerate and open the saved Assam 2022 comparison figures."""

import argparse
import json
import webbrowser
from pathlib import Path

from src.assam_plotting import save_assam_metric_comparison


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "data" / "exports" / "assam_2022"
TRAINED_DIR = OUTPUT_DIR / "trained"
REPORT_PATH = TRAINED_DIR / "assam_2022_spatial_holdout_metrics.json"
CHART_PATH = TRAINED_DIR / "assam_2022_proxy_agreement_comparison.png"
MODEL_MAP_PATH = TRAINED_DIR / "assam_2022_model_maps.png"
REFERENCE_PREVIEW_PATH = (
    OUTPUT_DIR / "reference" / "nrsc_cachar_reference_review.png"
)

METHOD_LABELS = {
    "vh_threshold": "VH threshold",
    "cnn": "CNN",
    "gcn": "GCN",
    "cnn_gcn": "CNN + GCN",
    "cnn_gcn_transformer": "CNN + GCN + Transformer",
}


def _read_report():
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Metrics report not found: {REPORT_PATH}\n"
            "Run train_assam_2022.py first."
        )
    with REPORT_PATH.open(encoding="utf-8") as report_file:
        report = json.load(report_file)

    results = report.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"Invalid or missing results in {REPORT_PATH}.")
    missing = set(METHOD_LABELS) - results.keys()
    if missing:
        raise ValueError(
            "The report does not contain every expected 2022 method: "
            + ", ".join(sorted(missing))
        )
    for name, result in results.items():
        metrics = result.get("test_metrics", {})
        for metric in ("accuracy", "f1", "precision", "recall"):
            if metric not in metrics:
                raise ValueError(
                    f"Missing {metric!r} test metric for {name!r}."
                )
        if "iou" in metrics:
            raise ValueError(
                f"The report for {name!r} includes IoU; "
                "regenerate it using the current 2022 workflow."
            )
    return report


def _print_results(report):
    split = report.get("spatial_split", {})
    blocks = split.get("block_counts", {}).get("test", "unknown")
    print("2022 Cachar results - agreement with provisional NRSC map proxy")
    print(f"Held-out spatial test blocks: {blocks}")
    print(f"{'Method':<29} {'Accuracy':>10} {'F1':>10} {'Precision':>11} {'Recall':>10}")
    for name, label in METHOD_LABELS.items():
        result = report["results"][name]
        metrics = result["test_metrics"]
        interval = result.get("spatial_block_bootstrap_95ci", {}).get("f1")
        f1_text = f"{metrics['f1']:.3f}"
        if interval and len(interval) == 2:
            f1_text += f" ({interval[0]:.3f}-{interval[1]:.3f})"
        print(
            f"{label:<29} {metrics['accuracy']:>9.1%} "
            f"{f1_text:>10} {metrics['precision']:>10.1%} "
            f"{metrics['recall']:>9.1%}"
        )
    print(
        "\nCaution: these are agreement scores against a provisional "
        "map-derived proxy, not field-validated flood accuracy."
    )


def _regenerate_chart(report):
    subtitle = (
        "2022 Cachar | NRSC false-colour map-derived proxy | "
        "spatial holdout with 95% block-bootstrap intervals | "
        "NOT ground-truth accuracy"
    )
    return save_assam_metric_comparison(
        report["results"],
        str(CHART_PATH),
        subtitle,
        title="2022 Cachar spatial-holdout model comparison",
    )


def _open_image(path):
    if not path.is_file():
        print(f"Image not found; skipping: {path}")
        return
    print(f"Opening: {path}")
    webbrowser.open(path.resolve().as_uri())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print the saved 2022 model results, regenerate the comparison "
            "chart, and open the charts/maps."
        )
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Regenerate and print outputs without opening image windows.",
    )
    args = parser.parse_args()

    report = _read_report()
    _print_results(report)
    chart = _regenerate_chart(report)
    print(f"\nComparison chart: {chart}")
    print(f"Model maps: {MODEL_MAP_PATH}")
    print(f"Reference review: {REFERENCE_PREVIEW_PATH}")

    if not args.no_open:
        for image_path in (
            CHART_PATH,
            MODEL_MAP_PATH,
            REFERENCE_PREVIEW_PATH,
        ):
            _open_image(image_path)


if __name__ == "__main__":
    main()
