"""Extract a reviewable flood-water proxy from NRSC's georeferenced Cachar PDF."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
import rasterio
from affine import Affine
from pypdf import PdfReader
from rasterio.enums import Resampling
from rasterio.warp import reproject


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(
    ROOT_DIR,
    "data",
    "public_assam_2022",
    "nrsc_reference",
    "cachar_2022.pdf",
)
OUTPUT_DIR = os.path.join(
    ROOT_DIR, "data", "exports", "assam_2022", "reference"
)
RENDER_SCALE = 4
NODATA = 255


def _geo_panels(pdf_path):
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    page_height = float(page.mediabox.height)
    panels = []
    for viewport_ref in page.get("/VP", []):
        viewport = viewport_ref.get_object()
        measure_ref = viewport.get("/Measure")
        if measure_ref is None:
            continue
        measure = measure_ref.get_object()
        points = np.asarray(measure.get("/GPTS"), dtype=np.float64).reshape(-1, 2)
        latitudes = points[:, 0]
        longitudes = points[:, 1]
        west, south = float(longitudes.min()), float(latitudes.min())
        east, north = float(longitudes.max()), float(latitudes.max())
        if not (90 <= west < east <= 95 and 23 <= south < north <= 27):
            continue
        x0, y0, x1, y1 = map(float, viewport.get("/BBox"))
        panels.append(
            {
                "bbox": pymupdf.Rect(
                    x0, page_height - y0, x1, page_height - y1
                ),
                "bounds": (west, south, east, north),
            }
        )
    panels.sort(key=lambda panel: panel["bbox"].x0)
    if len(panels) != 2:
        raise ValueError(
            f"Expected exactly two georeferenced Cachar panels in {pdf_path}; "
            f"found {len(panels)}."
        )
    return panels[0], panels[1]


def _render_panel(page, panel):
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE),
        clip=panel["bbox"],
        alpha=False,
    )
    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[:, :, :3].copy()
    west, south, east, north = panel["bounds"]
    transform = Affine(
        (east - west) / pixmap.width,
        0,
        west,
        0,
        -(north - south) / pixmap.height,
        north,
    )
    return rgb, transform, panel["bounds"]


def _water_and_validity(rgb):
    values = rgb.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    cyan = (
        (green - red >= 25)
        & (blue - red >= 25)
        & (green >= 55)
        & (blue >= 55)
    )
    near_white = (
        (red >= 238)
        & (green >= 238)
        & (blue >= 238)
        & ((np.maximum.reduce((red, green, blue)) -
            np.minimum.reduce((red, green, blue))) <= 24)
    )
    outside = np.all(values >= 245, axis=2) | np.all(
        values <= 12, axis=2
    )
    return cyan, ~outside & ~near_white


def _write_geotiff(path, data, transform, bounds, tags):
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=NODATA,
        compress="DEFLATE",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as output:
        output.write(data, 1)
        output.update_tags(
            **tags,
            geographic_bounds=",".join(f"{value:.8f}" for value in bounds),
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create an explicitly provisional map-derived reference raster "
            "from the official NRSC Cachar pre/post-flood map."
        )
    )
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    if not os.path.isfile(args.pdf):
        raise FileNotFoundError(f"NRSC map PDF not found: {args.pdf}")
    os.makedirs(args.output_dir, exist_ok=True)

    pre_panel, post_panel = _geo_panels(args.pdf)
    document = pymupdf.open(args.pdf)
    pre_rgb, pre_transform, _ = _render_panel(
        document[0], pre_panel
    )
    post_rgb, post_transform, post_bounds = _render_panel(
        document[0], post_panel
    )
    pre_water, pre_valid = _water_and_validity(pre_rgb)
    post_water, post_valid = _water_and_validity(post_rgb)

    pre_water_on_post = np.zeros(post_water.shape, dtype=np.uint8)
    pre_valid_on_post = np.zeros(post_water.shape, dtype=np.uint8)
    reproject(
        source=pre_water.astype(np.uint8),
        destination=pre_water_on_post,
        src_transform=pre_transform,
        src_crs="EPSG:4326",
        dst_transform=post_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
    )
    reproject(
        source=pre_valid.astype(np.uint8),
        destination=pre_valid_on_post,
        src_transform=pre_transform,
        src_crs="EPSG:4326",
        dst_transform=post_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
    )

    valid = post_valid & (pre_valid_on_post > 0)
    flood_proxy = post_water & (pre_water_on_post == 0) & valid
    reference = np.full(post_water.shape, NODATA, dtype=np.uint8)
    reference[valid] = 0
    reference[flood_proxy] = 1

    mask_path = os.path.join(
        args.output_dir, "nrsc_cachar_flood_proxy.tif"
    )
    _write_geotiff(
        mask_path,
        reference,
        post_transform,
        post_bounds,
        {
            "source": "NRSC/ISRO Cachar flood map, issued 2022-07-17",
            "pre_flood_panel": "IRS Resourcesat-2 AWiFS, 2022-03-02",
            "post_flood_panel": "IRS Resourcesat-2 AWiFS, 2022-07-14",
            "classes": "0=map-interpreted non-cyan; 1=post-only cyan; 255=unknown",
            "method": (
                "Color-rule digitization of post-panel cyan water-like pixels "
                "minus pre-panel cyan pixels; inspect preview before scoring."
            ),
            "warning": (
                "Provisional visual proxy, not an official vector mask or "
                "field-validated ground truth."
            ),
        },
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 7))
    axes[0].imshow(pre_rgb)
    axes[0].set_title("NRSC pre-flood AWiFS (2 Mar 2022)")
    axes[1].imshow(post_rgb)
    axes[1].set_title("NRSC post-flood AWiFS (14 Jul 2022)")
    display_mask = np.ma.masked_where(reference == NODATA, reference)
    axes[2].imshow(
        display_mask,
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[2].set_title("Provisional new-water proxy (review required)")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(
        "NRSC Cachar map-derived flood reference proxy — not ground truth",
        fontsize=14,
    )
    figure.tight_layout()
    preview_path = os.path.join(
        args.output_dir, "nrsc_cachar_reference_review.png"
    )
    figure.savefig(preview_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    known_pixels = int(valid.sum())
    proxy_pixels = int(flood_proxy.sum())
    print(f"Reference proxy GeoTIFF: {mask_path}")
    print(f"Review image: {preview_path}")
    print(f"Valid reference pixels: {known_pixels}")
    print(f"Proxy flood pixels: {proxy_pixels}")
    print(
        "Proxy fraction of valid map pixels: "
        f"{100.0 * proxy_pixels / known_pixels:.2f}%"
    )
    print(
        "Do not call resulting scores ground-truth accuracy; the proxy is "
        "digitized from false-colour map imagery."
    )


if __name__ == "__main__":
    main()
