# Assam 2022 flood-event data

The active experiment covers the July 2022 Cachar flood map area only.
GeoTIFFs are tracked with Git LFS; install Git LFS before cloning this
repository.

## Sentinel-1 model input

`sentinel1/assam_cachar_s1_vv_vh_20220710_20220720.tif` contains the
two-band VV/VH median composite used by the model. It was exported from
Google Earth Engine for the Cachar NRSC map extent using
`export_sentinel1_july2022.js`. The input was losslessly recompressed with
Zstandard; its pixel values, grid, band names, and metadata were verified
against the original.

## NRSC/ISRO reference

The checked-in `data/exports/assam_2022/reference/nrsc_cachar_flood_proxy.tif`
is a provisional cyan-change proxy derived from the NRSC/ISRO Cachar flood
map issued 17 July 2022. The map panels use a pre-flood AWiFS image from
2 March 2022 and a post-flood image from 14 July 2022. It is not an official
binary inundation raster, a field survey, or independent ground truth.

The source PDF reports are not included. To rebuild the proxy, obtain the
official Cachar map from NRSC/ISRO and save it as
`nrsc_reference/cachar_2022.pdf`, then run
`python prepare_nrsc_2022_reference.py` from the repository root. Review the
generated preview before evaluation: map colors, river water, labels,
graphics, registration, and acquisition-date differences can all introduce
label errors.

## Optional 2022 demo product

The threshold-derived 2022 flood-extent raster and polygons in this
directory are reference/demo material from a separate mapping workflow.
They are not independent ground truth and are not used to train or score the
Cachar models.

## Results

The latest saved test report compares a VH threshold, CNN, GCN, CNN+GCN, and
CNN+GCN+Transformer using geographically held-out blocks. The Transformer
scored 87.9% accuracy and 0.798 F1 (95% spatial-block interval 0.732–0.841)
against the proxy. CNN+GCN scored 87.6% accuracy and 0.789 F1 (0.722–0.833).
The GCN baseline had 27.2% accuracy and 99.9% recall because its predictions
were nearly all positive. The Transformer interval overlaps CNN+GCN's; these
proxy-agreement results are not validated flood-accuracy scores.

See the repository [README](../../README.md) for setup, run commands, full
metrics, saved graphs, maps, and prediction links.
