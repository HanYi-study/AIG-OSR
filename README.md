# AIG-OSR: observation-guided underwater elevation reconstruction

This repository contains a compact reference implementation of the eight-equation AIG-OSR method, a synthetic input pair, and an XY-aligned elevation evaluator. The method reconstructs a **single-valued elevation field** from sparse multibeam echo sounder (MBES) controls and a coextensive side-scan sonar (SSS) image.

**Reproducibility scope.** The synthetic run verifies the code path and file formats. It is not a reproduction of the paper's field results. Historical result folders were produced without complete per-run code, configuration and data records, so their reported table values cannot be attributed to this reference entry point one by one. Survey inputs and full-density reference data are not included here.

## Method

1. Filter and rasterize MBES points into metric elevation controls. Build an interpolated baseline.
2. Extract six SSS-derived conditioning features and an object-boundary mask. A FiLM-conditioned three-layer SIREN predicts a normalized elevation residual relative to the baseline.
3. Fit the residual at MBES cells with mask-dependent baseline and gradient penalties. The optional intensity–slope diagnostic is computed without a parameter-gradient path; it is **not** an effective training loss.
4. Locally filter the preliminary elevation map, then project its correction through a boundary-weighted eight-neighbour graph while fixing MBES control elevations.
5. Report only cells within the selected raster distance of an occupied MBES control.

The mathematical definitions and symbol descriptions are in [Methods: eight equations](docs/METHODS_EIGHT_EQUATIONS.md). The [equation-to-code map](docs/EQUATION_CODE_MAP.md) identifies the implementation for each equation. The [publication audit](docs/PUBLICATION_AUDIT.md) separates formula alignment from historical-result provenance. AIG-OSR produces a 2.5D height field `z=f(x,y)`; it does not independently recover vertical walls, overhangs, or a closed 3D solid.

## Inputs and outputs

| File | Role |
|---|---|
| `sparse.las` | MBES `x,y,z` controls in a local/projected metric coordinate frame. |
| `registered_sss.png` | Coextensive SSS image. The current code maps the MBES bounding box linearly to the image extent; alignment must be supplied. |
| `DEM_Final_SelectiveDenoise.xyz` | Final metric `x,y,z` DEM within the observation-support mask. |
| `MBES_Control_Raster.xyz` | Rasterized sparse input controls; **not** independent full ground truth. |
| `Input_Denoised_MBES.xyz` | Filtered MBES input points. |
| `run_manifest.json` | Parameters, input hashes, code hash and library versions for a run. |

The XYZ format contains no CRS or vertical-datum metadata. Record the survey coordinate and height references separately when using real data.

## Install and run the synthetic sample

Use a Python environment compatible with `requirements.txt`. The reference code has been checked with Python 3.8, NumPy 1.24.3, SciPy 1.10.1, OpenCV 4.13.0, laspy 2.5.4 and PyTorch 2.4.1.

```bash
python -m pip install -r requirements.txt
python reproducibility/generate_sample_data.py
python reproducibility/prepare_method_sample.py
python reproducibility/aig_osr_reference.py \
  --las sample_data/reef_sparse_10pct.las \
  --sonar sample_data/reef_sss_synthetic.png \
  --out sample_results/aig_osr_smoke \
  --grid_step 0.1 --steps 10 --seed 7
python reproducibility/evaluate_aligned_metrics.py \
  --pred sample_results/aig_osr_smoke/DEM_Final_SelectiveDenoise.xyz \
  --gt sample_data/reef_gt.xyz \
  --xy-tolerance 0.1 --inlier-threshold 0.10 \
  --out sample_results/aig_osr_smoke/metrics_xy_aligned.json
```

The sample contains 102 retained points from a 1,024-point synthetic surface (9.96%). Its SSS image is generated from a constructed intensity pattern and is not a calibrated acoustic observation. Ten optimization steps are a quick execution check; the resulting metric values must not be cited as paper performance.

## Optional height-field mesh

Install `requirements-visualization.txt` and triangulate a metric DEM as follows:

```bash
python -m pip install -r requirements-visualization.txt
python visualization/heightfield_mesh.py \
  --xyz sample_results/aig_osr_smoke/DEM_Final_SelectiveDenoise.xyz \
  --out sample_results/aig_osr_smoke/reef_demo_mesh.ply \
  --kind reef --grid_stride 1
```

The mesh script uses **regular-grid height-field triangulation**: it divides each supported four-corner DEM cell into two triangles. It applies limited local smoothing to display vertices; bridge visualization uses capped median spike reduction followed by a masked bilateral filter. The resulting PLY is an open 2.5D surface. The paper's DEM metrics must use the original XYZ output, not the smoothed mesh. For comparisons across retention rates, pass the same `--color_min` and `--color_max` for all meshes of a target.

For a real run, replace `--las` and `--sonar` with aligned survey inputs, choose a documented grid spacing and support radius, and use enough training steps for convergence. The reference data supplied to `--gt` are used **only after prediction**. The evaluator matches each reference point to the nearest prediction in XY within the chosen tolerance and then computes elevation errors; report coverage alongside RMSE, MAE and inlier ratio.

## Repository layout

```text
README.md
requirements.txt
docs/
  METHODS_EIGHT_EQUATIONS.md
  EQUATION_CODE_MAP.md
  PUBLICATION_AUDIT.md
reproducibility/
  aig_osr_reference.py
  generate_sample_data.py
  prepare_method_sample.py
  evaluate_aligned_metrics.py
visualization/
  heightfield_mesh.py
sample_data/
  reef_gt.xyz
  reef_sparse_10pct.xyz
  reef_sparse_10pct.las
  reef_sss_synthetic.png
  metadata.json
sample_results/
  aig_osr_smoke/             synthetic execution check only
requirements-visualization.txt
```

## Limitations and provenance

- The reference implementation corresponds to the eight-equation method description. Its numerical solver checks convergence before exporting a DEM.
- Historical field-result tables have not been reproduced by this release. Their input files, run manifests, independent reference data and evaluation scripts must be matched before making a reproducibility claim.
- The main output is a metric elevation surface. Any mesh generated by regular-grid height-field triangulation is a derived visualization of that surface; unsupported cells remain open.
- The SSS–MBES registration and vertical datum are inherited from the inputs. The code does not estimate a residual transformation.

The synthetic sample is generated by `generate_sample_data.py` and `prepare_method_sample.py`; it contains no original survey points or imagery.
