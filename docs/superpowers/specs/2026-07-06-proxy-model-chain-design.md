# Proxy Model Chain Design

Date: 2026-07-06

## Objective

Build a minimum runnable classification pipeline for weeks 1-2. The pipeline will use two proxy label sources in parallel:

1. `Subtype_Classification_Full.tif` plus `Subtype_Confidence_Full.tif` as the primary high-confidence subtype pseudo-label task.
2. FVC/NDVI-derived classes as an auxiliary proxy task for checking whether the model and downstream analysis chain can run when labels come from continuous vegetation condition data.

The goal is to validate the engineering chain: sample construction, feature alignment, model training, internal validation, model comparison, classification output, and later transition to real grassland subtype reference data.

This pipeline must not claim final grassland subtype accuracy. It is an engineering and method-prototype stage.

## Scope

### Included

- Config-driven local data paths.
- Raster-based sample extraction.
- High-confidence pseudo-label filtering.
- Optional FVC/NDVI proxy label generation from existing NDVI/FVC rasters.
- Traditional baseline models: Random Forest and sklearn gradient boosting.
- A small neural baseline: MLP.
- A minimal spatiotemporal placeholder model interface that can later be replaced by U-Net + Transformer.
- Internal train/validation split with a spatial block option.
- Confusion matrix, OA, PA, UA, F1, macro-F1, model comparison table.
- Outputs under `results/tables` and `results/figures_small`.

### Excluded For This Stage

- Full Sentinel-2 download and preprocessing.
- Full U-Net + Transformer production training.
- Final independent field validation.
- Formal paper-level accuracy claims.
- Large-area batch inference over all periods.

## Data Inputs

### Primary Proxy Label

- `D:\草地遥感\result\汇报\260513\output\Subtype_Classification_Full.tif`
- `D:\草地遥感\result\汇报\260513\output\Subtype_Confidence_Full.tif`

Initial rule:

- Use classes `0-4`.
- Keep pixels where confidence is at least `0.95`.
- If a class has too few samples, allow a documented fallback threshold of `0.90`.
- Sample approximately balanced counts per class.

### Auxiliary FVC/NDVI Proxy Label

Candidate source directories:

- `D:\草地遥感\result\tiff\*_ndvi.tif`
- `D:\草地遥感\result\output\一、NDVI源数据\innermonglia源数据\*_ndvi_clip.tif`
- `D:\草地遥感\result\output\一、NDVI源数据\monglia源数据\*_ndvi_clip.tif`

Initial rule:

- Compute a simple vegetation-condition class from available NDVI/FVC values.
- Use quantile classes or fixed thresholds, controlled in config.
- Treat these labels as auxiliary proxy labels only.

### Feature Inputs

Minimum feature set:

- DEM from `D:\草地遥感\result\汇报\260513\dem\Mongolian_Plateau_DEM.tif`.
- Existing NDVI/FVC rasters if aligned or resampled.
- Confidence raster as a sample filtering layer, not as a model feature by default.

Environment factor extension:

- ERA5-Land variables from the existing download script.
- SoilGrids variables from the existing download script.
- DEM-derived slope, aspect, and TWI.

## Architecture

### Component 1: Configuration

File: `configs/proxy_model.yaml`

Responsibilities:

- Store data paths.
- Define proxy label mode: `subtype`, `fvc_ndvi`, or `both`.
- Define sampling thresholds and sample counts.
- Define model list.
- Define output paths.
- Define split mode: random or spatial block.

### Component 2: Dataset Builder

File: `src/classification/build_proxy_dataset.py`

Responsibilities:

- Load label rasters and feature rasters.
- Align feature rasters to the label grid where needed.
- Apply nodata and confidence masks.
- Build balanced samples.
- Save sample table as `.csv` or `.npz`.
- Write a sample summary table.

Outputs:

- `results/tables/proxy_samples_subtype.csv`
- `results/tables/proxy_samples_fvc_ndvi.csv`
- `results/tables/proxy_sample_summary.csv`

### Component 3: Traditional Baselines

File: `src/classification/train_baselines.py`

Models:

- Random Forest.
- HistGradientBoostingClassifier as the default gradient boosting model, because it is available in scikit-learn and avoids adding an XGBoost dependency.

Responsibilities:

- Train models on the same split.
- Save predictions and metrics.
- Save feature importance where available.

Outputs:

- `results/tables/model_comparison.csv`
- `results/tables/feature_importance.csv`

### Component 4: Neural Prototype

File: `src/classification/train_spatiotemporal.py`

Initial implementation:

- MLP over tabular raster features.
- A minimal sequence-model interface if time-series features are available.

Purpose:

- Keep the API compatible with the later U-Net + Transformer model.
- Validate training loop, metrics, checkpointing, and output structure before full model implementation.

Outputs:

- `results/tables/neural_model_metrics.csv`
- `results/models/proxy_mlp_model.*`

### Component 5: Evaluation

File: `src/classification/evaluate.py`

Responsibilities:

- Compute confusion matrix.
- Compute OA, PA, UA, F1, macro-F1.
- Save per-class metrics.
- Save model comparison tables.

Outputs:

- `results/tables/confusion_matrix_<model>.csv`
- `results/tables/class_metrics_<model>.csv`
- `results/tables/model_comparison.csv`

## Data Flow

1. Read config.
2. Build subtype pseudo-label samples.
3. Build FVC/NDVI proxy samples if configured.
4. Generate train/validation split.
5. Train traditional models.
6. Train neural prototype.
7. Evaluate all models on the same validation samples.
8. Write metrics tables and figures.
9. Record limitations in the run summary.

## Validation Rules

- Report proxy validation as internal validation only.
- Do not use random split alone as final evidence.
- Prefer spatial block split when enough spatial coordinates are available.
- Keep FVC/NDVI proxy results separate from subtype pseudo-label results.
- Every metrics table must record the label source.

## Success Criteria For Weeks 1-2

Week 1 is successful if:

- The dataset builder creates balanced samples for the subtype pseudo-label task.
- The FVC/NDVI proxy task can be generated or is explicitly skipped with a documented reason.
- At least one traditional model trains and produces a confusion matrix.

Week 2 is successful if:

- Traditional baselines and a neural prototype can run on the same sample split.
- Model comparison tables are generated.
- The outputs clearly distinguish subtype pseudo-label and FVC/NDVI proxy tasks.
- The code can be rerun from config without editing script internals.

## Risks And Controls

| Risk | Control |
|---|---|
| Proxy labels overstate accuracy | Label all outputs as proxy/internal validation |
| Feature rasters are misaligned | Add raster shape/CRS/transform checks before sampling |
| Class imbalance | Use stratified balanced sampling and record actual counts |
| Missing optional features | Allow minimum feature set to run without ERA5/SoilGrids |
| XGBoost unavailable | Use sklearn HistGradientBoostingClassifier as default |
| Deep model too heavy | Start with MLP and compatible interface |

## Implementation Order

1. Add config file.
2. Add shared raster and metrics utilities.
3. Implement subtype pseudo-label dataset builder.
4. Add optional FVC/NDVI proxy builder.
5. Implement RF and gradient boosting baselines.
6. Implement MLP/neural prototype.
7. Implement evaluation and output tables.
8. Run a small smoke test.
9. Commit the runnable prototype.

## Review Notes

This design intentionally keeps the first implementation small. It preserves the final U-Net + Transformer direction by using compatible data and output interfaces, but it does not pretend that the proxy task is the final grassland subtype validation.
