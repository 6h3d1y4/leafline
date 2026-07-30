![leafline logo](https://github.com/6h3d1y4/leafline/blob/main/CoverImage/leafline_logo_inverted.png)

# Leafline: Tree-Cover Mapping for Kiel

## Repository Link

[https://github.com/6h3d1y4/leafline]

## Description

Leafline is a geospatial AI project for mapping urban tree cover in Kiel, Germany, using high-resolution RGBI orthomosaic imagery and deep-learning-based instance segmentation.

The project seeks to assess the effectiveness of using deep-learning to identify tree crowns in Kiel city. 
The city currently has extensive manually collected data on trees on city-owned grounds. This leaves out data about trees on any private ground, leaving a large data-gap when trying to assess green spaces in the city. 
High-resolution aerial images are usually collected by plane once a year and theoretically present a great chance to fill this data-gap. Filling this gap would allow the city to better track changes in tree numbers and tree cover, allowing for comprehensive analysis of city climate, shade corridors, green islands, and greenery development. 
Identifying individual trees combined with high-resolution aerial data further allows for tree species identification, as well as individual tree crown area, height, and health assessments.

Using pre-trained tree-segmentation models however comes with major challenges and uncertainties: 
1. Most models are trained on summer images, while the city mostly collects images in spring. 
2. Since publicly available data, which models are generally trained on, is lower-resolution, it is unclear what impact the higher resolution images have on model performance.
3. Height data is shown to benefit crown-segmentation, but is rarely included in models.

This project aims to adress these uncertainties, by finetuning an existing tree-segmentation model and testing it on different data sets. We also expand the model to include height data. We chose an existing tree segmentation model, since the task is quite complex and there are already a lot of models available. 
Our findings also give insight into a possible workflow to adjust the model to other geographical areas, since it is more efficient and requires less data than to train a model from scratch. This is very helpful for further development regarding tree sementation, because data availability in particular is one of the main challenges in aerial image segmentation.

### Task Type

Image Classification - Instance Segmentation

### Results Summary

We fine-tune the pretrained DeepTrees / `freudenberg2022` tree-crown model on Kiel
imagery. Evaluation is **crown-level (instance) F1** via greedy IoU matching. Because an
IoU≥0.5 cutoff is unusually strict for small tree crowns, we report both **IoU 0.5** and
the more common **IoU 0.3**. Postprocessing is tuned per resolution (`pp_sweep.py`).

#### Best Model Performance
- **Best Model (7.5 cm, Kiel's native spring resolution):** `step1_ndom` — DeepTrees
  fine-tuned on 100 % spring 7.5 cm, 6 channels (RGBI + NDVI + nDOM/height).
- **Best Model (20 cm):** `step3_mix20` — fine-tuned on a 50/50 summer+spring 20 cm mix.
- **Evaluation Metric:** crown-level F1 (IoU-matched instance segmentation).
- **Final Performance:** 7.5 cm F1 = **0.158** (IoU 0.5) / **0.38** (IoU 0.3);
  20 cm-spring F1 = **0.144** / 0.32; 20 cm-summer F1 = **0.317**.

#### Model Comparison
- **Baseline Performance (un-fine-tuned):** 7.5 cm F1 = **0.000** (out-of-distribution —
  the pretrained model detects almost nothing at Kiel's native resolution),
  20 cm-summer = 0.340, 20 cm-spring = 0.044.
- **Improvement Over Baseline:** 7.5 cm **0.000 → 0.158** (fine-tuning makes the native
  resolution usable at all); 20 cm-spring **0.044 → 0.144** (~3.3×).
- **Channel ablation @7.5 cm:** RGBI 0.123 < +NDVI 0.150 < +nDOM 0.158 — every channel
  helps, almost entirely via precision.

#### Key Insights
- **Most Important Features:** RGBI + NDVI carry the core signal (NDVI helps even in
  spring); the nDOM height channel adds a small precision gain (separates trees from
  green ground). Resolution and season are each a large domain factor.
- **Model Strengths:** works at Kiel's native 7.5 cm spring imagery where the baseline
  fails entirely; a single 20 cm model handles both seasons (separate season models are
  not needed) and even improves spring over a spring-only model.
- **Model Limitations:** recall is the ceiling (~0.12 at IoU 0.5, ~0.29 at IoU 0.3) —
  ~40 % of crowns are not detected and ~48 % are localized but fail IoU 0.5 due to a mix
  of over-segmentation and merging of adjacent crowns. Postprocessing, learning-rate
  tuning and extra channels do not move recall; it is a data/annotation-density limit.
- **Practical Impact:** the workflow shows a pretrained crown model can be adapted to a
  new city/resolution with little data, filling the gap of un-surveyed (private-land)
  trees — but reliable per-crown delineation at 7.5 cm still needs more/denser training
  annotations. Learned recipe: **one model per resolution, shared across seasons.**

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/exploratory_data_analysis.ipynb)**
3. **[Baseline Model](2_BaselineModel/baseline_model.ipynb)**
4. **[Model Definition and Evaluation](3_Model/model_definition_evaluation.ipynb)**
   — results overview: [results_all_steps.ipynb](3_Model/results_all_steps.ipynb);
   training/eval how-to and status: [3_Model/README.md](3_Model/README.md),
   [TRAINING_STATUS.md](3_Model/TRAINING_STATUS.md)
5. **[Presentation](4_Presentation/README.md)**

## Cover Image

