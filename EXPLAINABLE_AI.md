# Explainable AI

## Overview

This project explores three approaches to understanding model behaviour:

1. SHAP
2. LIME
3. Occlusion Sensitivity

These techniques provide different perspectives on model predictions. Their outputs should not be interpreted as clinical evidence.

## 1. SHAP

SHAP stands for SHapley Additive exPlanations.

It uses Shapley-value-based feature attribution to estimate how individual features contribute to model predictions.

In this project, SHAP was explored using the Random Forest model.

Exploratory results from 20 test samples:

| Feature Group | Average Per-Feature Importance |
|---|---:|
| HOG | 0.000106609 |
| GLCM | 0.002350635 |

GLCM has higher average importance per feature in this exploratory analysis.

However, HOG has 6,084 features, whereas GLCM has only 12. Per-feature importance and aggregate group importance are different measurements.

These results are exploratory and based on a limited sample.

## 2. LIME

LIME stands for Local Interpretable Model-agnostic Explanations.

It approximates model behaviour around an individual prediction by perturbing the input and observing prediction changes.

For image explanations, LIME can use superpixels to identify regions that influence a prediction.

The highlighted areas represent model-influential regions. They are not tumor segmentation masks.

## 3. Occlusion Sensitivity

Occlusion Sensitivity examines how a model's prediction changes when image regions are covered or replaced.

Implementation configuration:

- Patch size: 32 × 32
- Stride: 16
- Overlapping patch scores are averaged.

The resulting heatmap represents changes in model output caused by occluding image regions.

It helps investigate which regions influence predictions.

It does not identify tumor boundaries or establish that the model is focusing on medically meaningful tissue.

## 4. Important Limitations

- Explanations are dependent on the trained model.
- Different explanation methods may produce different results.
- Feature attribution does not establish causality.
- Heatmaps are not medical segmentation.
- These methods do not establish clinical reliability.
