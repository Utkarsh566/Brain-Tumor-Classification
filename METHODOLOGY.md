# Project Methodology

## 1. Overview

The project uses handcrafted Computer Vision features and traditional Machine Learning to classify MRI images into four categories.

The pipeline:

1. Dataset preparation
2. Exact duplicate removal
3. Stratified image-level splitting
4. Grayscale conversion
5. Image resizing
6. HOG extraction
7. GLCM extraction
8. Feature concatenation
9. StandardScaler transformation
10. SVM and Random Forest training
11. Evaluation and explainability

## 2. Dataset Preparation

- Unique images: 6,726
- Training images: 5,380
- Testing images: 1,346
- Split: 80:20
- Random state: 42
- Exact duplicates removed: 79

SHA-256 hashing was used to identify exact duplicate images across the original Training and Testing folders.

The resulting dataset was divided using stratified image-level splitting.

This was not a patient-level split. Patient-level generalization has not been established.

## 3. Preprocessing

Images were converted to grayscale and resized to 224 × 224 pixels.

Consistent dimensions are required for the configured HOG extraction pipeline.

StandardScaler was fitted using training features only. The fitted scaler was subsequently used to transform training and testing features.

## 4. HOG Feature Extraction

Histogram of Oriented Gradients captures local gradient and edge orientation information.

Configuration:

```python
hog(
    img,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    feature_vector=True
)
```

Features per image: 6,084.

## 5. GLCM Feature Extraction

Gray-Level Co-occurrence Matrix captures spatial relationships between grayscale intensity values.

Configuration:

```python
graycomatrix(
    img,
    distances=[1, 2],
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
    levels=256,
    symmetric=True,
    normed=True
)
```

Properties:

- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation
- ASM

The mean and standard deviation of each property were calculated across the distance-angle combinations.

Features per image: 12.

## 6. Feature Combination

| Feature Type | Count |
|---|---:|
| HOG | 6,084 |
| GLCM | 12 |
| Total | 6,096 |

Both feature groups were concatenated into a single feature vector.

## 7. Classification

Two models were evaluated:

- RBF-kernel Support Vector Machine
- Random Forest

SVM was selected as the primary model because it achieved higher test accuracy and lower training time in this experiment.

## 8. Inference

The deployed application:

1. Accepts an MRI image.
2. Converts it to grayscale.
3. Resizes it to 224 × 224.
4. Extracts the same 6,096 features.
5. Applies the saved StandardScaler.
6. Predicts using the saved SVM model.
7. Displays the predicted class and normalized decision scores.

The deployed application retrieves the model artifact from Hugging Face Hub.
