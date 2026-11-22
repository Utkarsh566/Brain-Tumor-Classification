# Model Evaluation

## 1. Dataset Split

| Split | Images |
|---|---:|
| Training | 5,380 |
| Testing | 1,346 |
| Total | 6,726 |

The split was stratified and performed at image level using random state 42.

## 2. Model Comparison

| Model | Test Accuracy | Training Time |
|---|---:|---:|
| SVM (RBF) | 93.24% | 93.09 seconds |
| Random Forest | 91.75% | 121.11 seconds |

SVM achieved the higher test accuracy and shorter training time in this experiment.

## 3. SVM Classification Report

```text
              precision    recall   f1-score   support

      glioma       0.98      0.89      0.93       324
  meningioma       0.83      0.91      0.87       327
     notumor       0.98      0.95      0.96       347
   pituitary       0.95      0.98      0.97       348

    accuracy                           0.93      1346
   macro avg       0.93      0.93      0.93      1346
weighted avg       0.94      0.93      0.93      1346
```

## 4. Metric Definitions

- Precision: Fraction of predictions for a class that were correct.
- Recall: Fraction of actual samples of a class that were correctly identified.
- F1-score: Harmonic mean of precision and recall.
- Support: Number of actual test samples in a class.

## 5. Observations

Meningioma has the lowest precision at 0.83.

Glioma recall is 0.89, indicating that some actual glioma images were classified as other categories.

These results show why per-class evaluation is important instead of relying exclusively on overall accuracy.

## 6. Limitations

- Image-level rather than patient-level split.
- No external dataset validation.
- No clinical validation.
- No tumor segmentation or grading.
- Potential dataset-specific patterns and image acquisition differences.

The reported results describe this experiment only. They do not establish clinical performance or patient-level generalization.
