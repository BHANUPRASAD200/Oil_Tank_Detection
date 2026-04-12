# Satellite Oil Storage Tank Detection

A professional machine learning pipeline for detecting oil storage tanks in high-resolution satellite imagery using YOLOv8. This project implements a full end-to-end workflow including data preprocessing, intelligent tiling, model training, and performance evaluation.

## Workflow Overview

The project follows a modular pipeline designed for high-resolution geospatial data:

1.  **Data Ingestion**: Processes the Airbus Oil Storage Detection dataset, consisting of high-resolution JPEG images and CSV annotations.
2.  **Preprocessing & Tiling**: 
    - **Tiling Strategy**: Satellite images are often too large (several thousand pixels) to be fed directly into a CNN without significant detail loss. We implement a tiling strategy using a **640x640 window** with a **512-pixel stride** (ensuring overlap to avoid cutting objects at edges).
    - **Filtering**: Only tiles containing oil tanks are retained for training to maintain a balanced dataset and reduce computational overhead.
    - **Normalization**: Conversion of bounding box coordinates from pixel-absolute to YOLO-normalized format.
3.  **Model Training**:
    - **Architecture**: YOLOv8s (Small) — selected for its optimal balance between inference speed and detection accuracy.
    - **Hyperparameters**: 100 epochs, image size 640, AdamW optimizer, and mosaic augmentation.
    - **Augmentation**: Applied 90-degree rotations, horizontal/vertical flips, and HSV adjustments to increase model robustness against varying satellite orientations and lighting conditions.

## 📁 Project Structure

```text
oil_tank_project/
├── oil-storage-detection.ipynb  # Primary exploration and training notebook
├── tiled_dataset.yaml           # YOLO configuration for the processed data
├── tiled/                       # Generated dataset (images & labels)
│   ├── images/ (train/val)
│   └── labels/ (train/val)
└── runs/detect/train/           # Training artifacts and metrics
    ├── weights/ (best.pt, last.pt)
    ├── results.csv              # Raw metrics per epoch
    ├── results.png              # Learning curves
    └── confusion_matrix.png     # Error analysis
```

## Model Performance

The model achieved high precision and recall, demonstrating robust detection capabilities across the validation set.

| Metric | Value |
| :--- | :--- |
| **mAP@50** | **0.964** |
| **mAP@50-95** | **0.641** |
| **Precision** | **0.917** |
| **Recall** | **0.903** |

### Training Results
![Training Results](runs/detect/train/results.png)
*Figure 1: Training and Validation Loss/Metric curves over 100 epochs.*

### Error Analysis & Precision-Recall
![Confusion Matrix](runs/detect/train/confusion_matrix_normalized.png)
*Figure 2: Normalized Confusion Matrix showing high accuracy for the 'oil_tank' class.*

![PR Curve](runs/detect/train/BoxPR_curve.png)
*Figure 3: Precision-Recall Curve demonstrating the trade-off at various thresholds.*

## Inference (How to Use the Model)

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference
results = model.predict(
    source="path_to_image_or_folder",
    conf=0.6,
    iou=0.3,
    save=True
)
```

**Output**:
- Bounding boxes with confidence scores.
- Saved results in `runs/detect/predict/`.

## Tiling Strategy Impact

| Feature | Before Tiling | After Tiling |
| :--- | :--- | :--- |
| **Object Size** | Objects are extremely small | Objects become larger and clearer |
| **Detection Quality** | Poor detection of small tanks | Improved recall and detection accuracy |
| **Reliability** | High false negatives | Better performance in dense regions |

![Comparison of Tiling Strategy](compare.png)
*Figure 4: Tiling impact comparison — Left: Original high-resolution image; Right: After tiling and labeling.*


> [!IMPORTANT]
> **Key Insight**: Satellite imagery presents a small object detection challenge due to high resolution and large spatial coverage. 
> 
> **Solution**:
> - Applied tiling (640×640 with overlap)
> - Improved feature visibility
> - Enabled robust detection of small objects

### Final Impact
This project demonstrates a production-ready approach to solving small object detection in satellite imagery using tiling, transfer learning, and optimized inference strategies.



## Assumptions & Challenges

### Assumptions
- **Static Background**: High-resolution imagery is assumed to be orthorectified.
- **Single Class**: The current implementation focuses exclusively on `oil_storage_tank`, treating other industrial structures as background.

### Challenges
- **Pixel Scale**: Small tanks appear as tiny pixel clusters in global views. Tiling was **mandatory** rather than optional to preserve the feature resolution (GSD).
- **Class Imbalance**: Large swathes of satellite images are empty. The pipeline filters these to ensure the gradient updates are meaningful.

## Potential Improvements

- **SAHI Integration**: Implement *Slicing Aided Hyper Inference* (SAHI) during deployment to run inference on overlapping tiles and merge results, which typically boosts small object detection.
- **Model Scaling**: Upgrade to YOLOv8m or YOLOv8l for higher feature extraction depth if computational budget allows.
- **Multi-Spectral Data**: Incorporate Near-Infrared (NIR) bands if available, as oil tanks often have distinct thermal/reflective signatures.
- **Temporal Analysis**: Apply the model across time-series data to track volume changes or industrial activity levels.
