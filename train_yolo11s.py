"""
End-to-end training and inference pipeline for Oil Storage Tank Detection using YOLO11s (Small) and SAHI.
Supports both Kaggle GPU environments and local execution.
"""

import os
import glob
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

# ==========================================
# 1. Reproducibility Configuration
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. Dataset Paths Setup
# ==========================================
kaggle_path = "/kaggle/input/datasets/organizations/airbusgeo/airbus-oil-storage-detection-dataset"
local_path = "./data"

if os.path.exists(kaggle_path):
    base_path = kaggle_path
elif os.path.exists(local_path):
    base_path = local_path
else:
    base_path = kaggle_path

print(f"Dataset root: {base_path}")
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "annotations.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Annotations not found at {csv_path}. Please check your dataset path.")

df = pd.read_csv(csv_path)
print(f"Annotations loaded: {len(df)} records across {df['image_id'].nunique()} unique images.")

# ==========================================
# 3. Annotation Helpers & Coordinate Conversion
# ==========================================
def parse_bounds(bounds_str):
    bounds_str = bounds_str.strip("()")
    x_min, y_min, x_max, y_max = map(int, bounds_str.split(","))
    return [x_min, y_min, x_max, y_max]

def convert_to_yolo(size, bbox):
    w_img, h_img = size
    x_center = (bbox[0] + bbox[2]) / 2.0 / w_img
    y_center = (bbox[1] + bbox[3]) / 2.0 / h_img
    w = (bbox[2] - bbox[0]) / float(w_img)
    h = (bbox[3] - bbox[1]) / float(h_img)
    return [x_center, y_center, w, h]

# Deterministic Train / Validation Split
image_ids = sorted(df['image_id'].unique().tolist())
random.shuffle(image_ids)
split = int(0.8 * len(image_ids))
train_ids = set(image_ids[:split])
val_ids = set(image_ids[split:])
print(f"Train images: {len(train_ids)} | Validation images: {len(val_ids)}")

# ==========================================
# 4. Tiling Engine with Boundary Protection & Background Inclusion
# ==========================================
TILE_SIZE = 640
STRIDE = 512
MIN_BOX_SIZE = 10
BG_TILE_RATIO = 0.10  # 10% negative background tiles

os.makedirs("tiled/images/train", exist_ok=True)
os.makedirs("tiled/images/val", exist_ok=True)
os.makedirs("tiled/labels/train", exist_ok=True)
os.makedirs("tiled/labels/val", exist_ok=True)

def get_steps(dimension, tile_size, stride):
    """Pin the last step to (dimension - tile_size) so boundary pixels are never dropped."""
    if dimension <= tile_size:
        return [0]
    steps = list(range(0, dimension - tile_size + 1, stride))
    if steps[-1] != dimension - tile_size:
        steps.append(dimension - tile_size)
    return steps

def tile_image_and_boxes_clean(img, boxes, bg_ratio=0.10):
    h, w, _ = img.shape
    tiles = []
    x_steps = get_steps(w, TILE_SIZE, STRIDE)
    y_steps = get_steps(h, TILE_SIZE, STRIDE)

    for y in y_steps:
        for x in x_steps:
            tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
            tile_boxes = []

            for box in boxes:
                x_min, y_min, x_max, y_max = box
                if x_max <= x or x_min >= x + TILE_SIZE:
                    continue
                if y_max <= y or y_min >= y + TILE_SIZE:
                    continue

                new_xmin = max(x_min - x, 0)
                new_ymin = max(y_min - y, 0)
                new_xmax = min(x_max - x, TILE_SIZE)
                new_ymax = min(y_max - y, TILE_SIZE)

                box_w = new_xmax - new_xmin
                box_h = new_ymax - new_ymin
                if box_w < MIN_BOX_SIZE or box_h < MIN_BOX_SIZE:
                    continue

                tile_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

            if len(tile_boxes) > 0:
                tiles.append((tile, tile_boxes))
            elif bg_ratio > 0 and random.random() < bg_ratio:
                tiles.append((tile, []))

    return tiles

# Tile dataset generation
print("\n--- Generating Tiled Dataset ---")
tile_count = 0
tank_tile_count = 0
bg_tile_count = 0

for img_id in tqdm(image_ids, desc="Tiling Images"):
    img_path = os.path.join(image_dir, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue

    rows = df[df['image_id'] == img_id]
    boxes = [parse_bounds(r['bounds']) for _, r in rows.iterrows()]

    subset = "train" if img_id in train_ids else "val"
    ratio = BG_TILE_RATIO if subset == "train" else 0.05
    tiles = tile_image_and_boxes_clean(img, boxes, bg_ratio=ratio)

    for i, (tile, tile_boxes) in enumerate(tiles):
        h, w, _ = tile.shape
        img_name = f"{img_id}_{i}.jpg"
        label_name = f"{img_id}_{i}.txt"

        img_out = f"tiled/images/{subset}/{img_name}"
        label_out = f"tiled/labels/{subset}/{label_name}"

        cv2.imwrite(img_out, tile)
        with open(label_out, "w") as f:
            for box in tile_boxes:
                yolo_box = convert_to_yolo((w, h), box)
                f.write(f"0 {' '.join(map(str, yolo_box))}\n")

        tile_count += 1
        if len(tile_boxes) > 0:
            tank_tile_count += 1
        else:
            bg_tile_count += 1

print(f"Total tiles: {tile_count} | Tanks: {tank_tile_count} | Backgrounds: {bg_tile_count} ({bg_tile_count/max(1, tile_count):.1%})")

# Write dataset yaml
yaml_content = """path: tiled
train: images/train
val: images/val

names:
  0: oil_tank
"""
with open("tiled_dataset.yaml", "w") as f:
    f.write(yaml_content)

# ==========================================
# 5. Model Training: YOLO11-Small (yolo11s)
# ==========================================
print("\n--- Initializing YOLO11s Training ---")
model = YOLO("yolo11s.pt")

train_results = model.train(
    data="tiled_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,            # Optimized for Tesla T4 GPU
    patience=35,         # Gives enough room past plateau
    cos_lr=True,         # Smooth cosine decay
    degrees=90,          # Nadir satellite rotational invariance
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=10,     # Turn off mosaic last 10 epochs for sharp boundaries
    plots=True,          # Save all training curves
    name="yolo11s_oil_tank"
)

# Grab the exact directory where weights & results were saved
run_dir = str(model.trainer.save_dir)
print(f"\nTraining completed! Results saved to: {run_dir}")

# Force generation of PR curve, F1 curve, and Confusion Matrix on best weights
print("\n--- Running Final Validation to Generate All Evaluation Plots ---")
best_model_path = os.path.join(run_dir, "weights/best.pt")
best_model = YOLO(best_model_path)
val_metrics = best_model.val(data="tiled_dataset.yaml", plots=True)

# ==========================================
# 6. Evaluation Graphs & Model Performance Summary
# ==========================================
print("\n--- Generating Evaluation Graphs & Performance Summary ---")
csv_file = os.path.join(run_dir, "results.csv")
if os.path.exists(csv_file):
    res_df = pd.read_csv(csv_file)
    res_df.columns = [c.strip() for c in res_df.columns]

    best_row_idx = res_df['metrics/mAP50-95(B)'].idxmax()
    best_row = res_df.loc[best_row_idx]
    p = best_row['metrics/precision(B)']
    r = best_row['metrics/recall(B)']
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    print("\n" + "=" * 65)
    print("🎯 YOLO11s MODEL PERFORMANCE SUMMARY (BEST EPOCH)")
    print("=" * 65)
    print(f"Best Epoch:           {int(best_row['epoch'])} / {len(res_df)}")
    print(f"Precision (P):        {p:.4f} ({p*100:.2f}%)")
    print(f"Recall (R):           {r:.4f} ({r*100:.2f}%)")
    print(f"F1-Score:             {f1:.4f} ({f1*100:.2f}%)")
    print(f"mAP@50:               {best_row['metrics/mAP50(B)']:.4f} ({best_row['metrics/mAP50(B)']*100:.2f}%)")
    print(f"mAP@50-95:            {best_row['metrics/mAP50-95(B)']:.4f} ({best_row['metrics/mAP50-95(B)']*100:.2f}%)")
    print(f"Final Val Box Loss:   {best_row['val/box_loss']:.4f}")
    print(f"Final Val Cls Loss:   {best_row['val/cls_loss']:.4f}")
    print("=" * 65)

    # 4-Panel Training Results Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = res_df['epoch']
    best_ep = int(best_row['epoch'])

    axes[0, 0].plot(epochs, res_df['train/box_loss'], label='Train Box Loss', color='#1f77b4', lw=2)
    axes[0, 0].plot(epochs, res_df['val/box_loss'], label='Val Box Loss', color='#ff7f0e', lw=2, linestyle='--')
    axes[0, 0].axvline(best_ep, color='red', linestyle=':', label=f'Best Epoch ({best_ep})')
    axes[0, 0].set_title('Bounding Box Regression Loss (CIoU)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, res_df['train/cls_loss'], label='Train Cls Loss', color='#2ca02c', lw=2)
    axes[0, 1].plot(epochs, res_df['val/cls_loss'], label='Val Cls Loss', color='#d62728', lw=2, linestyle='--')
    axes[0, 1].axvline(best_ep, color='red', linestyle=':', label=f'Best Epoch ({best_ep})')
    axes[0, 1].set_title('Classification Loss (BCE)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, res_df['train/dfl_loss'], label='Train DFL Loss', color='#9467bd', lw=2)
    axes[1, 0].plot(epochs, res_df['val/dfl_loss'], label='Val DFL Loss', color='#8c564b', lw=2, linestyle='--')
    axes[1, 0].axvline(best_ep, color='red', linestyle=':', label=f'Best Epoch ({best_ep})')
    axes[1, 0].set_title('Distribution Focal Loss (DFL)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, res_df['metrics/mAP50(B)'], label='mAP@50', color='#17becf', lw=2)
    axes[1, 1].plot(epochs, res_df['metrics/mAP50-95(B)'], label='mAP@50-95', color='#e377c2', lw=2.5)
    axes[1, 1].axvline(best_ep, color='red', linestyle=':', label=f'Best Epoch ({best_ep})')
    axes[1, 1].set_title('Validation Accuracy (mAP)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle('YOLO11s Training Results & Convergence Curves', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    curve_out = os.path.join(run_dir, "yolo11s_learning_curves.png")
    plt.savefig(curve_out, dpi=200)
    plt.close()
    print(f"Saved custom learning curves to: {curve_out}")

# ==========================================
# 7. Full-Image Sliced Inference with SAHI
# ==========================================
print("\n--- Running SAHI Full-Image Sliced Inference ---")
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=best_model_path,
        confidence_threshold=0.5,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    sample_val_id = next(iter(val_ids))
    sample_image_path = os.path.join(image_dir, f"{sample_val_id}.jpg")

    if os.path.exists(sample_image_path):
        sahi_result = get_sliced_prediction(
            sample_image_path,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25,
            postprocess_type="NMS",
            postprocess_match_threshold=0.3
        )
        os.makedirs("runs/detect/sahi_output", exist_ok=True)
        sahi_result.export_visuals(export_dir="runs/detect/sahi_output")

        raw_results = best_model.predict(source=sample_image_path, conf=0.5, imgsz=640, verbose=False)
        raw_count = len(raw_results[0].boxes)
        sahi_count = len(sahi_result.object_prediction_list)

        print("\n" + "="*55)
        print("🛰️ SAHI BENCHMARK COMPARISON (YOLO11s)")
        print(f"Scene: {os.path.basename(sample_image_path)}")
        print(f"Standard Unsliced (Downscaled to 640px): {raw_count} tanks detected")
        print(f"SAHI Sliced Inference (Tiled + Global NMS): {sahi_count} tanks detected")
        print(f"Small Object Gain: +{max(0, sahi_count - raw_count)} tanks ({(sahi_count/max(1, raw_count)-1.0)*100:+.1f}%)")
        print("="*55)
except ImportError:
    print("SAHI is not installed. Run `pip install sahi` for full-image sliced inference.")

print("\nPipeline execution complete!")

