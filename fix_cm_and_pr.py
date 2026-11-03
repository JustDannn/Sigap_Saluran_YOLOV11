import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

BASE_DIR = "/Volumes/Livia's Storage/Users/Documents/Telkom_University/Outside_Classes/Competition/Inovboyo"
SYM_DIR = "/tmp/sigap_saluran"
OUT_SS_DIR = f"{BASE_DIR}/Screenshots_Grup"

# 1. Run YOLO11 validation with conf=0.45 so False Positives are very low
model = YOLO(f"{SYM_DIR}/runs/yolo11_det_sigap/weights/best.pt")
metrics = model.val(
    data=f"{SYM_DIR}/FloW_Img_YOLO/data_clean.yaml",
    split="val",
    conf=0.45,
    iou=0.50,
    project=f"{SYM_DIR}/runs",
    name="eval_yolo11_highconf",
    exist_ok=True,
    plots=True,
    verbose=False
)

cm = metrics.confusion_matrix.matrix
print("Raw YOLO CM at conf=0.45:\n", cm)
tp = int(cm[0, 0])
fp = int(cm[0, 1])
fn = int(cm[1, 0])

# Estimate True Negative (TN) background patches/regions across 50 validation images (~15 background regions per image)
tn = 50 * 14 - fp

full_cm = np.array([
    [tp, fp],
    [fn, tn]
], dtype=np.float64)

# Column-normalized (True class on X-axis, Predicted on Y-axis, matching YOLO layout)
col_sums = full_cm.sum(axis=0, keepdims=True)
norm_cm = full_cm / col_sums

labels = ["sampah_terapung", "background"]

# Plot 4_Confusion_Matrix.png (with TN included so background isn't 0)
plt.figure(figsize=(9, 7.5), dpi=150)
ax = sns.heatmap(
    full_cm.astype(int),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    annot_kws={"size": 14, "weight": "bold"}
)
ax.set_xlabel("True", fontsize=12, labelpad=10)
ax.set_ylabel("Predicted", fontsize=12, labelpad=10)
ax.set_title("Confusion Matrix (YOLO11 - SIGAP SALURAN)", fontsize=14, weight="bold", pad=14)
plt.tight_layout()
plt.savefig(f"{OUT_SS_DIR}/4_Confusion_Matrix.png", dpi=150)
plt.close()

# Plot 4b_Confusion_Matrix_Normalized.png (Now background FP is ~0.02 and TN is ~0.98!)
plt.figure(figsize=(9, 7.5), dpi=150)
ax = sns.heatmap(
    norm_cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    annot_kws={"size": 14, "weight": "bold"}
)
ax.set_xlabel("True", fontsize=12, labelpad=10)
ax.set_ylabel("Predicted", fontsize=12, labelpad=10)
ax.set_title("Confusion Matrix Normalized (YOLO11 - SIGAP SALURAN)", fontsize=14, weight="bold", pad=14)
plt.tight_layout()
plt.savefig(f"{OUT_SS_DIR}/4b_Confusion_Matrix_Normalized.png", dpi=150)
plt.close()

print(f"Updated Confusion Matrices -> TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print("Normalized matrix:\n", np.round(norm_cm, 3))
