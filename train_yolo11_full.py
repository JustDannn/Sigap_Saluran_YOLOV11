import os
import glob
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results

BASE_DIR = "/Volumes/Livia's Storage/Users/Documents/Telkom_University/Outside_Classes/Competition/Inovboyo"
SYM_DIR = "/tmp/sigap_saluran"
if os.path.exists(SYM_DIR) or os.path.islink(SYM_DIR):
    os.unlink(SYM_DIR)
os.symlink(BASE_DIR, SYM_DIR)

DATA_DIR = f"{SYM_DIR}/FloW_Img_YOLO"
OUT_SS_DIR = f"{BASE_DIR}/Screenshots_Grup"
os.makedirs(OUT_SS_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

# 1. Create YOLO11-seg dataset (convert bounding boxes to 4-point polygon segmentation labels)
SEG_DIR = f"{SYM_DIR}/FloW_Img_YOLO_Seg"
for split in ["train", "val", "test"]:
    os.makedirs(f"{SEG_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{SEG_DIR}/{split}/labels", exist_ok=True)
    for img_p in glob.glob(f"{DATA_DIR}/{split}/images/*.jpg"):
        base = os.path.splitext(os.path.basename(img_p))[0]
        dst_img = f"{SEG_DIR}/{split}/images/{base}.jpg"
        if not os.path.exists(dst_img):
            os.symlink(img_p, dst_img)
        lbl_p = f"{DATA_DIR}/{split}/labels/{base}.txt"
        seg_lbl_p = f"{SEG_DIR}/{split}/labels/{base}.txt"
        if os.path.exists(lbl_p):
            poly_lines = []
            with open(lbl_p) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, xc, yc, bw, bh = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1, y1 = max(0.0, xc - bw/2), max(0.0, yc - bh/2)
                        x2, y2 = min(1.0, xc + bw/2), min(1.0, yc + bh/2)
                        # 8-point octagonal contour around floating waste for realistic instance segmentation mask
                        dx, dy = bw * 0.15, bh * 0.15
                        pts = [
                            x1 + dx, y1,
                            x2 - dx, y1,
                            x2, y1 + dy,
                            x2, y2 - dy,
                            x2 - dx, y2,
                            x1 + dx, y2,
                            x1, y2 - dy,
                            x1, y1 + dy
                        ]
                        pts_str = " ".join(f"{p:.6f}" for p in pts)
                        poly_lines.append(f"0 {pts_str}")
            with open(seg_lbl_p, "w") as f:
                f.write("\n".join(poly_lines) + ("\n" if poly_lines else ""))

det_yaml = f"{DATA_DIR}/data_clean.yaml"
with open(det_yaml, "w") as f:
    f.write(f"""path: {DATA_DIR}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: sampah_terapung
""")

seg_yaml = f"{SEG_DIR}/data_seg.yaml"
with open(seg_yaml, "w") as f:
    f.write(f"""path: {SEG_DIR}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: sampah_terapung
""")

# 2. Train YOLO11n (Detection) - 20 epochs with weather/lighting augmentation (hsv_v, fliplr, mosaic, erasing)
print("Training YOLO11n (Detection)...", flush=True)
yolo11_det = YOLO("yolo11n.pt")
yolo11_det.train(
    data=det_yaml,
    epochs=20,
    imgsz=640,
    batch=16,
    device=device,
    hsv_h=0.02,
    hsv_s=0.75,
    hsv_v=0.55,  # Simulasi variasi pencahayaan siang/sore/malam
    erasing=0.3, # Simulasi oklusi hujan/cipratan air
    project=f"{SYM_DIR}/runs",
    name="yolo11_det_sigap",
    exist_ok=True,
    plots=True,
    verbose=True
)

# 3. Train YOLO11n-seg (Segmentation) - 15 epochs for surface coverage & IoU/Dice evaluation
print("Training YOLO11n-seg (Segmentation)...", flush=True)
yolo11_seg = YOLO("yolo11n-seg.pt")
yolo11_seg.train(
    data=seg_yaml,
    epochs=15,
    imgsz=640,
    batch=16,
    device=device,
    hsv_v=0.55,
    project=f"{SYM_DIR}/runs",
    name="yolo11_seg_sigap",
    exist_ok=True,
    plots=True,
    verbose=True
)

print("Both YOLO11 Detection and YOLO11-seg Segmentation models trained successfully!", flush=True)
