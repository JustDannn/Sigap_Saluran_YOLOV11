import os
import glob
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = "/Volumes/Livia's Storage/Users/Documents/Telkom_University/Outside_Classes/Competition/Inovboyo"
DATA_DIR = f"{BASE_DIR}/FloW_Img_YOLO"
OUT_SS_DIR = f"{BASE_DIR}/Screenshots_Grup"
os.makedirs(OUT_SS_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

# 1. Generate Raw Dataset Grid & Ground-Truth Annotation Grid first
def make_dataset_and_annotation_previews():
    train_imgs = sorted(glob.glob(f"{DATA_DIR}/train/images/*.jpg"))
    # Find images with multiple bounding boxes so the preview looks rich
    rich_samples = []
    for img_path in train_imgs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = f"{DATA_DIR}/train/labels/{base}.txt"
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) >= 2:
                rich_samples.append((len(lines), img_path, lbl_path))
    rich_samples.sort(reverse=True)
    chosen = rich_samples[:4] if len(rich_samples) >= 4 else [(0, p, f"{DATA_DIR}/train/labels/{os.path.splitext(os.path.basename(p))[0]}.txt") for p in train_imgs[:4]]

    raw_tiles = []
    ann_tiles = []
    target_w, target_h = 640, 360

    for count, img_p, lbl_p in chosen:
        img = cv2.imread(img_p)
        if img is None:
            continue
        img_resized = cv2.resize(img, (target_w, target_h))
        
        # Add subtle title bar on raw tile
        raw_tile = img_resized.copy()
        cv2.rectangle(raw_tile, (0, 0), (target_w, 32), (20, 20, 20), -1)
        cv2.putText(raw_tile, f"FloW-Img Sample: {os.path.basename(img_p)} ({count} objects)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        raw_tiles.append(raw_tile)

        # Draw ground-truth annotations
        ann_tile = img_resized.copy()
        with open(lbl_p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, xc, yc, bw, bh = map(float, parts)
                    x1 = int((xc - bw / 2) * target_w)
                    y1 = int((yc - bh / 2) * target_h)
                    x2 = int((xc + bw / 2) * target_w)
                    y2 = int((yc + bh / 2) * target_h)
                    cv2.rectangle(ann_tile, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.rectangle(ann_tile, (x1, max(0, y1 - 20)), (x1 + 135, y1), (0, 255, 255), -1)
                    cv2.putText(ann_tile, "sampah_terapung", (x1 + 4, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(ann_tile, (0, 0), (target_w, 32), (20, 20, 20), -1)
        cv2.putText(ann_tile, f"Ground Truth Annotation (YOLO Format) - {os.path.basename(img_p)}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        ann_tiles.append(ann_tile)

    if len(raw_tiles) == 4:
        raw_grid = np.vstack([np.hstack(raw_tiles[:2]), np.hstack(raw_tiles[2:])])
        ann_grid = np.vstack([np.hstack(ann_tiles[:2]), np.hstack(ann_tiles[2:])])
        cv2.imwrite(f"{OUT_SS_DIR}/1_Contoh_Dataset_FloW_Img.jpg", raw_grid)
        cv2.imwrite(f"{OUT_SS_DIR}/2_Contoh_Anotasi_Bounding_Box.jpg", ann_grid)
        print("Saved 1_Contoh_Dataset_FloW_Img.jpg and 2_Contoh_Anotasi_Bounding_Box.jpg", flush=True)

make_dataset_and_annotation_previews()

# 2. Train YOLOv8n on FloW-Img
print("Starting YOLOv8n training on FloW-Img...", flush=True)
model = YOLO("yolov8n.pt")
results = model.train(
    data=f"{DATA_DIR}/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    device=device,
    project=f"{BASE_DIR}/runs",
    name="sigap_saluran_yolo",
    exist_ok=True,
    plots=True,
    verbose=True
)

run_dir = f"{BASE_DIR}/runs/sigap_saluran_yolo"
best_pt = f"{run_dir}/weights/best.pt"
if not os.path.exists(best_pt):
    best_pt = f"{run_dir}/weights/last.pt"

# Validate to make sure all evaluation plots (confusion matrix, PR curve, etc.) are fresh
best_model = YOLO(best_pt)
metrics = best_model.val(data=f"{DATA_DIR}/data.yaml", split="val", plots=True)
print(f"Validation Metrics -> Precision: {metrics.box.mp:.4f}, Recall: {metrics.box.mr:.4f}, mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}", flush=True)

# Copy training plots to Screenshots_Grup
for src_name, dst_name in [
    ("results.png", "3_Grafik_Training_Evaluasi_YOLO.png"),
    ("confusion_matrix.png", "4_Confusion_Matrix.png"),
    ("confusion_matrix_normalized.png", "4b_Confusion_Matrix_Normalized.png"),
    ("PR_curve.png", "5_PR_Curve_mAP.png"),
    ("val_batch0_pred.jpg", "6_Hasil_Deteksi_Batch_YOLO.jpg")
]:
    src_p = f"{run_dir}/{src_name}"
    if os.path.exists(src_p):
        shutil.copy(src_p, f"{OUT_SS_DIR}/{dst_name}")

# 3. Create Prototype SIGAP SALURAN Detection + Blockage Coverage Estimation Screenshots
def render_sigap_prototype(img_path, out_path, cam_name="CAM-01: PINTU AIR / SALURAN"):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    res = best_model.predict(img, conf=0.20, iou=0.45, verbose=False)[0]
    boxes = res.boxes

    # Create mask to compute union coverage area of detected floating waste
    # Also compute blockage index weighted to water surface region
    cov_mask = np.zeros((h, w), dtype=np.uint8)
    det_count = len(boxes)

    overlay = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        # Expand slightly around bounding box cluster to represent surface blockage footprint
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        mx1, my1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        mx2, my2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        cov_mask[my1:my2, mx1:mx2] = 255

        # Draw bounding box
        color = (0, 180, 255) if conf < 0.6 else (0, 60, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"sampah_terapung {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 4, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Calculate coverage percentage relative to active channel zone (approx 35% of frame)
    raw_box_ratio = (np.count_nonzero(cov_mask) / float(h * w)) * 100.0
    # Scale to channel cross-section blockage estimate
    sumbatan_pct = min(95.0, raw_box_ratio * 3.2 + (det_count * 1.8))

    if sumbatan_pct >= 35.0 or det_count >= 8:
        status_txt = "RAWAN SUMBATAN (SIAGA)"
        status_col = (0, 0, 230)
    elif sumbatan_pct >= 15.0 or det_count >= 3:
        status_txt = "WASPADA PENUMPUKAN"
        status_col = (0, 165, 255)
    else:
        status_txt = "AMAN / LANCAR"
        status_col = (0, 200, 0)

    # Draw HUD Dashboard Panel (Top-Left)
    panel_w, panel_h = 430, 135
    sub = overlay[15:15+panel_h, 15:15+panel_w]
    black_rect = np.zeros(sub.shape, dtype=np.uint8)
    res_panel = cv2.addWeighted(sub, 0.25, black_rect, 0.75, 1.0)
    overlay[15:15+panel_h, 15:15+panel_w] = res_panel
    cv2.rectangle(overlay, (15, 15), (15+panel_w, 15+panel_h), (200, 200, 200), 1)

    cv2.putText(overlay, f"SIGAP SALURAN AI - {cam_name}", (28, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Objek Terdeteksi : {det_count} Sampah Terapung", (28, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"Estimasi Sumbatan: {sumbatan_pct:.1f}% Area Saluran", (28, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(overlay, (28, 108), (28 + 390, 138), status_col, -1)
    cv2.putText(overlay, f"STATUS: {status_txt}", (38, 129),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, overlay)

# Pick best test/val images with multiple detections
test_imgs = sorted(glob.glob(f"{DATA_DIR}/test/images/*.jpg") + glob.glob(f"{DATA_DIR}/val/images/*.jpg"))
scored_test = []
for p in test_imgs:
    r = best_model.predict(p, conf=0.20, verbose=False)[0]
    if len(r.boxes) > 0:
        scored_test.append((len(r.boxes), p))
scored_test.sort(reverse=True)

for idx, (cnt, p) in enumerate(scored_test[:4]):
    render_sigap_prototype(
        p,
        f"{OUT_SS_DIR}/7_Prototype_Deteksi_SIGAP_{idx+1}.jpg",
        cam_name=f"CCTV SALURAN #{idx+1}"
    )

print("All prototype screenshots generated in:", OUT_SS_DIR, flush=True)
