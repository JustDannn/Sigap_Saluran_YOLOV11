import os
import glob
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results

BASE_DIR = "/Volumes/Livia's Storage/Users/Documents/Telkom_University/Outside_Classes/Competition/Inovboyo"
RUN_DIR = f"{BASE_DIR}/runs/sigap_saluran_yolo"
OUT_SS_DIR = f"{BASE_DIR}/Screenshots_Grup"
os.makedirs(OUT_SS_DIR, exist_ok=True)

# 1. Generate results.png from results.csv
plot_results(file=f"{RUN_DIR}/results.csv", dir=RUN_DIR)
if os.path.exists(f"{RUN_DIR}/results.png"):
    shutil.copy(f"{RUN_DIR}/results.png", f"{OUT_SS_DIR}/3_Grafik_Training_Evaluasi_YOLO.png")
    print("Generated 3_Grafik_Training_Evaluasi_YOLO.png")

# 2. Setup symlink without apostrophe so Ultralytics AutoBackend doesn't strip apostrophe
SYM_DIR = "/tmp/sigap_saluran"
if os.path.exists(SYM_DIR) or os.path.islink(SYM_DIR):
    os.unlink(SYM_DIR)
os.symlink(BASE_DIR, SYM_DIR)

# Create clean data.yaml using /tmp/sigap_saluran
clean_yaml = f"{SYM_DIR}/FloW_Img_YOLO/data_clean.yaml"
with open(clean_yaml, "w") as f:
    f.write(f"""path: {SYM_DIR}/FloW_Img_YOLO
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: sampah_terapung
""")

best_pt = f"{SYM_DIR}/runs/sigap_saluran_yolo/weights/best.pt"
best_model = YOLO(best_pt)

# Run validation to generate confusion_matrix.png, PR_curve.png, val_batch0_pred.jpg
metrics = best_model.val(
    data=clean_yaml,
    split="val",
    project=f"{SYM_DIR}/runs",
    name="val_eval",
    exist_ok=True,
    plots=True
)
print(f"FINAL METRICS -> Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%), Recall: {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%), mAP50: {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%), mAP50-95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")

val_dir = f"{BASE_DIR}/runs/val_eval"
for src_name, dst_name in [
    ("confusion_matrix.png", "4_Confusion_Matrix.png"),
    ("confusion_matrix_normalized.png", "4b_Confusion_Matrix_Normalized.png"),
    ("BoxPR_curve.png", "5_PR_Curve_mAP.png"),
    ("PR_curve.png", "5_PR_Curve_mAP.png"),
    ("BoxF1_curve.png", "5b_F1_Curve.png"),
    ("F1_curve.png", "5b_F1_Curve.png"),
    ("val_batch0_pred.jpg", "6_Hasil_Deteksi_Batch_YOLO.jpg")
]:
    src_p = f"{val_dir}/{src_name}"
    if os.path.exists(src_p):
        shutil.copy(src_p, f"{OUT_SS_DIR}/{dst_name}")
        print(f"Copied {src_name} -> {dst_name}")

# 3. Render Prototype SIGAP SALURAN Detection + Blockage Coverage Estimation Screenshots
def render_sigap_prototype(img_path, out_path, cam_name="CAM-01: PINTU AIR / SALURAN"):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    res = best_model.predict(img, conf=0.25, iou=0.45, verbose=False)[0]
    boxes = res.boxes

    cov_mask = np.zeros((h, w), dtype=np.uint8)
    det_count = len(boxes)

    overlay = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        mx1, my1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        mx2, my2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        cov_mask[my1:my2, mx1:mx2] = 255

        color = (0, 140, 255) if conf < 0.65 else (0, 40, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        label = f"sampah_terapung {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 5, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    raw_box_ratio = (np.count_nonzero(cov_mask) / float(h * w)) * 100.0
    sumbatan_pct = min(95.0, raw_box_ratio * 3.5 + (det_count * 2.2))

    if sumbatan_pct >= 30.0 or det_count >= 7:
        status_txt = "RAWAN SUMBATAN (SIAGA)"
        status_col = (0, 0, 220)
    elif sumbatan_pct >= 12.0 or det_count >= 3:
        status_txt = "WASPADA PENUMPUKAN"
        status_col = (0, 140, 255)
    else:
        status_txt = "AMAN / TERKENDALI"
        status_col = (0, 180, 0)

    # Scale HUD according to image resolution
    scale = max(1.0, w / 1280.0)
    pw, ph = int(460 * scale), int(145 * scale)
    sub = overlay[20:20+ph, 20:20+pw]
    black_rect = np.zeros(sub.shape, dtype=np.uint8)
    overlay[20:20+ph, 20:20+pw] = cv2.addWeighted(sub, 0.25, black_rect, 0.75, 1.0)
    cv2.rectangle(overlay, (20, 20), (20+pw, 20+ph), (220, 220, 220), 2)

    cv2.putText(overlay, f"SIGAP SALURAN AI - {cam_name}", (int(35*scale), int(50*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62*scale, (0, 255, 255), int(2*scale), cv2.LINE_AA)
    cv2.putText(overlay, f"Objek Terdeteksi : {det_count} Sampah Terapung", (int(35*scale), int(80*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55*scale, (255, 255, 255), max(1, int(1.5*scale)), cv2.LINE_AA)
    cv2.putText(overlay, f"Estimasi Sumbatan: {sumbatan_pct:.1f}% Area Saluran", (int(35*scale), int(108*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55*scale, (255, 255, 255), max(1, int(1.5*scale)), cv2.LINE_AA)
    cv2.rectangle(overlay, (int(35*scale), int(120*scale)), (int(440*scale), int(152*scale)), status_col, -1)
    cv2.putText(overlay, f"STATUS: {status_txt}", (int(48*scale), int(143*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58*scale, (255, 255, 255), int(2*scale), cv2.LINE_AA)

    cv2.imwrite(out_path, overlay)

test_imgs = sorted(glob.glob(f"{SYM_DIR}/FloW_Img_YOLO/test/images/*.jpg") + glob.glob(f"{SYM_DIR}/FloW_Img_YOLO/val/images/*.jpg"))
scored_test = []
for p in test_imgs:
    r = best_model.predict(p, conf=0.25, verbose=False)[0]
    if len(r.boxes) > 0:
        scored_test.append((len(r.boxes), p))
scored_test.sort(reverse=True)

for idx, (cnt, p) in enumerate(scored_test[:4]):
    render_sigap_prototype(
        p,
        f"{OUT_SS_DIR}/7_Prototype_Deteksi_SIGAP_{idx+1}.jpg",
        cam_name=f"CCTV SALURAN #{idx+1}"
    )

# Also create a 2x2 grid of prototype detections for easy single-image sharing
grid_tiles = []
for idx in range(min(4, len(scored_test))):
    im = cv2.imread(f"{OUT_SS_DIR}/7_Prototype_Deteksi_SIGAP_{idx+1}.jpg")
    if im is not None:
        grid_tiles.append(cv2.resize(im, (640, 360)))
if len(grid_tiles) == 4:
    proto_grid = np.vstack([np.hstack(grid_tiles[:2]), np.hstack(grid_tiles[2:])])
    cv2.imwrite(f"{OUT_SS_DIR}/7_Grid_Prototype_Deteksi_SIGAP.jpg", proto_grid)

print("All files in Screenshots_Grup:", sorted(os.listdir(OUT_SS_DIR)))
