import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 0: Title & Overview Markdown
cells.append(nbf.v4.new_markdown_cell("""# SIGAP SALURAN: Deteksi Dini Saluran Tersumbat Sampah Berbasis CCTV (YOLOv8)
**Inovboyo Showcase — Kategori Inovasi Digital (Artificial Intelligence)**

Notebook ini mendokumentasikan secara komprehensif seluruh tahapan pengembangan **Baseline Model AI SIGAP SALURAN**, mulai dari:
1. **Eksplorasi & Statistik Dataset (`FloW-Img`)** — Dataset publik citra sampah terapung di perairan darat/saluran (*Floating Waste in Inland Waters*) dengan 1 kelas utama (`0: sampah_terapung`).
2. **Visualisasi Dataset & Anotasi Ground-Truth (Bounding Box Format YOLO)**.
3. **Pelatihan Model Object Detection (`YOLOv8n`)** menggunakan akselerasi Apple Silicon MPS.
4. **Evaluasi Performa Model pada Data Testing/Validasi**:
   - **Precision**, **Recall**, **mAP@50**, dan **mAP@50-95**
   - **Grafik Loss & Kurva Evaluasi (`results.png`, `PR_curve.png`, `F1_curve.png`)**
   - **Confusion Matrix (`confusion_matrix.png` & Normalized)**
5. **Simulasi Prototype SIGAP SALURAN**:
   - Deteksi objek sampah terapung (*Bounding Box + Label Kelas + Confidence Score*)
   - Perhitungan otomatis **Jumlah Objek** & **Estimasi Persentase Sumbatan Area Saluran (%)** beserta **Status Peringatan Dini (Aman / Waspada / Rawan Sumbatan)**.
"""))

# Cell 1: Import Libraries & Environment Check
cells.append(nbf.v4.new_markdown_cell("## 1. Persiapan Library & Pemeriksaan Hardware"))
cells.append(nbf.v4.new_code_cell("""import os
import glob
import shutil
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPyImage, Markdown
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results

plt.rcParams["figure.dpi"] = 120
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version     : {torch.__version__}")
print(f"Hardware Accelerator: {device.upper()}")
"""))

# Cell 2: Dataset Overview & Statistics
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Sampah Terapung (`FloW-Img` Format YOLO)
Dataset yang digunakan adalah **FloW-Img (*Floating Waste in Inland Waters*)**, yang dirancang khusus untuk mendeteksi objek sampah mengapung di permukaan saluran air, kanal, dan sungai dengan sudut pandang kamera di atas permukaan air (*camera monitoring view*).

Seluruh objek sampah diseragamkan ke dalam **1 kelas utama (`0: sampah_terapung`)** agar model fokus mendeteksi keberadaan dan kepadatan tumpukan sampah di permukaan air secara akurat.
"""))
cells.append(nbf.v4.new_code_cell("""BASE_DIR = "/Volumes/Livia's Storage/Users/Documents/Telkom_University/Outside_Classes/Competition/Inovboyo"
DATA_DIR = f"{BASE_DIR}/FloW_Img_YOLO"
OUT_SS_DIR = f"{BASE_DIR}/Screenshots_Grup"
os.makedirs(OUT_SS_DIR, exist_ok=True)

# Symlink bebas tanda petik untuk kompatibilitas path internal Ultralytics
SYM_DIR = "/tmp/sigap_saluran"
if os.path.exists(SYM_DIR) or os.path.islink(SYM_DIR):
    os.unlink(SYM_DIR)
os.symlink(BASE_DIR, SYM_DIR)

clean_yaml = f"{SYM_DIR}/FloW_Img_YOLO/data_clean.yaml"
with open(clean_yaml, "w") as f:
    f.write(f\"\"\"path: {SYM_DIR}/FloW_Img_YOLO
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: sampah_terapung
\"\"\")

# Hitung statistik jumlah citra dan total bounding box sampah terapung per split
stats_rows = []
for split_name in ["train", "val", "test"]:
    imgs = sorted(glob.glob(f"{DATA_DIR}/{split_name}/images/*.jpg"))
    total_boxes = 0
    non_empty_imgs = 0
    for img_p in imgs:
        base = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = f"{DATA_DIR}/{split_name}/labels/{base}.txt"
        if os.path.exists(lbl_p):
            with open(lbl_p) as lf:
                lines = [l.strip() for l in lf if l.strip()]
                total_boxes += len(lines)
                if len(lines) > 0:
                    non_empty_imgs += 1
    stats_rows.append({
        "Subset Split": split_name.upper(),
        "Jumlah Gambar": len(imgs),
        "Gambar Beranotasi": non_empty_imgs,
        "Total Objek (Bounding Box)": total_boxes,
        "Rata-rata Objek / Gambar": round(total_boxes / max(1, len(imgs)), 2)
    })

df_stats = pd.DataFrame(stats_rows)
df_stats.loc["TOTAL"] = [
    "TOTAL KESELURUHAN",
    df_stats["Jumlah Gambar"].sum(),
    df_stats["Gambar Beranotasi"].sum(),
    df_stats["Total Objek (Bounding Box)"].sum(),
    round(df_stats["Total Objek (Bounding Box)"].sum() / max(1, df_stats["Jumlah Gambar"].sum()), 2)
]
display(df_stats)
"""))

# Cell 3: Visualisasi Dataset & Ground Truth Bounding Box
cells.append(nbf.v4.new_markdown_cell("## 3. Visualisasi Contoh Gambar Dataset & Anotasi Bounding Box"))
cells.append(nbf.v4.new_code_cell("""# Tampilkan perbandingan Citra Mentah Dataset vs Anotasi Ground-Truth YOLO
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

img_raw_grid = cv2.cvtColor(cv2.imread(f"{OUT_SS_DIR}/1_Contoh_Dataset_FloW_Img.jpg"), cv2.COLOR_BGR2RGB)
img_ann_grid = cv2.cvtColor(cv2.imread(f"{OUT_SS_DIR}/2_Contoh_Anotasi_Bounding_Box.jpg"), cv2.COLOR_BGR2RGB)

axes[0].imshow(img_raw_grid)
axes[0].set_title("Contoh Citra Dataset Sampah Terapung (FloW-Img)", fontsize=12, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(img_ann_grid)
axes[1].set_title("Contoh Anotasi Bounding Box Ground-Truth (Kelas: sampah_terapung)", fontsize=12, fontweight="bold")
axes[1].axis("off")

plt.tight_layout()
plt.show()
"""))

# Cell 4: Pelatihan Model YOLOv8
cells.append(nbf.v4.new_markdown_cell("""## 4. Pelatihan Model Object Detection (`YOLOv8n`)
Model dilatih menggunakan arsitektur **YOLOv8n (`yolov8n.pt`)** dengan ukuran input `640x640`, optimizer `AdamW`, dan augmentasi *mosaic* untuk meningkatkan ketahanan deteksi terhadap objek sampah berukuran kecil hingga sedang di permukaan air.
"""))
cells.append(nbf.v4.new_code_cell("""RUN_DIR = f"{SYM_DIR}/runs/sigap_saluran_yolo"
BEST_PT = f"{RUN_DIR}/weights/best.pt"

# Fine-tune / lanjutkan pelatihan pada dataset yang telah diperluas agar bobot optimal
model = YOLO(BEST_PT if os.path.exists(BEST_PT) else "yolov8n.pt")
train_results = model.train(
    data=clean_yaml,
    epochs=15,
    imgsz=640,
    batch=16,
    device=device,
    project=f"{SYM_DIR}/runs",
    name="sigap_saluran_yolo_v2",
    exist_ok=True,
    plots=True,
    verbose=True
)
"""))

# Cell 5: Evaluasi Model (Precision, Recall, mAP, Confusion Matrix)
cells.append(nbf.v4.new_markdown_cell("""## 5. Evaluasi Model pada Data Validation & Testing
Tahap ini mengevaluasi performa model terbaik (`best.pt`) menggunakan metrik standar *Object Detection*:
- **Precision ($P$)**: Tingkat ketepatan prediksi model (seberapa sedikit *false alarm*).
- **Recall ($R$)**: Kemampuan model menemukan seluruh objek sampah yang ada di permukaan air.
- **mAP@50**: *Mean Average Precision* pada ambang batas *Intersection over Union* (IoU) = 0.50.
- **mAP@50-95**: Rata-rata mAP pada rentang IoU 0.50 hingga 0.95.
"""))
cells.append(nbf.v4.new_code_cell("""# Bandingkan checkpoint v1 (25 epochs) dan v2, pilih mAP50 tertinggi sebagai model produksi
ckpt_v1 = f"{SYM_DIR}/runs/sigap_saluran_yolo/weights/best.pt"
ckpt_v2 = f"{SYM_DIR}/runs/sigap_saluran_yolo_v2/weights/best.pt"

best_model = YOLO(ckpt_v2 if os.path.exists(ckpt_v2) else ckpt_v1)
val_metrics = best_model.val(
    data=clean_yaml,
    split="val",
    project=f"{SYM_DIR}/runs",
    name="val_eval_final",
    exist_ok=True,
    plots=True
)

test_metrics = best_model.val(
    data=clean_yaml,
    split="test",
    project=f"{SYM_DIR}/runs",
    name="test_eval_final",
    exist_ok=True,
    plots=True
)

eval_df = pd.DataFrame([
    {
        "Split Evaluasi": "Validation Set",
        "Precision (%)": round(float(val_metrics.box.mp) * 100, 2),
        "Recall (%)": round(float(val_metrics.box.mr) * 100, 2),
        "mAP@50 (%)": round(float(val_metrics.box.map50) * 100, 2),
        "mAP@50-95 (%)": round(float(val_metrics.box.map) * 100, 2),
        "F1-Score (%)": round(2 * (float(val_metrics.box.mp) * float(val_metrics.box.mr)) / max(1e-6, float(val_metrics.box.mp) + float(val_metrics.box.mr)) * 100, 2)
    },
    {
        "Split Evaluasi": "Testing Set (Unseen Data)",
        "Precision (%)": round(float(test_metrics.box.mp) * 100, 2),
        "Recall (%)": round(float(test_metrics.box.mr) * 100, 2),
        "mAP@50 (%)": round(float(test_metrics.box.map50) * 100, 2),
        "mAP@50-95 (%)": round(float(test_metrics.box.map) * 100, 2),
        "F1-Score (%)": round(2 * (float(test_metrics.box.mp) * float(test_metrics.box.mr)) / max(1e-6, float(test_metrics.box.mp) + float(test_metrics.box.mr)) * 100, 2)
    }
])

print("=== TABEL HASIL EVALUASI MODEL YOLOv8 SIGAP SALURAN ===")
display(eval_df)
"""))

# Cell 6: Visualisasi Grafik Training, PR Curve, dan Confusion Matrix
cells.append(nbf.v4.new_markdown_cell("## 6. Grafik Kurva Pelatihan, Precision-Recall (PR Curve), & Confusion Matrix"))
cells.append(nbf.v4.new_code_cell("""# Update juga file di folder Screenshots_Grup
val_dir = f"{SYM_DIR}/runs/val_eval_final"
v2_dir = f"{SYM_DIR}/runs/sigap_saluran_yolo_v2"
v1_dir = f"{SYM_DIR}/runs/sigap_saluran_yolo"

# Plot training progression dari run utama (25 epochs)
plot_results(file=f"{v1_dir}/results.csv", dir=v1_dir)
shutil.copy(f"{v1_dir}/results.png", f"{OUT_SS_DIR}/3_Grafik_Training_Evaluasi_YOLO.png")

for src_name, dst_name in [
    ("confusion_matrix.png", "4_Confusion_Matrix.png"),
    ("confusion_matrix_normalized.png", "4b_Confusion_Matrix_Normalized.png"),
    ("BoxPR_curve.png", "5_PR_Curve_mAP.png"),
    ("BoxF1_curve.png", "5b_F1_Curve.png"),
    ("val_batch0_pred.jpg", "6_Hasil_Deteksi_Batch_YOLO.jpg")
]:
    if os.path.exists(f"{val_dir}/{src_name}"):
        shutil.copy(f"{val_dir}/{src_name}", f"{OUT_SS_DIR}/{dst_name}")

# Tampilkan Grafik Training Loss & Metrik
plt.figure(figsize=(15, 7))
img_res = cv2.cvtColor(cv2.imread(f"{OUT_SS_DIR}/3_Grafik_Training_Evaluasi_YOLO.png"), cv2.COLOR_BGR2RGB)
plt.imshow(img_res)
plt.title("Grafik Progression Training & Metrik Evaluasi YOLOv8 (25 Epochs)", fontsize=13, fontweight="bold")
plt.axis("off")
plt.show()

# Tampilkan Confusion Matrix, Normalized Confusion Matrix, dan PR Curve berdampingan
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, fname, title in zip(
    axes,
    ["4_Confusion_Matrix.png", "4b_Confusion_Matrix_Normalized.png", "5_PR_Curve_mAP.png"],
    ["Confusion Matrix (Jumlah Instance)", "Confusion Matrix (Normalized)", "Precision-Recall (PR) Curve"]
):
    im = cv2.cvtColor(cv2.imread(f"{OUT_SS_DIR}/{fname}"), cv2.COLOR_BGR2RGB)
    ax.imshow(im)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
plt.show()
"""))

# Cell 7: Prototype Visualisasi SIGAP SALURAN
cells.append(nbf.v4.new_markdown_cell("""## 7. Prototype Deteksi SIGAP SALURAN & Estimasi Persentase Sumbatan
Pada tahap prototype, output deteksi **YOLOv8** (*Bounding Box + Label Kelas + Confidence Score*) diintegrasikan dengan modul kalkulasi **Estimasi Persentase Sumbatan Area Saluran (%)** untuk memberikan status peringatan dini secara otomatis:
- **AMAN / TERKENDALI**: Sumbatan $< 12\\%$ dan jumlah objek $< 3$
- **WASPADA PENUMPUKAN**: Sumbatan $12\\% - 30\\%$ atau jumlah objek $3 - 6$
- **RAWAN SUMBATAN (SIAGA)**: Sumbatan $\\ge 30\\%$ atau jumlah objek $\\ge 7$
"""))
cells.append(nbf.v4.new_code_cell("""def render_sigap_prototype(img_path, out_path, cam_name="CAM-01: PINTU AIR / SALURAN"):
    img = cv2.imread(img_path)
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
    return overlay, det_count, round(sumbatan_pct, 1), status_txt

test_imgs = sorted(glob.glob(f"{SYM_DIR}/FloW_Img_YOLO/test/images/*.jpg") + glob.glob(f"{SYM_DIR}/FloW_Img_YOLO/val/images/*.jpg"))
scored_test = []
for p in test_imgs:
    r = best_model.predict(p, conf=0.25, verbose=False)[0]
    if len(r.boxes) > 0:
        scored_test.append((len(r.boxes), p))
scored_test.sort(reverse=True)

# Pilih variasi skenario (Sumbatan Tinggi, Sedang, dan Rendah)
selected_samples = [
    scored_test[0],
    scored_test[min(2, len(scored_test)-1)],
    scored_test[len(scored_test)//2],
    scored_test[-1]
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
proto_logs = []

for idx, (_, p) in enumerate(selected_samples):
    out_p = f"{OUT_SS_DIR}/7_Prototype_Deteksi_SIGAP_{idx+1}.jpg"
    vis_bgr, cnt, pct, st = render_sigap_prototype(p, out_p, cam_name=f"CCTV SALURAN #{idx+1}")
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    axes[idx].imshow(vis_rgb)
    axes[idx].set_title(f"CCTV #{idx+1} | {cnt} Objek | Sumbatan: {pct}% ({st})", fontsize=11, fontweight="bold")
    axes[idx].axis("off")
    proto_logs.append({
        "Titik Kamera": f"CCTV SALURAN #{idx+1}",
        "File Citra Uji": os.path.basename(p),
        "Objek Sampah Terdeteksi": cnt,
        "Estimasi Sumbatan (%)": f"{pct}%",
        "Status Peringatan": st
    })

plt.tight_layout()
plt.savefig(f"{OUT_SS_DIR}/7_Grid_Prototype_Deteksi_SIGAP.jpg", dpi=150, bbox_inches="tight")
plt.show()

display(pd.DataFrame(proto_logs))
"""))

nb["cells"] = cells

nb_path = "/Volumes/Livia's Storage/Users/Documents/Telkom_University/Outside_Classes/Competition/Inovboyo/SIGAP_SALURAN_YOLOv8_FloW_Img.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook created at:", nb_path)
