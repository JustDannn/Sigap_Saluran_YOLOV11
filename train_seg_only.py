import os
import shutil
import torch
from ultralytics import YOLO

BASE_DIR = "/Volumes/Livia's Storage/Users/Documents/Telkom_University/Outside_Classes/Competition/Inovboyo"
SYM_DIR = "/tmp/sigap_saluran"
SEG_DIR = f"{SYM_DIR}/FloW_Img_YOLO_Seg"
seg_yaml = f"{SEG_DIR}/data_seg.yaml"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Training YOLO11n-seg (Segmentation)...", flush=True)
yolo11_seg = YOLO("yolo11n-seg.pt")
try:
    yolo11_seg.train(
        data=seg_yaml,
        epochs=15,
        imgsz=640,
        batch=16,
        device=device,
        hsv_v=0.55,
        project="/tmp/sigap_runs",
        name="yolo11_seg_sigap",
        exist_ok=True,
        plots=True,
        verbose=True
    )
except Exception as e:
    print("Note:", e)

os.makedirs(f"{BASE_DIR}/runs", exist_ok=True)
if os.path.exists("/tmp/sigap_runs/yolo11_seg_sigap"):
    if os.path.exists(f"{BASE_DIR}/runs/yolo11_seg_sigap"):
        shutil.rmtree(f"{BASE_DIR}/runs/yolo11_seg_sigap")
    shutil.copytree("/tmp/sigap_runs/yolo11_seg_sigap", f"{BASE_DIR}/runs/yolo11_seg_sigap")
print("YOLO11n-seg saved to runs/yolo11_seg_sigap!", flush=True)
