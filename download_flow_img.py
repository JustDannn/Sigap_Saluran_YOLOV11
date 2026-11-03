import urllib.request
import struct
import zlib
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = 'https://huggingface.co/datasets/gonzz2026/A-dataset-of-images-of-floating-rubbish-on-the-surface-of-the-water/resolve/main/A-dataset-of-images-of-floating-rubbish-on-the-surface-of-the-water.zip'
OUT_DIR = "/Volumes/Livia's Storage/Users/Documents/Telkom_University/Outside_Classes/Competition/Inovboyo/FloW_Img_YOLO"

print("Fetching Central Directory from Hugging Face...", flush=True)
req = urllib.request.Request(URL, method='HEAD')
size = int(urllib.request.urlopen(req).headers['Content-Length'])

req2 = urllib.request.Request(URL, headers={'Range': f'bytes={size-2000000}-{size-1}'})
tail = urllib.request.urlopen(req2).read()

pos = 0
entries = {}
while True:
    idx = tail.find(b'PK\x01\x02', pos)
    if idx == -1 or idx + 46 > len(tail):
        break
    method = struct.unpack('<H', tail[idx+10:idx+12])[0]
    comp_size, uncomp_size, fname_len, extra_len, comment_len, _, _, _, offset = struct.unpack('<IIHHHHIHI', tail[idx+20:idx+46])
    fname = tail[idx+46:idx+46+fname_len].decode('utf-8', errors='ignore')
    if comp_size > 0 and not fname.endswith('/'):
        entries[fname] = (method, comp_size, uncomp_size, offset)
    pos = idx + 46 + fname_len + extra_len + comment_len

def fetch_entry(fname, info):
    method, comp_size, uncomp_size, offset = info
    end_byte = offset + 30 + len(fname) + 128 + comp_size
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{end_byte}'})
            raw = urllib.request.urlopen(req, timeout=25).read()
            if raw[:4] != b'PK\x03\x04':
                raise ValueError(f"Bad local header for {fname}")
            lf_fname_len, lf_extra_len = struct.unpack('<HH', raw[26:30])
            data_start = 30 + lf_fname_len + lf_extra_len
            comp_data = raw[data_start:data_start + comp_size]
            if method == 0:
                return comp_data
            elif method == 8:
                return zlib.decompress(comp_data, -zlib.MAX_WBITS)
        except Exception as e:
            if attempt == 2:
                raise e

random.seed(42)
counts = {'train': 120, 'val': 30, 'test': 30}
to_download = []

for split, limit in counts.items():
    os.makedirs(f"{OUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/{split}/labels", exist_ok=True)
    # Keep already downloaded images in train if present
    existing = [os.path.splitext(f)[0] for f in os.listdir(f"{OUT_DIR}/{split}/images") if f.endswith('.jpg')]
    img_keys = sorted([k for k in entries if k.startswith(f"datasets/{split}/images/") and k.endswith('.jpg')])
    selected = []
    for ex in existing:
        ik = f"datasets/{split}/images/{ex}.jpg"
        if ik in entries and len(selected) < limit:
            selected.append(ik)
    remaining = [k for k in img_keys if k not in selected]
    if len(selected) < limit:
        selected.extend(random.sample(remaining, min(limit - len(selected), len(remaining))))
    for ik in selected:
        base = os.path.splitext(os.path.basename(ik))[0]
        lk = f"datasets/{split}/labels/{base}.txt"
        if lk in entries:
            to_download.append((split, base, ik, lk))

print(f"Ensuring {len(to_download)} image+label pairs from FloW-Img...", flush=True)

def process_pair(item):
    split, base, ik, lk = item
    img_path = f"{OUT_DIR}/{split}/images/{base}.jpg"
    lbl_path = f"{OUT_DIR}/{split}/labels/{base}.txt"
    if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
        img_bytes = fetch_entry(ik, entries[ik])
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        lbl_bytes = fetch_entry(lk, entries[lk])
        lines = lbl_bytes.decode('utf-8', errors='ignore').strip().splitlines()
        cleaned = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                parts[0] = '0'
                cleaned.append(' '.join(parts))
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(cleaned) + ('\n' if cleaned else ''))
    return base

done = 0
with ThreadPoolExecutor(max_workers=24) as ex:
    futures = [ex.submit(process_pair, item) for item in to_download]
    for fut in as_completed(futures):
        try:
            fut.result()
            done += 1
            if done % 20 == 0 or done == len(to_download):
                print(f"Progress: {done}/{len(to_download)} pairs ready", flush=True)
        except Exception as e:
            print("Error:", e, flush=True)

yaml_content = f"""path: {OUT_DIR}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: sampah_terapung
"""
with open(f"{OUT_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)

print("FloW-Img YOLO dataset ready at:", OUT_DIR, flush=True)
