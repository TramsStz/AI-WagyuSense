"""
AI-WagyuSense: Automated 224x224 ROI Dataset Extraction for Model 2
Wat Khemaphirataram School - KIDE 2026 Project

Uses trained YOLOv8n-seg Version 4 to:
  1. Detect Longissimus dorsi muscle instances across Dataset Version 4 splits (train, valid, test)
  2. Perform Specular Glare Inpainting (Telea) & CLAHE enhancement
  3. Compute ground truth %IMF via Otsu morphometry
  4. Classify into TAS Quality Grade (Grade 1 - 5)
  5. Letterbox pad & resize into standardized 224x224 patches for EfficientNet-B0
  6. Generate structured metadata CSV files for training & evaluation
"""

import os
import glob
import time
import csv
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

def load_image_robust(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        pil_img = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def write_image_unicode(path, img):
    try:
        ext = os.path.splitext(path)[1]
        res, enc = cv2.imencode(ext, img)
        if res:
            enc.tofile(path)
            return True
    except Exception as e:
        print(f"Error writing {path}: {e}")
    return False

def letterbox_224(img, target_size=(224, 224)):
    """Resize with preserved aspect ratio and zero-padding (black edges)."""
    th, tw = target_size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def compute_tas_grade(imf):
    """
    Maps %IMF to TAS Beef Quality Grade (1 to 5)
    Thai Agricultural Standard TAS 6001-2014 mapping:
      Grade 1: Trace (< 20%)
      Grade 2: Slight (20% <= IMF < 30%)
      Grade 3: Moderate (30% <= IMF < 40%)
      Grade 4: Abundant (40% <= IMF < 50%)
      Grade 5: Prime (>= 50%)
    """
    if imf < 20.0:
        return 1
    elif imf < 30.0:
        return 2
    elif imf < 40.0:
        return 3
    elif imf < 50.0:
        return 4
    else:
        return 5

def process_muscle_roi(img_bgr, polygon_pts):
    if polygon_pts is None or len(polygon_pts) < 3:
        return None

    try:
        pts = np.asarray(polygon_pts, dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) < 3:
            return None
    except Exception:
        return None

    h, w = img_bgr.shape[:2]
    muscle_mask = np.zeros((h, w), dtype=np.uint8)
    try:
        cv2.fillPoly(muscle_mask, [pts], 255)
    except Exception:
        return None
    
    muscle_pixel_count = np.count_nonzero(muscle_mask)
    if muscle_pixel_count < 400: # filter out tiny noise fragments
        return None
        
    isolated_muscle = cv2.bitwise_and(img_bgr, img_bgr, mask=muscle_mask)
    
    # 1. Specular Glare Inpainting (S < 30, V > 220)
    hsv = cv2.cvtColor(isolated_muscle, cv2.COLOR_BGR2HSV)
    specular_mask = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 30, 255]))
    specular_mask = cv2.bitwise_and(specular_mask, specular_mask, mask=muscle_mask)
    glare_pixels = np.count_nonzero(specular_mask)
    
    if glare_pixels > 0:
        inpainted_bgr = cv2.inpaint(isolated_muscle, specular_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        inpainted_bgr = isolated_muscle.copy()
        
    # 2. CLAHE on L-channel (CIE L*a*b*)
    lab = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_chan)
    enhanced_lab = cv2.merge([enhanced_l, a_chan, b_chan])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_bgr = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=muscle_mask)
    
    # 3. %IMF via Otsu
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    roi_pixels = gray[muscle_mask == 255]
    if len(roi_pixels) == 0:
        return None
        
    otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fat_mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
    fat_mask = cv2.bitwise_and(fat_mask, fat_mask, mask=muscle_mask)
    fat_pixels = np.count_nonzero(fat_mask)
    
    imf_percent = (fat_pixels / muscle_pixel_count) * 100.0
    tas_grade = compute_tas_grade(imf_percent)
    
    # 4. Crop Bounding Box of Muscle
    x, y, bw, bh = cv2.boundingRect(pts)
    # small padding
    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    
    cropped_muscle = enhanced_bgr[y1:y2, x1:x2]
    if cropped_muscle.size == 0 or cropped_muscle.shape[0] < 5 or cropped_muscle.shape[1] < 5:
        return None

    # 5. Letterbox to 224x224
    patch_224 = letterbox_224(cropped_muscle, (224, 224))
    
    return {
        "patch_224": patch_224,
        "imf_percent": imf_percent,
        "tas_grade": tas_grade,
        "muscle_pixels": muscle_pixel_count,
        "fat_pixels": fat_pixels,
        "glare_pixels": glare_pixels
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))
    
    model_path = os.path.join(root_dir, "AI-Model", "v8-seg", "Version4", "runs", "segment", "ai_wagyusense_segment_v4", "weights", "best.pt")
    dataset_v4_dir = os.path.join(root_dir, "AI-Model", "v8-seg", "Version4")
    out_dataset_dir = os.path.join(base_dir, "dataset_roi")
    
    print("=" * 80)
    print("  AI-WagyuSense: Generating 224x224 Muscle ROI Dataset for Model 2")
    print("=" * 80)
    print(f"[*] YOLOv8-seg v4 Model : {model_path}")
    print(f"[*] Input Dataset Root  : {dataset_v4_dir}")
    print(f"[*] Output ROI Dataset  : {out_dataset_dir}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[*] Acceleration Device : {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    model = YOLO(model_path)
    
    splits = ["train", "valid", "test"]
    all_metadata = []
    
    total_rois = 0
    t_start = time.perf_counter()
    
    for split in splits:
        split_in_dir = os.path.join(dataset_v4_dir, split, "images")
        split_out_dir = os.path.join(out_dataset_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)
        
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp')
        img_paths = sorted([f for f in glob.glob(os.path.join(split_in_dir, "*.*")) if f.lower().endswith(valid_exts)])
        print(f"\n[*] Processing '{split}' split ({len(img_paths)} images)...")
        
        split_records = []
        for idx, path in enumerate(img_paths):
            filename = os.path.basename(path)
            stem = os.path.splitext(filename)[0]
            
            img_bgr = load_image_robust(path)
            if img_bgr is None:
                continue
                
            pred = model.predict(source=img_bgr, conf=0.30, imgsz=640, device=device, verbose=False)[0]
            if pred.masks is None or len(pred.masks) == 0:
                continue
                
            for inst_idx, poly in enumerate(pred.masks.xy):
                poly_pts = poly.astype(np.int32)
                conf = float(pred.boxes.conf[inst_idx].cpu().numpy())
                
                try:
                    res = process_muscle_roi(img_bgr, poly_pts)
                except Exception as ex:
                    continue
                if res is None:
                    continue
                    
                roi_name = f"{stem}_roi{inst_idx+1}.jpg"
                roi_path = os.path.join(split_out_dir, roi_name)
                write_image_unicode(roi_path, res["patch_224"])
                
                record = {
                    "roi_filename": roi_name,
                    "split": split,
                    "original_image": filename,
                    "conf": conf,
                    "tas_grade": res["tas_grade"],
                    "imf_percent": res["imf_percent"],
                    "muscle_pixels": res["muscle_pixels"],
                    "fat_pixels": res["fat_pixels"],
                    "glare_pixels": res["glare_pixels"],
                    "roi_path": roi_path
                }
                split_records.append(record)
                all_metadata.append(record)
                total_rois += 1
                
            if (idx + 1) % 150 == 0 or (idx + 1) == len(img_paths):
                print(f"    - Processed {idx+1}/{len(img_paths)} images -> Generated {len(split_records)} ROIs so far...")
                
        # Save split CSV
        split_csv = os.path.join(out_dataset_dir, f"metadata_{split}.csv")
        with open(split_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "roi_filename", "split", "original_image", "conf",
                "tas_grade", "imf_percent", "muscle_pixels", "fat_pixels", "glare_pixels", "roi_path"
            ])
            writer.writeheader()
            for r in split_records:
                writer.writerow({
                    "roi_filename": r["roi_filename"],
                    "split": r["split"],
                    "original_image": r["original_image"],
                    "conf": f"{r['conf']:.4f}",
                    "tas_grade": r["tas_grade"],
                    "imf_percent": f"{r['imf_percent']:.2f}",
                    "muscle_pixels": r["muscle_pixels"],
                    "fat_pixels": r["fat_pixels"],
                    "glare_pixels": r["glare_pixels"],
                    "roi_path": r["roi_path"]
                })
        print(f"    [+] Saved {len(split_records)} records to {split_csv}")
        
    t_end = time.perf_counter()
    
    # Save overall master metadata CSV
    master_csv = os.path.join(out_dataset_dir, "metadata_master.csv")
    with open(master_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "roi_filename", "split", "original_image", "conf",
            "tas_grade", "imf_percent", "muscle_pixels", "fat_pixels", "glare_pixels", "roi_path"
        ])
        writer.writeheader()
        for r in all_metadata:
            writer.writerow({
                "roi_filename": r["roi_filename"],
                "split": r["split"],
                "original_image": r["original_image"],
                "conf": f"{r['conf']:.4f}",
                "tas_grade": r["tas_grade"],
                "imf_percent": f"{r['imf_percent']:.2f}",
                "muscle_pixels": r["muscle_pixels"],
                "fat_pixels": r["fat_pixels"],
                "glare_pixels": r["glare_pixels"],
                "roi_path": r["roi_path"]
            })
            
    print("\n" + "=" * 80)
    print("                    ROI EXTRACTION SUMMARY (MODEL 2)                   ")
    print("=" * 80)
    print(f"Total Muscle ROIs Extracted : {total_rois}")
    print(f"Master Metadata CSV Saved   : {master_csv}")
    print(f"Total Elapsed Time          : {t_end - t_start:.2f} seconds")
    
    # Grade distribution
    grades = [r["tas_grade"] for r in all_metadata]
    print("\nTAS Quality Grade Distribution across Dataset:")
    for g in range(1, 6):
        cnt = grades.count(g)
        pct = (cnt / len(grades) * 100.0) if len(grades) > 0 else 0.0
        print(f"    - Grade {g} (มกษ. {g}): {cnt:4d} samples ({pct:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
