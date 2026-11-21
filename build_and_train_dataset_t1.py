"""
AI-WagyuSense: Automated Dataset T1 Processing & Model 2 Fine-Tuning Pipeline
Wat Khemaphirataram School - KIDE 2026 Project

Executes the full automated workflow on the user-provided zip dataset (T1):
  1. Automated ROI Extraction using YOLOv8n-seg Version 4
  2. Calibrated Preprocessing (Glare S<20, V>235 + Telea + CLAHE clip=1.8)
  3. Morphometric %IMF Calculation via Clamped Otsu (safe_thresh >= 140)
  4. Complete Fat Dispersion Analysis (Particles, Fineness, Grid SD/CoV)
  5. TAS Quality Grade Labeling (Grade 1 - 5)
  6. Standardized Letterbox Padding (224x224)
  7. Train/Valid Split (80/20) and CSV Metadata Generation
  8. Fine-tuning Dual-Head EfficientNet-B0 & ONNX Export
"""

import os
import sys
import glob
import time
import math
import csv
import random
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# Add model2 directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
model2_dir = os.path.join(base_dir, "AI-Model", "model2-efficientnet")
sys.path.append(model2_dir)
from model2_architecture import DualHeadEfficientNetB0

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
        print(f"Error loading {path}: {e}")
        return None

def write_image_unicode(path, img):
    try:
        ext = os.path.splitext(path)[1]
        success, enc = cv2.imencode(ext, img)
        if success:
            enc.tofile(path)
            return True
    except Exception as e:
        print(f"Error writing {path}: {e}")
    return False

def sanitize_filename(name):
    clean = re.sub(r'[^\w\.-]', '_', name)
    return clean[:45]

def letterbox_224(img, target_size=(224, 224)):
    """Resize preserving aspect ratio with zero-padding (black borders)."""
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

def clean_muscle_mask(raw_mask):
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.zeros_like(opened)
        clean[labels == largest_label] = 255
    else:
        clean = opened
    k_rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.erode(clean, k_rim, iterations=1)
    return clean

def analyze_fat_dispersion(fat_mask, muscle_mask, grid_size=(3, 3)):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fat_mask, connectivity=8)
    valid_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 2]
    
    particle_count = len(valid_areas)
    mean_particle_area = float(np.mean(valid_areas)) if particle_count > 0 else 0.0
    max_particle_area = int(np.max(valid_areas)) if particle_count > 0 else 0
    total_muscle_px = int(np.count_nonzero(muscle_mask))
    fineness_density = (particle_count / total_muscle_px) * 1000.0 if total_muscle_px > 0 else 0.0
    
    x, y, w, h = cv2.boundingRect(muscle_mask)
    rows, cols = grid_size
    cell_w, cell_h = w / cols, h / rows
    
    cell_imfs = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(x + c * cell_w), int(y + r * cell_h)
            x2, y2 = int(x + (c + 1) * cell_w), int(y + (r + 1) * cell_h)
            c_muscle = np.count_nonzero(muscle_mask[y1:y2, x1:x2])
            if c_muscle > (total_muscle_px * 0.02):
                c_fat = np.count_nonzero(fat_mask[y1:y2, x1:x2])
                cell_imfs.append((c_fat / c_muscle) * 100.0)
                
    spatial_sd = float(np.std(cell_imfs)) if len(cell_imfs) > 1 else 0.0
    mean_imf = float(np.mean(cell_imfs)) if len(cell_imfs) > 0 else 0.0
    spatial_cov = (spatial_sd / (mean_imf + 1e-5)) * 100.0
    
    return {
        "particle_count": particle_count,
        "mean_area": round(mean_particle_area, 2),
        "max_area": max_particle_area,
        "fineness_density": round(fineness_density, 2),
        "spatial_sd": round(spatial_sd, 2),
        "spatial_cov": round(spatial_cov, 2)
    }

class BeefROIDataset(Dataset):
    def __init__(self, records, is_train=True):
        self.samples = records
        self.is_train = is_train
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["roi_path"]).convert("RGB")
        tensor_img = self.transform(img)
        grade_label = torch.tensor(int(item["tas_grade"]) - 1, dtype=torch.long)
        imf_label = torch.tensor(float(item["imf_percent"]), dtype=torch.float32)
        return tensor_img, grade_label, imf_label

def step1_build_dataset(yolo_model_path, input_dir, output_dataset_dir):
    print("=" * 80)
    print(" STEP 1: AUTOMATED DATASET GENERATION FROM EXTRACTED ZIP (T1)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing YOLOv8n-seg on {device}...")
    model = YOLO(yolo_model_path)
    
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model.predict(source=dummy, imgsz=640, device=device, verbose=False)
    
    images_dir = os.path.join(output_dataset_dir, "images")
    diagnostics_dir = os.path.join(output_dataset_dir, "diagnostics")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp')
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    all_files = sorted(all_files)
    
    print(f"[*] Discovered {len(all_files)} images in T1.")
    records = []
    diag_count = 0
    
    for idx, path in enumerate(all_files):
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]
        sanitized = sanitize_filename(stem)
        
        img_bgr = load_image_robust(path)
        if img_bgr is None:
            continue
            
        h, w = img_bgr.shape[:2]
        pred = model.predict(source=img_bgr, conf=0.25, imgsz=640, device=device, verbose=False)[0]
        if pred.masks is None or len(pred.masks) == 0:
            print(f"[-] [{idx+1:03d}/{len(all_files)}] No muscle detected: {fname[:25]}")
            continue
            
        # Select largest muscle instance
        areas = [cv2.contourArea(poly.astype(np.int32)) for poly in pred.masks.xy]
        best_idx = int(np.argmax(areas))
        polygon = pred.masks.xy[best_idx].astype(np.int32)
        conf = float(pred.boxes.conf[best_idx].cpu().numpy())
        
        # 1. Binary Mask & Cleanup
        raw_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(raw_mask, [polygon], 255)
        clean_mask = clean_muscle_mask(raw_mask)
        m_px = int(np.count_nonzero(clean_mask))
        if m_px < 500: # discard noise
            continue
            
        isolated = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)
        
        # 2. Glare Inpainting (S < 20, V > 235)
        hsv = cv2.cvtColor(isolated, cv2.COLOR_BGR2HSV)
        glare = cv2.inRange(hsv, np.array([0, 0, 235]), np.array([180, 20, 255]))
        glare = cv2.bitwise_and(glare, glare, mask=clean_mask)
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        glare = cv2.dilate(glare, k_dilate, iterations=1)
        glare_px = int(np.count_nonzero(glare))
        
        inpainted = cv2.inpaint(isolated, glare, 3, cv2.INPAINT_TELEA) if glare_px > 0 else isolated
        
        # 3. Green Channel & CLAHE (clip=1.8, grid=8x8)
        green = inpainted[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        enh = clahe.apply(green)
        enh = cv2.bitwise_and(enh, enh, mask=clean_mask)
        
        # 4. Clamped Otsu Thresholding (safe_thresh >= 140)
        roi_px = enh[clean_mask == 255]
        otsu_th, _ = cv2.threshold(roi_px, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        safe_th = max(int(otsu_th), 140)
        
        _, fat_mask = cv2.threshold(enh, safe_th, 255, cv2.THRESH_BINARY)
        fat_mask = cv2.bitwise_and(fat_mask, fat_mask, mask=clean_mask)
        fat_px = int(np.count_nonzero(fat_mask))
        imf = (fat_px / m_px) * 100.0 if m_px > 0 else 0.0
        tas_grade = compute_tas_grade(imf)
        
        # 5. Fat Dispersion
        disp = analyze_fat_dispersion(fat_mask, clean_mask, grid_size=(3, 3))
        
        # 6. Bounding Box & Letterbox 224x224
        x, y, bw, bh = cv2.boundingRect(clean_mask)
        pad = 4
        crop = inpainted[max(0, y-pad):min(h, y+bh+pad), max(0, x-pad):min(w, x+bw+pad)]
        patch_224 = letterbox_224(crop, (224, 224))
        
        # Save patch
        roi_filename = f"t1_{idx+1:03d}_{sanitized}_roi.jpg"
        roi_path = os.path.join(images_dir, roi_filename)
        write_image_unicode(roi_path, patch_224)
        
        records.append({
            "roi_filename": roi_filename,
            "roi_path": roi_path,
            "original_filename": fname,
            "conf": round(conf, 4),
            "tas_grade": tas_grade,
            "imf_percent": round(imf, 2),
            "muscle_pixels": m_px,
            "fat_pixels": fat_px,
            "glare_pixels": glare_px,
            "particle_count": disp["particle_count"],
            "fineness_density": disp["fineness_density"],
            "spatial_sd": disp["spatial_sd"],
            "spatial_cov": disp["spatial_cov"]
        })
        
        # Save sample diagnostic dashboards (first 10)
        if diag_count < 10:
            diag_count += 1
            p1 = img_bgr.copy()
            clean_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p1, clean_contours, -1, (0, 255, 255), 2)
            cv2.putText(p1, f"YOLOv8: {conf:.1%}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            p2 = inpainted.copy()
            cv2.putText(p2, f"Glare: {glare_px}px", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            p3 = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
            cv2.putText(p3, f"CLAHE Th:{safe_th}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
            
            fat_overlay = isolated.copy()
            fat_overlay[fat_mask == 255] = [0, 230, 255]
            p4 = cv2.addWeighted(isolated, 0.45, fat_overlay, 0.55, 0)
            cv2.putText(p4, f"TAS Grade {tas_grade} ({imf:.1f}%)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)
            
            p5 = cv2.resize(patch_224, (p1.shape[1], p1.shape[0]))
            cv2.putText(p5, "Patch 224x224", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            montage = np.hstack([p1, p2, p3, p4, p5])
            montage_resized = cv2.resize(montage, (1920, 384))
            diag_path = os.path.join(diagnostics_dir, f"diag_t1_{idx+1:03d}_{sanitized}.jpg")
            write_image_unicode(diag_path, montage_resized)

        print(f"[+] [{idx+1:03d}/{len(all_files)}] Processed: {fname[:20]:<20} | Conf: {conf:.1%} | TAS Grade: {tas_grade} | %IMF: {imf:5.1f}% | Particles: {disp['particle_count']:3d} | Fineness: {disp['fineness_density']:4.1f}/k")

    print(f"\n[+] Total successfully extracted ROIs: {len(records)} / {len(all_files)} ({len(records)/len(all_files)*100:.1f}%)")
    
    # 7. Split Train (80%) and Valid (20%)
    random.seed(42)
    random.shuffle(records)
    split_idx = int(len(records) * 0.80)
    train_records = records[:split_idx]
    valid_records = records[split_idx:]
    
    for r in train_records:
        r["split"] = "train"
    for r in valid_records:
        r["split"] = "valid"
        
    master_records = train_records + valid_records
    
    # Write CSVs
    fieldnames = [
        "roi_filename", "split", "original_filename", "conf", "tas_grade",
        "imf_percent", "muscle_pixels", "fat_pixels", "glare_pixels",
        "particle_count", "fineness_density", "spatial_sd", "spatial_cov", "roi_path"
    ]
    
    def write_csv(csv_path, data):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
                
    master_csv = os.path.join(output_dataset_dir, "metadata_T1_master.csv")
    train_csv = os.path.join(output_dataset_dir, "metadata_T1_train.csv")
    valid_csv = os.path.join(output_dataset_dir, "metadata_T1_valid.csv")
    
    write_csv(master_csv, master_records)
    write_csv(train_csv, train_records)
    write_csv(valid_csv, valid_records)
    
    print(f"[*] CSVs saved:")
    print(f"    - Master : {master_csv} ({len(master_records)} samples)")
    print(f"    - Train  : {train_csv} ({len(train_records)} samples)")
    print(f"    - Valid  : {valid_csv} ({len(valid_records)} samples)")
    
    # Print Grade distribution
    grades = [r["tas_grade"] for r in master_records]
    print("\n[*] TAS Grade Distribution:")
    for g in range(1, 6):
        cnt = grades.count(g)
        pct = (cnt / len(grades) * 100) if len(grades) > 0 else 0
        print(f"    - Grade {g}: {cnt:3d} samples ({pct:5.1f}%)")
        
    return master_records, train_records, valid_records

def step2_train_model2(train_records, valid_records, save_dir, epochs=25):
    print("\n" + "=" * 80)
    print(" STEP 2: FINE-TUNING DUAL-HEAD EFFICIENTNET-B0 ON DATASET T1")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
    train_dataset = BeefROIDataset(train_records, is_train=True)
    valid_dataset = BeefROIDataset(valid_records, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Load pretrained model
    model = DualHeadEfficientNetB0(num_classes=5, pretrained=True).to(device)
    
    # Losses: Classification (CrossEntropy) + Regression (SmoothL1)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float("inf")
    best_weights_path = os.path.join(save_dir, "best_model2_efficientnet_T1.pth")
    
    print(f"[*] Training for {epochs} epochs...")
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^9} | {'Train MAE':^9} | {'Val Loss':^9} | {'Val Acc':^8} | {'Val MAE':^8} | {'LR':^9}")
    print("-" * 80)
    
    lambda_reg = 0.05
    
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_cls, tr_reg, tr_correct, tr_total = 0.0, 0.0, 0.0, 0, 0
        tr_errors = []
        
        for imgs, g_labels, imf_labels in train_loader:
            imgs = imgs.to(device)
            g_labels = g_labels.to(device)
            imf_labels = imf_labels.to(device)
            
            optimizer.zero_grad()
            logits, pred_imf = model(imgs)
            
            loss_c = criterion_cls(logits, g_labels)
            loss_r = criterion_reg(pred_imf, imf_labels)
            loss = loss_c + (lambda_reg * loss_r)
            
            loss.backward()
            optimizer.step()
            
            bs = imgs.size(0)
            tr_loss += loss.item() * bs
            preds = torch.argmax(logits, dim=1)
            tr_correct += (preds == g_labels).sum().item()
            tr_total += bs
            tr_errors.extend(torch.abs(pred_imf - imf_labels).detach().cpu().numpy())
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_errors = []
        with torch.no_grad():
            for imgs, g_labels, imf_labels in valid_loader:
                imgs = imgs.to(device)
                g_labels = g_labels.to(device)
                imf_labels = imf_labels.to(device)
                
                logits, pred_imf = model(imgs)
                loss_c = criterion_cls(logits, g_labels)
                loss_r = criterion_reg(pred_imf, imf_labels)
                loss = loss_c + (lambda_reg * loss_r)
                
                bs = imgs.size(0)
                val_loss += loss.item() * bs
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == g_labels).sum().item()
                val_total += bs
                val_errors.extend(torch.abs(pred_imf - imf_labels).cpu().numpy())
                
        tr_l = tr_loss / tr_total
        tr_a = tr_correct / tr_total
        tr_m = np.mean(tr_errors)
        
        vl_l = val_loss / val_total
        vl_a = val_correct / val_total
        vl_m = np.mean(val_errors)
        lr_curr = optimizer.param_groups[0]["lr"]
        
        is_best = vl_l < best_val_loss
        if is_best:
            best_val_loss = vl_l
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": vl_l,
                "val_acc": vl_a,
                "val_mae": vl_m
            }, best_weights_path)
            mark = "★ BEST"
        else:
            mark = ""
            
        print(f"{epoch:^7d} | {tr_l:^10.4f} | {tr_a*100:^8.1f}% | {tr_m:^9.2f} | {vl_l:^9.4f} | {vl_a*100:^7.1f}% | {vl_m:^8.2f} | {lr_curr:^9.2e} {mark}")
        
    print("=" * 80)
    print(f"[+] Best Model weights saved to: {best_weights_path}")
    
    # Export to ONNX
    onnx_path = os.path.join(save_dir, "model2_dualhead_T1.onnx")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["grade_logits", "imf_percent"],
        dynamic_axes={"input": {0: "batch_size"}, "grade_logits": {0: "batch_size"}, "imf_percent": {0: "batch_size"}}
    )
    print(f"[+] Exported ONNX model to: {onnx_path}")
    return best_weights_path, onnx_path

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    yolo_model_path = os.path.join(root, "AI-Model", "v8-seg", "Version4", "runs", "segment", "ai_wagyusense_segment_v4", "weights", "best.pt")
    input_dir = os.path.join(root, "AI-Model", "dataset_T1_raw", "T1")
    output_dataset_dir = os.path.join(root, "AI-Model", "model2-efficientnet", "dataset_T1")
    
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    # 1. Build Dataset
    master, train_rec, val_rec = step1_build_dataset(yolo_model_path, input_dir, output_dataset_dir)
    
    # 2. Train Model 2
    best_weights, onnx_path = step2_train_model2(train_rec, val_rec, output_dataset_dir, epochs=25)
    
    print("\n" + "=" * 80)
    print("                    WORKFLOW COMPLETED SUCCESSFULLY!                  ")
    print("=" * 80)
    print(f"[*] Dataset T1 Directory : {output_dataset_dir}")
    print(f"[*] Trained Weights       : {best_weights}")
    print(f"[*] ONNX Model            : {onnx_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
