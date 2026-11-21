"""
AI-WagyuSense: Comprehensive Evaluation on Test Dataset (Folder: รอทดสอบ)
Model: YOLOv8n-seg Version 4 (Dataset v4 trained)
Wat Khemaphirataram School - KIDE 2026 Project

Evaluates the trained YOLOv8n-seg v4 model on the test images in 'รอทดสอบ',
benchmarks inference latency on NVIDIA RTX 4060, and executes the full
Hybrid Computer Vision & Mathematical Morphometry Pipeline:
  1. YOLOv8n-seg v4 ROI Extraction (Longissimus dorsi)
  2. Specular Glare Inpainting (HSV S<30, V>220)
  3. CLAHE Contrast Enhancement
  4. Otsu's Thresholding & %IMF Calculation
"""

import os
import glob
import time
import re
import csv
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

def load_image_robust(path):
    """
    Robust image loader supporting Unicode Windows paths, JPG, PNG, WEBP, and AVIF.
    """
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
    """Safely write image with Unicode/Thai path on Windows."""
    try:
        ext = os.path.splitext(path)[1]
        success, enc = cv2.imencode(ext, img)
        if success:
            enc.tofile(path)
            return True
    except Exception as e:
        print(f"Error writing image {path}: {e}")
    return False

def sanitize_filename(name):
    """Sanitize filename for safe filesystem writing."""
    clean = re.sub(r'[^\w\.-]', '_', name)
    return clean[:40]

def run_hybrid_pipeline(img_bgr, polygon_pts):
    """
    Executes the Hybrid CV pipeline on the detected muscle polygon.
    """
    h, w = img_bgr.shape[:2]
    
    # 1. Muscle binary mask
    muscle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(muscle_mask, [polygon_pts], 255)
    
    muscle_pixel_count = np.count_nonzero(muscle_mask)
    if muscle_pixel_count == 0:
        return None
        
    isolated_muscle = cv2.bitwise_and(img_bgr, img_bgr, mask=muscle_mask)
    
    # 2. Specular Glare detection & inpainting (HSV)
    hsv = cv2.cvtColor(isolated_muscle, cv2.COLOR_BGR2HSV)
    specular_mask = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 30, 255]))
    specular_mask = cv2.bitwise_and(specular_mask, specular_mask, mask=muscle_mask)
    
    # Dilation to cover blurred glare boundaries (Penumbra)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    specular_mask_dilated = cv2.dilate(specular_mask, kernel, iterations=1)
    specular_mask_dilated = cv2.bitwise_and(specular_mask_dilated, specular_mask_dilated, mask=muscle_mask)
    specular_pixels = np.count_nonzero(specular_mask_dilated)
    
    if specular_pixels > 0:
        inpainted_bgr = cv2.inpaint(isolated_muscle, specular_mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        inpainted_bgr = isolated_muscle.copy()
        
    # 3. CLAHE Contrast Enhancement on L-Channel (CIE L*a*b*)
    lab = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_chan)
    enhanced_lab = cv2.merge([enhanced_l, a_chan, b_chan])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_bgr = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=muscle_mask)
    
    # 4. Otsu's Thresholding for Marbling & %IMF
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    roi_pixels = gray[muscle_mask == 255]
    
    otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fat_mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
    fat_mask = cv2.bitwise_and(fat_mask, fat_mask, mask=muscle_mask)
    fat_pixels = np.count_nonzero(fat_mask)
    
    imf_percent = (fat_pixels / muscle_pixel_count) * 100.0
    
    # 5. Visual Marbling Highlight (Gold/Yellow: B=0, G=230, R=255)
    fat_overlay = isolated_muscle.copy()
    fat_overlay[fat_mask == 255] = [0, 230, 255]
    fat_vis = cv2.addWeighted(isolated_muscle, 0.45, fat_overlay, 0.55, 0)
    
    return {
        "isolated_muscle": isolated_muscle,
        "inpainted": inpainted_bgr,
        "enhanced_bgr": enhanced_bgr,
        "fat_mask": fat_mask,
        "fat_vis": fat_vis,
        "muscle_pixels": muscle_pixel_count,
        "fat_pixels": fat_pixels,
        "glare_pixels": specular_pixels,
        "otsu_thresh": otsu_thresh,
        "imf_percent": imf_percent
    }

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "AI-Model", "v8-seg", "Version4", "runs", "segment", "ai_wagyusense_segment_v4", "weights", "best.pt")
    input_dir = os.path.join(root_dir, "AI-Model", "รอทดสอบ")
    output_dir = os.path.join(root_dir, "AI-Model", "ผลการทดสอบ_รอทดสอบ_v4")
    
    print("=" * 80)
    print("    AI-WagyuSense: Evaluation on New Test Photos with YOLOv8n-seg Version 4")
    print("=" * 80)
    print(f"[*] Model Weights : {model_path}")
    print(f"[*] Input Dir     : {input_dir}")
    print(f"[*] Output Dir    : {output_dir}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Model
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[*] Computing Device: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    model = YOLO(model_path)
    
    # Collect test images (excluding zip files)
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp')
    all_files = glob.glob(os.path.join(input_dir, "*.*"))
    img_paths = sorted([f for f in all_files if f.lower().endswith(valid_exts)])
    
    print(f"[*] Discovered {len(img_paths)} test image candidates in 'รอทดสอบ'.")
    print("-" * 80)
    
    results_records = []
    
    # Warmup GPU
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model.predict(source=dummy, imgsz=640, device=device, verbose=False)
    
    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]
        sanitized = sanitize_filename(stem)
        
        img_bgr = load_image_robust(path)
        if img_bgr is None:
            print(f"[-] Image {idx+1:02d}: Failed to decode {filename}")
            continue
            
        h, w = img_bgr.shape[:2]
        
        # Inference
        t0 = time.perf_counter()
        pred = model.predict(source=img_bgr, conf=0.25, imgsz=640, device=device, verbose=False)[0]
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        
        if pred.masks is None or len(pred.masks) == 0:
            print(f"[-] Image {idx+1:02d} [{filename[:20]}]: No muscle ROI detected (Conf < 0.25)")
            results_records.append({
                "index": idx + 1,
                "filename": filename,
                "detected": False,
                "confidence": 0.0,
                "instances_found": 0,
                "latency_ms": latency_ms,
                "muscle_pixels": 0,
                "fat_pixels": 0,
                "imf_percent": 0.0,
                "glare_pixels": 0,
                "montage_path": "N/A"
            })
            continue
            
        num_instances = len(pred.masks)
        # Select largest detected polygon
        areas = [cv2.contourArea(poly.astype(np.int32)) for poly in pred.masks.xy]
        best_inst_idx = int(np.argmax(areas))
        conf = float(pred.boxes.conf[best_inst_idx].cpu().numpy())
        polygon = pred.masks.xy[best_inst_idx].astype(np.int32)
        
        # Hybrid Pipeline
        analysis = run_hybrid_pipeline(img_bgr, polygon)
        if analysis is None:
            continue
            
        imf = analysis["imf_percent"]
        glare_px = analysis["glare_pixels"]
        muscle_px = analysis["muscle_pixels"]
        fat_px = analysis["fat_pixels"]
        
        # --- 4-Panel Diagnostic Montage ---
        p1 = img_bgr.copy()
        cv2.polylines(p1, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(p1, f"YOLOv8 v4: {conf:.1%}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(p1, f"Lat: {latency_ms:.1f}ms", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        p2 = analysis["isolated_muscle"].copy()
        cv2.putText(p2, "1. Meat ROI Isolated", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(p2, f"Muscle: {muscle_px:,} px", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        
        p3 = analysis["enhanced_bgr"].copy()
        cv2.putText(p3, "2. Inpaint + CLAHE", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(p3, f"Glare Inpainted: {glare_px:,} px", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        p4 = analysis["fat_vis"].copy()
        cv2.putText(p4, f"3. Otsu IMF: {imf:.2f}%", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(p4, f"Fat: {fat_px:,} px", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA)
        
        montage = np.hstack([p1, p2, p3, p4])
        montage_resized = cv2.resize(montage, (1600, 400))
        
        out_name = f"v4_eval_{idx+1:02d}_{sanitized}.jpg"
        out_path = os.path.join(output_dir, out_name)
        write_image_unicode(out_path, montage_resized)
        
        results_records.append({
            "index": idx + 1,
            "filename": filename,
            "detected": True,
            "confidence": conf,
            "instances_found": num_instances,
            "latency_ms": latency_ms,
            "muscle_pixels": muscle_px,
            "fat_pixels": fat_px,
            "imf_percent": imf,
            "glare_pixels": glare_px,
            "montage_path": out_path
        })
        
        print(f"[+] {idx+1:02d}/{len(img_paths)}: {filename[:22]:<22} | Conf: {conf:.1%} | %IMF: {imf:.2f}% | Lat: {latency_ms:.1f}ms | Instances: {num_instances}")
        
    print("-" * 80)
    
    # Save CSV Report
    csv_file = os.path.join(output_dir, "test_summary_v4_report.csv")
    fieldnames = [
        "index", "filename", "detected", "confidence", "instances_found",
        "latency_ms", "muscle_pixels", "fat_pixels", "imf_percent", "glare_pixels", "montage_path"
    ]
    
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in results_records:
            writer.writerow({
                "index": rec["index"],
                "filename": rec["filename"],
                "detected": rec["detected"],
                "confidence": f"{rec['confidence']:.4f}",
                "instances_found": rec["instances_found"],
                "latency_ms": f"{rec['latency_ms']:.2f}",
                "muscle_pixels": rec["muscle_pixels"],
                "fat_pixels": rec["fat_pixels"],
                "imf_percent": f"{rec['imf_percent']:.2f}",
                "glare_pixels": rec["glare_pixels"],
                "montage_path": rec["montage_path"]
            })
            
    print(f"[*] CSV Report successfully saved to: {csv_file}")
    
    # Statistical Summary
    detected_count = sum(1 for r in results_records if r["detected"])
    total_count = len(results_records)
    det_rate = (detected_count / total_count * 100.0) if total_count > 0 else 0.0
    
    confs = [r["confidence"] for r in results_records if r["detected"]]
    latencies = [r["latency_ms"] for r in results_records]
    imfs = [r["imf_percent"] for r in results_records if r["detected"]]
    
    print("\n" + "=" * 80)
    print("               BENCHMARK & EVALUATION SUMMARY (VERSION 4)              ")
    print("=" * 80)
    print(f"Total Images Tested     : {total_count}")
    print(f"Successful Detections   : {detected_count} ({det_rate:.1f}%)")
    if confs:
        print(f"Mean Confidence         : {np.mean(confs):.1%} (Min: {np.min(confs):.1%}, Max: {np.max(confs):.1%})")
        print(f"Mean Inference Latency  : {np.mean(latencies):.2f} ms (~{1000.0/np.mean(latencies):.1f} FPS)")
        print(f"Mean %IMF               : {np.mean(imfs):.2f}% (Min: {np.min(imfs):.2f}%, Max: {np.max(imfs):.2f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
