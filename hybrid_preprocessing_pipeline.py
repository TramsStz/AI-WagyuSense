"""
AI-WagyuSense: Hybrid Preprocessing Pipeline v2 (Calibrated)
Wat Khemaphirataram School - KIDE 2026 Project

Upgraded Preprocessing Pipeline addressing the 4 key real-world challenges:
  1. Clamped Otsu Thresholding (safe_thresh = max(otsu, 140)) to prevent overestimation on lean meat.
  2. YOLO Mask Cleanup: Morphological Opening (5x5) + Largest Connected Component to remove stray spikes/tags.
  3. Tightened Specular Highlight Detection (S < 20, V > 235) to protect intense white marbling from inpainting.
  4. Local Contrast Equalization via CLAHE (clipLimit=1.8, tileGridSize=(8, 8)).
"""

import os
import glob
import time
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

def load_image_robust(path):
    """Safely load image supporting Unicode/Thai Windows paths and multiple formats."""
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
        print(f"[-] Error loading {path}: {e}")
        return None

def write_image_unicode(path, img):
    """Safely save image supporting Unicode/Thai Windows paths."""
    try:
        ext = os.path.splitext(path)[1]
        success, enc = cv2.imencode(ext, img)
        if success:
            enc.tofile(path)
            return True
    except Exception as e:
        print(f"[-] Error writing {path}: {e}")
    return False

class AIWagyuHybridPreprocessorV2:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"[*] Initializing YOLOv8n-seg Model on {self.device}...")
        self.model = YOLO(model_path)
        
        # Warmup GPU
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(source=dummy, imgsz=640, device=self.device, verbose=False)
        print("[+] YOLOv8n-seg model ready.")

    def detect_muscle_roi(self, img_bgr, conf_thresh=0.25):
        """Step 1 (YOLO): Detect Ribeye / Longissimus dorsi muscle ROI."""
        t0 = time.perf_counter()
        results = self.model.predict(source=img_bgr, conf=conf_thresh, imgsz=640, device=self.device, verbose=False)[0]
        yolo_latency_ms = (time.perf_counter() - t0) * 1000.0
        
        if results.masks is None or len(results.masks) == 0:
            return None, 0.0, yolo_latency_ms
            
        # Select instance with the largest contour area
        areas = [cv2.contourArea(poly.astype(np.int32)) for poly in results.masks.xy]
        best_idx = int(np.argmax(areas))
        polygon = results.masks.xy[best_idx].astype(np.int32)
        conf = float(results.boxes.conf[best_idx].cpu().numpy())
        
        return polygon, conf, yolo_latency_ms

    def clean_muscle_mask(self, raw_mask):
        """
        Improvement 2: Clean up mask artifacts:
          - Morphological Opening (5x5) to sever thin tails and spurs
          - Extract Largest Connected Component to drop isolated fragments
          - Subtle edge erosion (2px) to suppress subcutaneous rim fat bleed
        """
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean = np.zeros_like(opened)
            clean[labels == largest_label] = 255
        else:
            clean = opened
            
        # Subtle 2px erosion to prevent subcutaneous edge contamination
        k_rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean = cv2.erode(clean, k_rim, iterations=1)
        return clean

    def preprocess_and_calculate_imf(self, img_bgr, polygon, luminance_mode="green", min_fat_thresh=140):
        """
        Executes Step 2 (Calibrated OpenCV Preprocessing) and Step 3 (%IMF calculation).
        """
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        
        # 1. Raw mask from polygon & Post-processing cleanup
        raw_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(raw_mask, [polygon], 255)
        
        # Clean mask
        clean_mask = self.clean_muscle_mask(raw_mask)
        muscle_pixels = int(np.count_nonzero(clean_mask))
        if muscle_pixels == 0:
            return None
            
        isolated_muscle = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)
        
        # -------------------------------------------------------------
        # 2. Specular Highlight Removal (จุดที่ 3: ปรับเกณฑ์เฉพาะ Glare จริง)
        # -------------------------------------------------------------
        hsv = cv2.cvtColor(isolated_muscle, cv2.COLOR_BGR2HSV)
        
        # Tightened condition: S < 20 (strictly achromatic) and V > 235 (hotspot LED)
        lower_glare = np.array([0, 0, 235], dtype=np.uint8)
        upper_glare = np.array([180, 20, 255], dtype=np.uint8)
        raw_glare_mask = cv2.inRange(hsv, lower_glare, upper_glare)
        raw_glare_mask = cv2.bitwise_and(raw_glare_mask, raw_glare_mask, mask=clean_mask)
        raw_glare_pixels = int(np.count_nonzero(raw_glare_mask))
        
        # Morphological Dilation to cover penumbra
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_glare_mask = cv2.dilate(raw_glare_mask, k_dilate, iterations=1)
        dilated_glare_mask = cv2.bitwise_and(dilated_glare_mask, dilated_glare_mask, mask=clean_mask)
        dilated_glare_pixels = int(np.count_nonzero(dilated_glare_mask))
        
        # Telea Inpainting
        if dilated_glare_pixels > 0:
            inpainted_bgr = cv2.inpaint(
                isolated_muscle,
                dilated_glare_mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
        else:
            inpainted_bgr = isolated_muscle.copy()
            
        # -------------------------------------------------------------
        # 3. Luminance Extraction
        # -------------------------------------------------------------
        if luminance_mode == "green":
            luminance_raw = inpainted_bgr[:, :, 1] # Green Channel
            luminance_name = "Green Channel"
        else:
            lab = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2LAB)
            luminance_raw = lab[:, :, 0] # L* Channel
            luminance_name = "CIE L* (Lightness)"
            
        luminance_roi = cv2.bitwise_and(luminance_raw, luminance_raw, mask=clean_mask)
        
        # -------------------------------------------------------------
        # 4. CLAHE (clipLimit=1.8, tileGridSize=(8, 8))
        # -------------------------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        enhanced_luminance = clahe.apply(luminance_roi)
        enhanced_luminance = cv2.bitwise_and(enhanced_luminance, enhanced_luminance, mask=clean_mask)
        
        # -------------------------------------------------------------
        # 5. Clamped Otsu's Thresholding (จุดที่ 1: กำหนดเพดานขั้นต่ำ)
        # -------------------------------------------------------------
        roi_pixel_values = enhanced_luminance[clean_mask == 255]
        raw_otsu_thresh, _ = cv2.threshold(roi_pixel_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_otsu_val = int(raw_otsu_thresh)
        
        # Clamped Safe Threshold: safe_thresh = max(otsu, min_fat_thresh)
        safe_thresh = max(raw_otsu_val, min_fat_thresh)
        
        # Raw Otsu Fat Mask (for diagnostic comparison)
        _, raw_fat_mask = cv2.threshold(enhanced_luminance, raw_otsu_val, 255, cv2.THRESH_BINARY)
        raw_fat_mask = cv2.bitwise_and(raw_fat_mask, raw_fat_mask, mask=clean_mask)
        raw_imf_percent = (np.count_nonzero(raw_fat_mask) / muscle_pixels) * 100.0
        
        # Safe Clamped Fat Mask
        _, safe_fat_mask = cv2.threshold(enhanced_luminance, safe_thresh, 255, cv2.THRESH_BINARY)
        safe_fat_mask = cv2.bitwise_and(safe_fat_mask, safe_fat_mask, mask=clean_mask)
        safe_fat_pixels = int(np.count_nonzero(safe_fat_mask))
        safe_imf_percent = (safe_fat_pixels / muscle_pixels) * 100.0
        
        # -------------------------------------------------------------
        # 6. Fat Dispersion & Morphometry Analysis (ขั้นตอนที่ 3 ส่วนต่อขยาย)
        # -------------------------------------------------------------
        dispersion_metrics = self.analyze_fat_dispersion(safe_fat_mask, clean_mask, grid_size=(3, 3))
        
        cv_latency_ms = (time.perf_counter() - t0) * 1000.0
        
        # Fat overlay visualization (Gold/Yellow: [0, 230, 255])
        fat_overlay = isolated_muscle.copy()
        fat_overlay[safe_fat_mask == 255] = [0, 230, 255]
        fat_vis = cv2.addWeighted(isolated_muscle, 0.45, fat_overlay, 0.55, 0)
        
        return {
            "raw_mask": raw_mask,
            "clean_mask": clean_mask,
            "isolated_muscle": isolated_muscle,
            "dilated_glare_mask": dilated_glare_mask,
            "inpainted_bgr": inpainted_bgr,
            "luminance_raw": luminance_roi,
            "luminance_name": luminance_name,
            "enhanced_luminance": enhanced_luminance,
            "raw_fat_mask": raw_fat_mask,
            "safe_fat_mask": safe_fat_mask,
            "fat_vis": fat_vis,
            "raw_otsu_thresh": raw_otsu_val,
            "safe_thresh": safe_thresh,
            "muscle_pixels": muscle_pixels,
            "raw_glare_pixels": raw_glare_pixels,
            "dilated_glare_pixels": dilated_glare_pixels,
            "safe_fat_pixels": safe_fat_pixels,
            "raw_imf_percent": raw_imf_percent,
            "safe_imf_percent": safe_imf_percent,
            "dispersion": dispersion_metrics,
            "cv_latency_ms": cv_latency_ms
        }

    def analyze_fat_dispersion(self, fat_mask, muscle_mask, grid_size=(3, 3)):
        """
        Mathematical Morphometry: Fat Dispersion & Particle Characterization
        - Particle count, mean area, max area
        - Fineness density (particles per 1,000 px of muscle)
        - Spatial Dispersion Index (Grid SD & CoV across 3x3 local regions)
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fat_mask, connectivity=8)
        valid_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 2]
        
        particle_count = len(valid_areas)
        mean_particle_area = float(np.mean(valid_areas)) if particle_count > 0 else 0.0
        max_particle_area = int(np.max(valid_areas)) if particle_count > 0 else 0
        
        total_muscle_px = int(np.count_nonzero(muscle_mask))
        fineness_density = (particle_count / total_muscle_px) * 1000.0 if total_muscle_px > 0 else 0.0
        
        # Spatial Dispersion across M x N grid
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
                    c_imf = (c_fat / c_muscle) * 100.0
                    cell_imfs.append(c_imf)
                    
        spatial_sd = float(np.std(cell_imfs)) if len(cell_imfs) > 1 else 0.0
        mean_imf = float(np.mean(cell_imfs)) if len(cell_imfs) > 0 else 0.0
        spatial_cov = (spatial_sd / (mean_imf + 1e-5)) * 100.0
        
        return {
            "particle_count": particle_count,
            "mean_area": round(mean_particle_area, 2),
            "max_area": max_particle_area,
            "fineness_density": round(fineness_density, 2),
            "spatial_sd": round(spatial_sd, 2),
            "spatial_cov": round(spatial_cov, 2),
            "active_cells": len(cell_imfs)
        }

    def create_diagnostic_dashboard(self, img_bgr, polygon, conf, res, yolo_latency_ms):
        """
        Creates a clean 5-panel Diagnostic Montage (1920x384):
          Panel 1: Original + Clean YOLO Contour
          Panel 2: Glare Inpainted (Telea)
          Panel 3: Extracted Luminance (Green Channel)
          Panel 4: CLAHE Contrast Enhanced
          Panel 5: Calibrated %IMF Fat Overlay
        """
        # Panel 1: Original + YOLOv8 Clean Contour
        p1 = img_bgr.copy()
        clean_contours, _ = cv2.findContours(res["clean_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p1, clean_contours, -1, (0, 255, 255), 2)
        cv2.putText(p1, f"YOLOv8n-seg: {conf:.1%}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(p1, f"YOLO Lat: {yolo_latency_ms:.1f}ms", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    
        # Panel 2: Glare Detected & Inpainted
        p2 = res["inpainted_bgr"].copy()
        glare_contours, _ = cv2.findContours(res["dilated_glare_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(glare_contours) > 0:
            cv2.drawContours(p2, glare_contours, -1, (255, 255, 0), 1)
        cv2.putText(p2, "1. Glare Inpainted (Telea)", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(p2, f"Glare: {res['dilated_glare_pixels']:,} px (S<20, V>235)", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
                    
        # Panel 3: Extracted Luminance Channel
        p3 = cv2.cvtColor(res["luminance_raw"], cv2.COLOR_GRAY2BGR)
        cv2.putText(p3, f"2. {res['luminance_name']}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 100), 2, cv2.LINE_AA)
        cv2.putText(p3, "Luminance Extracted", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                    
        # Panel 4: CLAHE Contrast Enhanced
        p4 = cv2.cvtColor(res["enhanced_luminance"], cv2.COLOR_GRAY2BGR)
        cv2.putText(p4, "3. CLAHE Enhanced", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(p4, f"Otsu: {res['raw_otsu_thresh']} -> Clamped: {res['safe_thresh']}", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)
                    
        # Panel 5: Fat Overlay & %IMF & Dispersion
        p5 = res["fat_vis"].copy()
        disp = res["dispersion"]
        cv2.putText(p5, f"4. %IMF: {res['safe_imf_percent']:.2f}%", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(p5, f"Particles: {disp['particle_count']:,} | Fine: {disp['fineness_density']:.1f}/k", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(p5, f"Grid SD: {disp['spatial_sd']:.1f} (CoV: {disp['spatial_cov']:.1f}%)", (12, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA)

        # Combine into horizontal banner
        banner = np.hstack([p1, p2, p3, p4, p5])
        banner_resized = cv2.resize(banner, (1920, 384))
        return banner_resized

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "AI-Model", "v8-seg", "Version4", "runs", "segment", "ai_wagyusense_segment_v4", "weights", "best.pt")
    input_dir = os.path.join(root_dir, "AI-Model", "รอทดสอบ")
    output_dir = os.path.join(root_dir, "AI-Model", "ผลการทดสอบ_Hybrid_Preprocessing_v2")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 85)
    print("   AI-WagyuSense: Hybrid Preprocessing Pipeline v2 (Calibrated Clamped Otsu)")
    print("=" * 85)
    print(f"[*] Model Weights : {model_path}")
    print(f"[*] Input Dir     : {input_dir}")
    print(f"[*] Output Dir    : {output_dir}")
    
    pipeline = AIWagyuHybridPreprocessorV2(model_path)
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp')
    all_files = glob.glob(os.path.join(input_dir, "*.*"))
    img_paths = sorted([f for f in all_files if f.lower().endswith(valid_exts)])
    
    print(f"[*] Processing 8 representative test cases...")
    print("-" * 85)
    
    records = []
    
    for idx, path in enumerate(img_paths[:8]):
        fname = os.path.basename(path)
        img_bgr = load_image_robust(path)
        if img_bgr is None:
            continue
            
        polygon, conf, yolo_lat = pipeline.detect_muscle_roi(img_bgr, conf_thresh=0.25)
        if polygon is None:
            continue
            
        # Process with Calibrated Pipeline (Green Channel + Clamped Otsu 140)
        res = pipeline.preprocess_and_calculate_imf(img_bgr, polygon, luminance_mode="green", min_fat_thresh=140)
        
        # Diagnostic Montage
        montage = pipeline.create_diagnostic_dashboard(img_bgr, polygon, conf, res, yolo_lat)
        out_path = os.path.join(output_dir, f"diag_v2_{idx+1:02d}_{os.path.splitext(fname)[0]}.jpg")
        write_image_unicode(out_path, montage)
        
        total_lat = yolo_lat + res["cv_latency_ms"]
        disp = res["dispersion"]
        print(f"[+] Image {idx+1:02d}: {fname[:14]:<14} | %IMF: {res['safe_imf_percent']:5.2f}% | Particles: {disp['particle_count']:3d} | Fineness: {disp['fineness_density']:4.1f}/k | Grid SD: {disp['spatial_sd']:4.1f} (CoV: {disp['spatial_cov']:4.1f}%) | Lat: {total_lat:.1f}ms")
        
        records.append({
            "idx": idx + 1,
            "filename": fname,
            "conf": conf,
            "raw_otsu_th": res["raw_otsu_thresh"],
            "safe_th": res["safe_thresh"],
            "raw_imf": res["raw_imf_percent"],
            "calibrated_imf": res["safe_imf_percent"],
            "glare_px": res["dilated_glare_pixels"],
            "latency_ms": total_lat
        })
        
    print("=" * 85)
    print(f"[+] Successfully generated Calibrated Diagnostic Dashboards in:\n    {output_dir}")
    print("=" * 85)

if __name__ == "__main__":
    main()
