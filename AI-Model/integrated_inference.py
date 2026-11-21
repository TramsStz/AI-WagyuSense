"""
AI-WagyuSense: Official Production Pipeline (YOLOv8n-seg + Mathematical Morphometry)
Wat Khemaphirataram School - KIDE 2026 Project

Core Architecture:
  1. Semantic Segmentation: YOLOv8n-seg Version 4 (Longissimus dorsi ROI)
  2. Calibrated Preprocessing: Glare Inpainting (Telea, S<20, V>235) + Green Channel + CLAHE
  3. Mathematical Morphometry:
     - Exact %IMF via Calibrated Clamped Otsu (safe_thresh >= 140)
     - Connected Components Fat Particle Analysis (Count, Mean Area, Fineness Density)
     - Spatial Dispersion & Uniformity Index (3x3 Grid SD & CoV)
     - Meat Color Spectrophotometry (CIE L*a*b* Lightness & Redness)
     - Deterministic TAS Quality Grading (TAS 6001-2014 Grade 1 - 5)
  4. Real-time Synthesis: High-Resolution Explainable Diagnostic Dashboard (1600x900)
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

def load_image_robust(path):
    """Safely load image supporting Unicode/Thai Windows paths and all standard formats."""
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
    """Safely write image with Unicode/Thai path on Windows."""
    try:
        ext = os.path.splitext(path)[1]
        success, enc = cv2.imencode(ext, img)
        if success:
            enc.tofile(path)
            return True
    except Exception as e:
        print(f"[-] Error writing {path}: {e}")
    return False

class AIWagyuSensePipeline:
    def __init__(self, seg_model_path, cls_model_path=None, device="cuda"):
        """
        Initializes AI-WagyuSense Production Pipeline.
        cls_model_path is maintained for backward compatibility, but the primary
        decision and grading engine operates on 100% Mathematical Morphometry.
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print("=" * 80)
        print("   AI-WAGYUSENSE: OFFICIAL PRODUCTION PIPELINE (YOLOv8 + MATH MORPHOMETRY)")
        print("=" * 80)
        print(f"[*] Compute Device : {torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'}")
        print(f"[*] Loading Model  : {seg_model_path}")
        
        self.seg_model = YOLO(seg_model_path)
        
        # Warmup GPU
        dummy_seg = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.seg_model.predict(source=dummy_seg, imgsz=640, device=self.device, verbose=False)
        print("[+] YOLOv8n-seg Segmentation Engine initialized and warmed up successfully!")

    def clean_muscle_mask(self, raw_mask):
        """Morphological cleanup: opening (5x5) + largest component + 2px rim erosion."""
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

    def analyze_fat_dispersion(self, fat_mask, muscle_mask, grid_size=(3, 3)):
        """
        Extracts structural marbling metrics:
          - Particle count, mean area, max area
          - Fineness density (particles / 1,000 px)
          - Spatial Dispersion: Grid SD & CoV across 3x3 local regions
          - Local cell IMF grid for heatmap visualization
        """
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
        cell_data = []
        for r in range(rows):
            for c in range(cols):
                x1, y1 = int(x + c * cell_w), int(y + r * cell_h)
                x2, y2 = int(x + (c + 1) * cell_w), int(y + (r + 1) * cell_h)
                c_muscle = np.count_nonzero(muscle_mask[y1:y2, x1:x2])
                c_fat = np.count_nonzero(fat_mask[y1:y2, x1:x2])
                c_imf = (c_fat / c_muscle * 100.0) if c_muscle > (total_muscle_px * 0.02) else 0.0
                if c_muscle > (total_muscle_px * 0.02):
                    cell_imfs.append(c_imf)
                cell_data.append({"bbox": (x1, y1, x2, y2), "imf": c_imf, "active": c_muscle > (total_muscle_px * 0.02)})
                
        spatial_sd = float(np.std(cell_imfs)) if len(cell_imfs) > 1 else 0.0
        mean_imf = float(np.mean(cell_imfs)) if len(cell_imfs) > 0 else 0.0
        spatial_cov = (spatial_sd / (mean_imf + 1e-5)) * 100.0
        
        # Qualitative dispersion rating
        if spatial_cov <= 12.0:
            uniformity_rating = "Excellent (Highly Uniform)"
        elif spatial_cov <= 25.0:
            uniformity_rating = "Good (Well Dispersed)"
        elif spatial_cov <= 40.0:
            uniformity_rating = "Moderate"
        else:
            uniformity_rating = "Irregular / Clustered"
            
        return {
            "particle_count": particle_count,
            "mean_area": round(mean_particle_area, 2),
            "max_area": max_particle_area,
            "fineness_density": round(fineness_density, 2),
            "spatial_sd": round(spatial_sd, 2),
            "spatial_cov": round(spatial_cov, 2),
            "uniformity_rating": uniformity_rating,
            "grid_cells": cell_data
        }

    def compute_deterministic_tas_grade(self, imf_percent, spatial_cov=None):
        """
        Deterministic Rule-based TAS Beef Quality Grade (มกษ. 6001-2014):
          Grade 1: Trace (< 20%)
          Grade 2: Slight (20% <= IMF < 30%)
          Grade 3: Moderate (30% <= IMF < 40%)
          Grade 4: Abundant (40% <= IMF < 50%)
          Grade 5: Prime / Wagyu (>= 50%)
        """
        if imf_percent < 20.0:
            grade = 1
            grade_name = "Grade 1 (Trace Marbling)"
        elif imf_percent < 30.0:
            grade = 2
            grade_name = "Grade 2 (Slight Marbling)"
        elif imf_percent < 40.0:
            grade = 3
            grade_name = "Grade 3 (Moderate Marbling)"
        elif imf_percent < 50.0:
            grade = 4
            grade_name = "Grade 4 (Abundant Marbling)"
        else:
            grade = 5
            grade_name = "Grade 5 (Prime / Wagyu A5)"
            
        return grade, grade_name

    def measure_meat_color(self, inpainted_bgr, muscle_mask, fat_mask):
        """Measures CIE L*a*b* of pure lean meat (excluding fat and background)."""
        lean_mask = cv2.bitwise_and(muscle_mask, cv2.bitwise_not(fat_mask))
        if np.count_nonzero(lean_mask) == 0:
            return {"L": 0.0, "a": 0.0, "b": 0.0, "color_score": "Normal"}
            
        lab = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2LAB)
        l_vals = lab[:, :, 0][lean_mask == 255]
        a_vals = lab[:, :, 1][lean_mask == 255]
        b_vals = lab[:, :, 2][lean_mask == 255]
        
        mean_l = float(np.mean(l_vals))
        mean_a = float(np.mean(a_vals))
        mean_b = float(np.mean(b_vals))
        
        # Qualitative Beef Color Score (BCS 1-7 estimate)
        if mean_l > 120 and mean_a > 145:
            color_desc = "Cherry Red (Fresh & Bright)"
        elif mean_l < 80:
            color_desc = "Dark Red (Firm & Dry - DFD risk)"
        else:
            color_desc = "Standard Beef Red"
            
        return {
            "L": round(mean_l, 1),
            "a": round(mean_a, 1),
            "b": round(mean_b, 1),
            "color_desc": color_desc
        }

    def process_image(self, img_bgr, fast_mode=False):
        h, w = img_bgr.shape[:2]
        
        # =============================================================
        # Stage 1: Muscle ROI Detection & Segmentation (YOLOv8n-seg)
        # =============================================================
        t0 = time.perf_counter()
        pred = self.seg_model.predict(source=img_bgr, conf=0.25, imgsz=640, device=self.device, verbose=False)[0]
        t1 = time.perf_counter()
        seg_lat_ms = (t1 - t0) * 1000.0
        
        if pred.masks is None or len(pred.masks) == 0:
            return None, "No muscle detected"
            
        # Select primary muscle instance (largest area)
        areas = [cv2.contourArea(poly.astype(np.int32)) for poly in pred.masks.xy]
        best_idx = int(np.argmax(areas))
        polygon = pred.masks.xy[best_idx].astype(np.int32)
        seg_conf = float(pred.boxes.conf[best_idx].cpu().numpy())
        
        # =============================================================
        # Stage 2: Calibrated Image Preprocessing (OpenCV)
        # =============================================================
        t2 = time.perf_counter()
        raw_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(raw_mask, [polygon], 255)
        
        # Clean muscle mask
        muscle_mask = self.clean_muscle_mask(raw_mask)
        muscle_px = int(np.count_nonzero(muscle_mask))
        if muscle_px == 0:
            return None, "Invalid muscle mask"
            
        # FAST STREAMING MODE (Optimized for Live Camera on Jetson)
        # Skips heavy Telea inpainting and 1600x900 dashboard rendering for real-time 30+ FPS HUD
        if fast_mode:
            green = img_bgr[:, :, 1]
            roi_pixels = green[muscle_mask == 255]
            if len(roi_pixels) > 0:
                otsu_th, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                safe_th = max(int(otsu_th), 140)
                _, fat_mask = cv2.threshold(green, safe_th, 255, cv2.THRESH_BINARY)
                fat_mask = cv2.bitwise_and(fat_mask, fat_mask, mask=muscle_mask)
                fat_px = int(np.count_nonzero(fat_mask))
                otsu_imf = (fat_px / muscle_px) * 100.0
            else:
                otsu_imf = 0.0
                fat_mask = np.zeros_like(muscle_mask)
                fat_px = 0
                safe_th = 140
                
            tas_grade, grade_desc = self.compute_deterministic_tas_grade(otsu_imf)
            t_end = time.perf_counter()
            
            return None, {
                "tas_grade": tas_grade,
                "grade_desc": grade_desc,
                "otsu_imf_percent": otsu_imf,
                "particle_count": 0,
                "fineness_density": 0.0,
                "spatial_sd": 0.0,
                "spatial_cov": 0.0,
                "uniformity_rating": "Live Stream",
                "seg_confidence": seg_conf,
                "threshold_used": safe_th,
                "muscle_pixels": muscle_px,
                "fat_pixels": fat_px,
                "glare_pixels": 0,
                "polygon": polygon,
                "fat_mask": fat_mask,
                "muscle_mask": muscle_mask,
                "latency_breakdown_ms": {
                    "segmentation": seg_lat_ms,
                    "total": (t_end - t0) * 1000.0
                }
            }

        isolated = cv2.bitwise_and(img_bgr, img_bgr, mask=muscle_mask)
        
        # Glare Inpainting (Tightened S<20, V>235 + Dilation + Telea)
        hsv = cv2.cvtColor(isolated, cv2.COLOR_BGR2HSV)
        glare_mask = cv2.inRange(hsv, np.array([0, 0, 235]), np.array([180, 20, 255]))
        glare_mask = cv2.bitwise_and(glare_mask, glare_mask, mask=muscle_mask)
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        glare_mask_dilated = cv2.dilate(glare_mask, k_dilate, iterations=1)
        glare_mask_dilated = cv2.bitwise_and(glare_mask_dilated, glare_mask_dilated, mask=muscle_mask)
        glare_px = int(np.count_nonzero(glare_mask_dilated))
        
        if glare_px > 0:
            inpainted = cv2.inpaint(isolated, glare_mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        else:
            inpainted = isolated.copy()
            
        # Green Channel + CLAHE (clipLimit=1.8, tileGridSize=(8, 8))
        green = inpainted[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        enhanced_green = clahe.apply(green)
        enhanced_green = cv2.bitwise_and(enhanced_green, enhanced_green, mask=muscle_mask)
        t3 = time.perf_counter()
        prep_lat_ms = (t3 - t2) * 1000.0
        
        # =============================================================
        # Stage 3: Mathematical Morphometry & Deterministic Grading
        # =============================================================
        t4 = time.perf_counter()
        roi_pixels = enhanced_green[muscle_mask == 255]
        otsu_th, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        safe_th = max(int(otsu_th), 140)
        
        _, fat_mask = cv2.threshold(enhanced_green, safe_th, 255, cv2.THRESH_BINARY)
        fat_mask = cv2.bitwise_and(fat_mask, fat_mask, mask=muscle_mask)
        fat_px = int(np.count_nonzero(fat_mask))
        
        otsu_imf = (fat_px / muscle_px) * 100.0 if muscle_px > 0 else 0.0
        
        # Fat Dispersion & Structural Particle Analysis
        disp = self.analyze_fat_dispersion(fat_mask, muscle_mask, grid_size=(3, 3))
        
        # Meat Color Analysis
        color_info = self.measure_meat_color(inpainted, muscle_mask, fat_mask)
        
        # Deterministic TAS Beef Quality Grade
        tas_grade, grade_desc = self.compute_deterministic_tas_grade(otsu_imf, disp["spatial_cov"])
        
        t5 = time.perf_counter()
        morph_lat_ms = (t5 - t4) * 1000.0
        total_lat_ms = seg_lat_ms + prep_lat_ms + morph_lat_ms
        
        # Visual Overlays
        # Panel 1: Original image + Muscle Contour
        p1 = img_bgr.copy()
        clean_contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p1, clean_contours, -1, (0, 255, 255), 2)
        cv2.putText(p1, f"YOLOv8-seg ({seg_conf:.1%})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        
        # Panel 2: Isolated Muscle + Glare Highlights
        p2 = inpainted.copy()
        if glare_px > 0:
            glare_cnts, _ = cv2.findContours(glare_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p2, glare_cnts, -1, (255, 255, 0), 1)
        cv2.putText(p2, f"Glare Removed: {glare_px:,} px", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 2)
        
        # Panel 3: Marbling Highlight Overlay (Gold/Yellow: B=0, G=230, R=255)
        fat_overlay = isolated.copy()
        fat_overlay[fat_mask == 255] = [0, 230, 255]
        p3 = cv2.addWeighted(isolated, 0.40, fat_overlay, 0.60, 0)
        cv2.putText(p3, f"Marbling: {otsu_imf:.2f}% IMF", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)
        
        # Panel 4: Spatial Dispersion Grid (3x3)
        p4 = isolated.copy()
        for cell in disp["grid_cells"]:
            x1, y1, x2, y2 = cell["bbox"]
            color = (0, 255, 120) if cell["active"] else (60, 60, 60)
            cv2.rectangle(p4, (x1, y1), (x2, y2), color, 1)
            if cell["active"]:
                cv2.putText(p4, f"{cell['imf']:.1f}%", (x1 + 6, y1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(p4, f"Dispersion CoV: {disp['spatial_cov']:.1f}%", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 120), 2)
        
        # =============================================================
        # Stage 4: Synthesize 1600x900 Diagnostic Dashboard
        # =============================================================
        dashboard = self._render_dashboard(
            p1, p2, p3, p4,
            tas_grade, grade_desc, otsu_imf, disp, color_info,
            seg_lat_ms, prep_lat_ms, morph_lat_ms, total_lat_ms,
            muscle_px, fat_px, glare_px, safe_th
        )
        
        results = {
            "tas_grade": tas_grade,
            "grade_desc": grade_desc,
            "otsu_imf_percent": otsu_imf,
            "particle_count": disp["particle_count"],
            "fineness_density": disp["fineness_density"],
            "spatial_sd": disp["spatial_sd"],
            "spatial_cov": disp["spatial_cov"],
            "uniformity_rating": disp["uniformity_rating"],
            "color_info": color_info,
            "seg_confidence": seg_conf,
            "threshold_used": safe_th,
            "muscle_pixels": muscle_px,
            "fat_pixels": fat_px,
            "glare_pixels": glare_px,
            "latency_breakdown_ms": {
                "segmentation": seg_lat_ms,
                "preprocessing": prep_lat_ms,
                "morphometry": morph_lat_ms,
                "total": total_lat_ms
            }
        }
        
        return dashboard, results

    def _render_dashboard(self, p1, p2, p3, p4, grade, grade_desc, imf, disp, color_info, lat_seg, lat_prep, lat_morph, lat_total, muscle_px, fat_px, glare_px, safe_th):
        canvas = np.zeros((900, 1600, 3), dtype=np.uint8)
        canvas[:] = (20, 24, 30) # Navy Dark Slate (#1E1814 BGR)
        
        # Header Bar
        cv2.rectangle(canvas, (0, 0), (1600, 70), (10, 15, 20), -1)
        cv2.putText(canvas, "AI-WagyuSense: High-Precision Beef Morphometry System (KIDE 2026)", (30, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.92, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Physics-Informed Computer Vision | Wat Khemaphirataram School", (1000, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)
        
        # 4 Left/Middle Diagnostic Panels (480x360 each in 2x2 grid)
        p1_r = cv2.resize(p1, (480, 360))
        p2_r = cv2.resize(p2, (480, 360))
        p3_r = cv2.resize(p3, (480, 360))
        p4_r = cv2.resize(p4, (480, 360))
        
        canvas[90:450, 30:510] = p1_r
        canvas[90:450, 540:1020] = p2_r
        canvas[490:850, 30:510] = p3_r
        canvas[490:850, 540:1020] = p4_r
        
        # Sub-labels for visual panels
        cv2.putText(canvas, "[A] YOLOv8n-seg v4: Muscle ROI Extraction", (30, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[B] Preprocessing: Glare Inpainted (Telea) & CLAHE", (540, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[C] Marbling Metrology: Pixel-by-Pixel %IMF Mask", (30, 875), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[D] Spatial Dispersion: 3x3 Regional Uniformity Heatmap", (540, 875), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)
        
        # Right Assessment Card (1050 to 1570, 90 to 865)
        card_x1, card_y1, card_x2, card_y2 = 1050, 90, 1570, 865
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (32, 38, 48), -1)
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (65, 80, 100), 1)
        
        # Title of Card
        cv2.putText(canvas, "BEEF QUALITY CERTIFICATE", (card_x1 + 25, card_y1 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Deterministic Standard Grading (มกษ. 6001-2014)", (card_x1 + 25, card_y1 + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.line(canvas, (card_x1 + 25, card_y1 + 75), (card_x2 - 25, card_y1 + 75), (65, 80, 100), 1)
        
        # TAS Grade Badge
        grade_colors = {
            1: (100, 100, 100), # Gray
            2: (180, 140, 50),  # Bronze
            3: (200, 200, 200), # Silver
            4: (0, 215, 255),   # Gold
            5: (0, 255, 180)    # Platinum/Prime
        }
        badge_color = grade_colors.get(grade, (0, 229, 255))
        
        cv2.rectangle(canvas, (card_x1 + 25, card_y1 + 95), (card_x1 + 480, card_y1 + 185), (42, 52, 68), -1)
        cv2.rectangle(canvas, (card_x1 + 25, card_y1 + 95), (card_x1 + 480, card_y1 + 185), badge_color, 2)
        cv2.putText(canvas, f"GRADE {grade}", (card_x1 + 45, card_y1 + 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, badge_color, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"{imf:.2f}% IMF", (card_x1 + 310, card_y1 + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, grade_desc, (card_x1 + 25, card_y1 + 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1, cv2.LINE_AA)
        
        # Section 1: Quantitative Morphometrics
        cv2.line(canvas, (card_x1 + 25, card_y1 + 225), (card_x2 - 25, card_y1 + 225), (65, 80, 100), 1)
        cv2.putText(canvas, "MARBLING MORPHOMETRY (PHYSICAL METRICS)", (card_x1 + 25, card_y1 + 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 255), 1, cv2.LINE_AA)
        
        row_y = card_y1 + 280
        metrics = [
            ("Total Marbling %IMF", f"{imf:.2f}%", (0, 230, 255)),
            ("Particle Count", f"{disp['particle_count']:,} islands", (220, 220, 220)),
            ("Fineness Density", f"{disp['fineness_density']:.1f} / 1k px", (0, 255, 120)),
            ("Max Clump Area", f"{disp['max_area']:,} px", (200, 200, 200)),
            ("Dispersion CoV", f"{disp['spatial_cov']:.1f}% (SD: {disp['spatial_sd']:.1f})", (220, 220, 220)),
            ("Uniformity Quality", disp["uniformity_rating"], (0, 229, 255))
        ]
        
        for label, val, col in metrics:
            cv2.putText(canvas, f"{label:<20} :", (card_x1 + 25, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
            cv2.putText(canvas, val, (card_x1 + 245, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
            row_y += 24
            
        # Section 2: Physiological Lean Color
        cv2.line(canvas, (card_x1 + 25, row_y + 5), (card_x2 - 25, row_y + 5), (65, 80, 100), 1)
        row_y += 28
        cv2.putText(canvas, "LEAN MEAT COLOR (SPECTROPHOTOMETRY)", (card_x1 + 25, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 255), 1, cv2.LINE_AA)
        row_y += 26
        cv2.putText(canvas, f"Lightness (L*) : {color_info['L']}   |   Redness (a*) : {color_info['a']}", (card_x1 + 25, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        row_y += 22
        cv2.putText(canvas, f"Visual Color    : {color_info['color_desc']}", (card_x1 + 25, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 160, 160), 1)
        
        # Section 3: Physical Pixel Audit (XAI / Explainability)
        cv2.line(canvas, (card_x1 + 25, row_y + 8), (card_x2 - 25, row_y + 8), (65, 80, 100), 1)
        row_y += 30
        cv2.putText(canvas, "AUDITABLE GROUND TRUTH (WHITE-BOX)", (card_x1 + 25, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 255), 1, cv2.LINE_AA)
        row_y += 25
        cv2.putText(canvas, f"Muscle Area : {muscle_px:,} px   |   Fat Area : {fat_px:,} px", (card_x1 + 25, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 190, 190), 1)
        row_y += 22
        cv2.putText(canvas, f"Glare Corr. : {glare_px:,} px   |   Threshold : {safe_th} / 255", (card_x1 + 25, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 190, 190), 1)
        
        # Section 4: Real-time Edge Benchmarks
        cv2.line(canvas, (card_x1 + 25, row_y + 10), (card_x2 - 25, row_y + 10), (65, 80, 100), 1)
        row_y += 32
        cv2.putText(canvas, f"LATENCY: {lat_total:.1f} ms (~{1000.0/lat_total:.1f} FPS)", (card_x1 + 25, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)
        row_y += 22
        cv2.putText(canvas, f"YOLOv8: {lat_seg:.1f}ms  |  CV Prep: {lat_prep:.1f}ms  |  Morph: {lat_morph:.1f}ms", (card_x1 + 25, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
                    
        return canvas

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    seg_model_path = os.path.join(root_dir, "v8-seg", "Version4", "runs", "segment", "ai_wagyusense_segment_v4", "weights", "best.pt")
    input_dir = os.path.join(root_dir, "new_test_web_beef")
    output_dir = os.path.join(root_dir, "ผลการประเมิน_ระบบบูรณาการ_Production")
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = AIWagyuSensePipeline(seg_model_path)
    
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))])
    print(f"\n[*] Evaluating on {len(files)} test images in: {input_dir}")
    print("-" * 80)
    
    for idx, fname in enumerate(files):
        img_path = os.path.join(input_dir, fname)
        img = load_image_robust(img_path)
        if img is None:
            continue
            
        dashboard, rec = pipeline.process_image(img)
        if dashboard is None:
            print(f"[-] [{idx+1}/{len(files)}] {fname}: No muscle detected.")
            continue
            
        out_path = os.path.join(output_dir, f"dashboard_{idx+1:02d}_{os.path.splitext(fname)[0]}.jpg")
        write_image_unicode(out_path, dashboard)
        
        lats = rec["latency_breakdown_ms"]
        print(f"[+] [{idx+1:02d}/{len(files):02d}] {fname[:24]:<24} | Grade {rec['tas_grade']} | %IMF: {rec['otsu_imf_percent']:5.2f}% | Particles: {rec['particle_count']:4d} | Fineness: {rec['fineness_density']:4.1f}/k | CoV: {rec['spatial_cov']:4.1f}% | Lat: {lats['total']:4.1f}ms")
        
    print("=" * 80)
    print(f"[+] High-Precision Diagnostic Dashboards saved to:\n    {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
