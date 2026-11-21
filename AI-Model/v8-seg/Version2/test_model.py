"""
AI-WagyuSense: Model Testing & Visual Montage Generator (Version 2)
Wat Khemaphirataram School - KIDE 2026 Project

Tests YOLOv8n-seg Version 2 model on 'valid' and 'test' splits,
executes the full Hybrid Computer Vision Pipeline, and generates 4-stage visual montages.
"""

import os
import time
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def read_image_unicode(file_path):
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def write_image_unicode(file_path, img):
    try:
        ext = os.path.splitext(file_path)[1]
        res, enc = cv2.imencode(ext, img)
        if res:
            enc.tofile(file_path)
            return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
    return False

def run_hybrid_pipeline(img_bgr, polygon_pts):
    h, w = img_bgr.shape[:2]
    muscle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(muscle_mask, [polygon_pts], 255)
    
    muscle_pixel_count = np.count_nonzero(muscle_mask)
    if muscle_pixel_count == 0:
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
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_bgr = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=muscle_mask)
    
    # 3. Mathematical Morphometry (%IMF via Otsu's Thresholding)
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    roi_pixels = gray[muscle_mask == 255]
    
    otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fat_mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
    fat_mask = cv2.bitwise_and(fat_mask, fat_mask, mask=muscle_mask)
    fat_pixels = np.count_nonzero(fat_mask)
    
    imf_percent = (fat_pixels / muscle_pixel_count) * 100.0
    
    # Visual Marbling Highlight (Gold/Green: B=0, G=230, R=255)
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
        "glare_pixels": glare_pixels,
        "otsu_thresh": otsu_thresh,
        "imf_percent": imf_percent
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "runs", "segment", "ai_wagyusense_segment_v2", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"[!] Model weights not found at: {model_path}")
        return
        
    print("=" * 80)
    print("  AI-WagyuSense: YOLOv8n-seg Version 2 Visual Test & Montage Generator  ")
    print("=" * 80)
    print(f"[*] Loading model from: {model_path}")
    model = YOLO(model_path)
    
    output_dir = os.path.join(base_dir, "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    splits_to_test = ["valid", "test"]
    all_reports = []
    
    for split_name in splits_to_test:
        split_dir = os.path.join(base_dir, split_name, "images")
        img_paths = sorted(glob.glob(os.path.join(split_dir, "*.*")))
        print(f"\n[*] Processing '{split_name}' split ({len(img_paths)} images)...")
        
        for idx, img_path in enumerate(img_paths):
            img_bgr = read_image_unicode(img_path)
            if img_bgr is None:
                continue
                
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            t0 = time.perf_counter()
            results = model.predict(source=img_bgr, conf=0.35, imgsz=640, verbose=False)[0]
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            
            if results.masks is None or len(results.masks) == 0:
                print(f"    [-] {split_name}-{idx+1:02d}: No muscle detected in {base_name[:20]}")
                continue
                
            # Select largest contour area
            areas = [cv2.contourArea(poly.astype(np.int32)) for poly in results.masks.xy]
            best_idx = int(np.argmax(areas))
            conf = float(results.boxes.conf[best_idx].cpu().numpy())
            polygon = results.masks.xy[best_idx].astype(np.int32)
            
            analysis = run_hybrid_pipeline(img_bgr, polygon)
            if analysis is None:
                continue
                
            imf = analysis["imf_percent"]
            glare_px = analysis["glare_pixels"]
            
            # --- Build 4-Panel Visual Montage ---
            h, w = img_bgr.shape[:2]
            
            # Panel 1: Original + YOLO Polygon & Confidence
            p1 = img_bgr.copy()
            cv2.polylines(p1, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(p1, f"YOLOv8-seg: {conf:.1%}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(p1, f"Time: {lat_ms:.1f}ms", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Panel 2: Isolated Muscle ROI (Subcutaneous fat & background trimmed)
            p2 = analysis["isolated_muscle"].copy()
            cv2.putText(p2, "1. Muscle ROI Isolated", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(p2, f"Area: {analysis['muscle_pixels']:,} px", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Panel 3: Inpainting + CLAHE
            p3 = analysis["enhanced_bgr"].copy()
            cv2.putText(p3, "2. Inpaint + CLAHE", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(p3, f"Glare Inpainted: {glare_px:,} px", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Panel 4: Otsu Marbling Segmentation (%IMF)
            p4 = analysis["fat_vis"].copy()
            cv2.putText(p4, f"3. Otsu Marbling: {imf:.2f}%", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(p4, f"Fat Pixels: {analysis['fat_pixels']:,} px", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
            
            # Combine 4 horizontal panels
            montage = np.hstack([p1, p2, p3, p4])
            montage_resized = cv2.resize(montage, (1600, 400))
            
            out_filename = f"montage_{split_name}_{idx+1:02d}_{base_name[:20]}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            write_image_unicode(out_path, montage_resized)
            
            all_reports.append({
                "split": split_name,
                "sample": f"{split_name}-{idx+1:02d}",
                "filename": base_name[:25],
                "conf": conf,
                "latency_ms": lat_ms,
                "muscle_px": analysis["muscle_pixels"],
                "fat_px": analysis["fat_pixels"],
                "imf": imf,
                "glare_px": glare_px,
                "out_path": out_path
            })
            print(f"    [+] {split_name}-{idx+1:02d}: Conf={conf:.1%}, %IMF={imf:.2f}%, Latency={lat_ms:.1f}ms -> Saved: {out_filename}")
            
    print("\n" + "=" * 90)
    print(f"Successfully generated {len(all_reports)} visual montages in: {output_dir}")
    print("=" * 90)
    
    # Save CSV Report
    csv_path = os.path.join(output_dir, "v2_test_summary_report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("split,sample,filename,confidence,latency_ms,muscle_pixels,fat_pixels,imf_percent,glare_pixels,montage_path\n")
        for r in all_reports:
            f.write(f"{r['split']},{r['sample']},{r['filename']},{r['conf']:.4f},{r['latency_ms']:.2f},{r['muscle_px']},{r['fat_px']},{r['imf']:.2f},{r['glare_px']},{r['out_path']}\n")
    print(f"[*] Summary CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
