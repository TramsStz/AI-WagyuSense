"""
AI-WagyuSense: Comprehensive Evaluation on New Test Images (โฟลเดอร์ รอทดสอบ)
Wat Khemaphirataram School - KIDE 2026 Project

This script evaluates the trained YOLOv8n-seg model on 49 real test images,
benchmarks inference latency on NVIDIA RTX 4060, and executes the full
Hybrid Computer Vision & Mathematical Morphometry Pipeline:
  1. YOLOv8n-seg ROI Extraction (Longissimus dorsi)
  2. Specular Glare Inpainting (HSV S<30, V>220)
  3. CLAHE Contrast Enhancement
  4. Otsu's Thresholding & %IMF Calculation
"""

import os
import glob
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def read_image_unicode(path):
    """Safely read image with Unicode/Thai path on Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def write_image_unicode(path, img):
    """Safely write image with Unicode/Thai path on Windows."""
    ext = os.path.splitext(path)[1]
    success, enc = cv2.imencode(ext, img)
    if success:
        enc.tofile(path)
    return success

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
    
    # 2. Glare detection & inpainting (HSV)
    hsv = cv2.cvtColor(isolated_muscle, cv2.COLOR_BGR2HSV)
    specular_mask = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 30, 255]))
    specular_mask = cv2.bitwise_and(specular_mask, specular_mask, mask=muscle_mask)
    specular_pixels = np.count_nonzero(specular_mask)
    
    if specular_pixels > 0:
        inpainted_bgr = cv2.inpaint(isolated_muscle, specular_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        inpainted_bgr = isolated_muscle.copy()
        
    # 3. CLAHE Contrast Enhancement on L-Channel
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
    
    if len(roi_pixels) == 0:
        return None
        
    otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fat_mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
    fat_mask = cv2.bitwise_and(fat_mask, fat_mask, mask=muscle_mask)
    
    fat_pixel_count = np.count_nonzero(fat_mask)
    imf_percentage = (fat_pixel_count / muscle_pixel_count) * 100.0
    
    # Visual marbling overlay
    fat_vis = np.zeros_like(img_bgr)
    fat_vis[muscle_mask == 255] = [30, 30, 30] # dark background for lean muscle
    fat_vis[fat_mask == 255] = [0, 230, 255]   # bright yellow for marbling
    
    return {
        "muscle_pixels": muscle_pixel_count,
        "fat_pixels": fat_pixel_count,
        "imf_percentage": imf_percentage,
        "specular_pixels": specular_pixels,
        "isolated_muscle": isolated_muscle,
        "enhanced_bgr": enhanced_bgr,
        "fat_vis": fat_vis,
        "otsu_thresh": otsu_thresh
    }

def main():
    print("=" * 80)
    print("   AI-WagyuSense: High-Resolution Test Pipeline (Folder: รอทดสอบ)   ")
    print("=" * 80)
    
    weights_path = os.path.join("AI-Model", "v8-seg", "attemp1", "runs", "segment", "ai_wagyusense_segment", "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
        
    print(f"[*] Loading YOLOv8n-seg Model: {weights_path}")
    model = YOLO(weights_path)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Inference Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    input_dir = os.path.join("AI-Model", "รอทดสอบ")
    output_dir = os.path.join("AI-Model", "ผลการทดสอบ_รอทดสอบ")
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    print(f"[*] Found {len(img_files)} test images in '{input_dir}'.")
    if not img_files:
        print("[!] No images found to test.")
        return
        
    # GPU Warmup
    print("[*] Warming up GPU inference engine...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(dummy, imgsz=640, device=device, verbose=False)
        
    # Process Images
    print(f"[*] Processing {len(img_files)} images through End-to-End Hybrid AI Pipeline...\n")
    
    report_data = []
    latencies = []
    
    for idx, fpath in enumerate(img_files):
        fname = os.path.basename(fpath)
        img_bgr = read_image_unicode(fpath)
        if img_bgr is None:
            continue
            
        t0 = time.perf_counter()
        results = model.predict(img_bgr, imgsz=640, device=device, conf=0.30, verbose=False)[0]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        if len(results.boxes) == 0:
            print(f"[-] [{idx+1:02d}/{len(img_files)}] {fname}: No muscle detected!")
            continue
            
        # Select best detection (primary Longissimus dorsi steak)
        # Choose by confidence or area
        best_box_idx = 0
        best_conf = float(results.boxes.conf[0].cpu().numpy())
        
        # In case multiple detections exist, prioritize the largest bounding box area if conf > 0.4
        max_area = 0
        for i in range(len(results.boxes)):
            box = results.boxes.xyxy[i].cpu().numpy().astype(int)
            area = (box[2] - box[0]) * (box[3] - box[1])
            conf_i = float(results.boxes.conf[i].cpu().numpy())
            if conf_i >= 0.40 and area > max_area:
                max_area = area
                best_box_idx = i
                best_conf = conf_i
                
        polygon = results.masks.xy[best_box_idx].astype(np.int32)
        
        # Run Preprocessing & Morphometry
        analysis = run_hybrid_pipeline(img_bgr, polygon)
        if analysis is None:
            continue
            
        imf = analysis["imf_percentage"]
        glare = analysis["specular_pixels"]
        muscle_px = analysis["muscle_pixels"]
        fat_px = analysis["fat_pixels"]
        
        # Simple Thai Marbling Grade heuristic (Reference: %IMF bracket)
        # Class 1 (<5%), Class 2 (5-10%), Class 3 (10-15%), Class 4 (15-20%), Class 5 (>20%)
        if imf < 10.0:
            est_grade = "Grade 1-2"
        elif imf < 20.0:
            est_grade = "Grade 3"
        elif imf < 30.0:
            est_grade = "Grade 4"
        else:
            est_grade = "Grade 5 (Premium)"
            
        # Create 4-panel visual montage
        # 1. Original + Polygon Outline
        col1 = img_bgr.copy()
        cv2.polylines(col1, [polygon], isClosed=True, color=(0, 255, 255), thickness=3)
        cv2.putText(col1, f"{fname} | Conf: {best_conf:.1%}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                    
        # 2. Extracted Muscle ROI
        col2 = analysis["isolated_muscle"].copy()
        cv2.putText(col2, f"Muscle ROI ({muscle_px:,} px)", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
                    
        # 3. Specular Glare Inpaint + CLAHE
        col3 = analysis["enhanced_bgr"].copy()
        cv2.putText(col3, f"CLAHE (Inpaint: {glare:,} px)", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
                    
        # 4. Otsu Marbling Mask & %IMF
        col4 = analysis["fat_vis"].copy()
        cv2.putText(col4, f"%IMF: {imf:.2f}% [{est_grade}]", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 255), 3, cv2.LINE_AA)
                    
        # Resize each to width 800, height 600 for high clarity
        h_target, w_target = 600, 800
        p1 = cv2.resize(col1, (w_target, h_target))
        p2 = cv2.resize(col2, (w_target, h_target))
        p3 = cv2.resize(col3, (w_target, h_target))
        p4 = cv2.resize(col4, (w_target, h_target))
        
        # 2x2 grid montage:
        # [Panel 1] [Panel 2]
        # [Panel 3] [Panel 4]
        row_top = np.hstack([p1, p2])
        row_bot = np.hstack([p3, p4])
        grid_montage = np.vstack([row_top, row_bot])
        
        out_name = f"result_{fname}"
        out_path = os.path.join(output_dir, out_name)
        write_image_unicode(out_path, grid_montage)
        
        report_data.append({
            "filename": fname,
            "confidence": best_conf,
            "muscle_pixels": muscle_px,
            "fat_pixels": fat_px,
            "imf_percentage": imf,
            "specular_pixels": glare,
            "grade": est_grade,
            "output_path": out_path
        })
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(img_files):
            print(f"[*] Processed {idx+1}/{len(img_files)} images...")
            
    # Speed statistics
    avg_lat = np.mean(latencies)
    fps = 1000.0 / avg_lat
    print("\n" + "=" * 80)
    print(f"[+] Performance Summary on {torch.cuda.get_device_name(0)}:")
    print(f"    - Total Images Tested: {len(report_data)} / {len(img_files)} (Detection Rate: {len(report_data)/len(img_files):.1%})")
    print(f"    - Average Latency    : {avg_lat:.2f} ms")
    print(f"    - Throughput         : {fps:.1f} FPS")
    print("=" * 80)
    
    # Save CSV Report
    csv_path = os.path.join(output_dir, "test_summary_report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Filename,Confidence,Muscle_Pixels,Fat_Pixels,IMF_Percentage,Specular_Glare_Pixels,Estimated_Grade,Output_Image\n")
        for r in report_data:
            f.write(f"{r['filename']},{r['confidence']:.4f},{r['muscle_pixels']},{r['fat_pixels']},{r['imf_percentage']:.2f},{r['specular_pixels']},{r['grade']},{r['output_path']}\n")
    print(f"[*] Full CSV Report saved to: {csv_path}")
    
    # Print sample summary table
    print("\n" + "-" * 90)
    print(f"{'Filename':<12} | {'YOLO Conf':<10} | {'Muscle Area':<14} | {'Fat Area':<12} | {'%IMF':<8} | {'Est. Grade':<18}")
    print("-" * 90)
    for r in report_data[:15]:
        print(f"{r['filename']:<12} | {r['confidence']:>8.1%} | {r['muscle_pixels']:>11,} px | {r['fat_pixels']:>9,} px | {r['imf_percentage']:>6.2f}% | {r['grade']:<18}")
    if len(report_data) > 15:
        print(f"... and {len(report_data) - 15} more images (see CSV file for all).")
    print("-" * 90)

if __name__ == "__main__":
    main()
