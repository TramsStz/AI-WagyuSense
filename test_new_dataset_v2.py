"""
AI-WagyuSense: Comprehensive Evaluation on New Test Dataset (Folder: รอทดสอบ)
Model: YOLOv8n-seg Version 2 (Dataset v2 trained)
Wat Khemaphirataram School - KIDE 2026 Project

Evaluates the trained YOLOv8n-seg v2 model on 73 new test images,
benchmarks inference latency on NVIDIA RTX 4060, and executes the full
Hybrid Computer Vision & Mathematical Morphometry Pipeline:
  1. YOLOv8n-seg v2 ROI Extraction (Longissimus dorsi)
  2. Specular Glare Inpainting (HSV S<30, V>220)
  3. CLAHE Contrast Enhancement
  4. Otsu's Thresholding & %IMF Calculation
"""

import os
import glob
import time
import re
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
    specular_pixels = np.count_nonzero(specular_mask)
    
    if specular_pixels > 0:
        inpainted_bgr = cv2.inpaint(isolated_muscle, specular_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
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
    
    if len(roi_pixels) == 0:
        return None
        
    otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fat_mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
    fat_mask = cv2.bitwise_and(fat_mask, fat_mask, mask=muscle_mask)
    
    fat_pixel_count = np.count_nonzero(fat_mask)
    imf_percentage = (fat_pixel_count / muscle_pixel_count) * 100.0
    
    # Visual marbling overlay
    fat_vis = isolated_muscle.copy()
    fat_vis[fat_mask == 255] = [0, 230, 255] # Gold-yellow for marbling
    fat_overlay = cv2.addWeighted(isolated_muscle, 0.40, fat_vis, 0.60, 0)
    
    return {
        "muscle_pixels": muscle_pixel_count,
        "fat_pixels": fat_pixel_count,
        "imf_percentage": imf_percentage,
        "specular_pixels": specular_pixels,
        "isolated_muscle": isolated_muscle,
        "enhanced_bgr": enhanced_bgr,
        "fat_overlay": fat_overlay,
        "otsu_thresh": otsu_thresh
    }

def main():
    print("=" * 85)
    print("   AI-WagyuSense: High-Resolution Test Pipeline with YOLOv8n-seg Version 2   ")
    print("=" * 85)
    
    weights_path = os.path.join("AI-Model", "v8-seg", "Version2", "runs", "segment", "ai_wagyusense_segment_v2", "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
    print(f"[*] Loading YOLOv8n-seg v2 Model: {weights_path}")
    model = YOLO(weights_path)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Inference Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    input_dir = os.path.join("AI-Model", "รอทดสอบ")
    output_dir = os.path.join("AI-Model", "ผลการทดสอบ_รอทดสอบ")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all image files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}
    all_files = sorted(os.listdir(input_dir))
    img_files = [os.path.join(input_dir, f) for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"[*] Found {len(img_files)} test images in '{input_dir}'.")
    if not img_files:
        print("[!] No valid images found to test.")
        return
        
    # GPU Warmup
    print("[*] Warming up GPU engine...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(dummy, imgsz=640, device=device, verbose=False)
        
    print(f"[*] Running Full Hybrid AI Pipeline on {len(img_files)} test images...\n")
    
    report_rows = []
    latencies = []
    detected_count = 0
    
    for idx, fpath in enumerate(img_files):
        fname = os.path.basename(fpath)
        img_bgr = load_image_robust(fpath)
        if img_bgr is None:
            print(f"[-] [{idx+1:02d}/{len(img_files)}] {fname}: Failed to load image!")
            continue
            
        h, w = img_bgr.shape[:2]
        
        # Inference with conf=0.25 (and fallback to 0.15 if nothing detected)
        t0 = time.perf_counter()
        results = model.predict(img_bgr, imgsz=640, device=device, conf=0.25, verbose=False)[0]
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)
        
        has_detection = results.masks is not None and len(results.masks) > 0
        
        if not has_detection:
            # Try with lower conf=0.15
            results = model.predict(img_bgr, imgsz=640, device=device, conf=0.15, verbose=False)[0]
            has_detection = results.masks is not None and len(results.masks) > 0
            
        if not has_detection:
            print(f"[-] [{idx+1:02d}/{len(img_files)}] {fname[:25]}: No muscle detected ({w}x{h}, {lat_ms:.1f}ms)")
            # Create a simple failure montage
            col1 = img_bgr.copy()
            cv2.putText(col1, f"No Detection | {fname[:20]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            col2 = np.zeros_like(col1)
            cv2.putText(col2, "No Longissimus dorsi muscle detected", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            montage = np.hstack([col1, col2])
            montage_resized = cv2.resize(montage, (1200, 400))
            safe_name = sanitize_filename(fname)
            out_path = os.path.join(output_dir, f"test_{idx+1:02d}_NODETECT_{safe_name}.jpg")
            write_image_unicode(out_path, montage_resized)
            
            report_rows.append({
                "idx": idx + 1,
                "filename": fname,
                "status": "No Detection",
                "resolution": f"{w}x{h}",
                "confidence": 0.0,
                "latency_ms": lat_ms,
                "muscle_pixels": 0,
                "fat_pixels": 0,
                "imf_percentage": 0.0,
                "specular_pixels": 0,
                "grade": "N/A",
                "montage_path": out_path
            })
            continue
            
        # Prioritize largest contour area
        areas = [cv2.contourArea(poly.astype(np.int32)) for poly in results.masks.xy]
        best_idx = int(np.argmax(areas))
        best_conf = float(results.boxes.conf[best_idx].cpu().numpy())
        polygon = results.masks.xy[best_idx].astype(np.int32)
        
        # Execute Hybrid Pipeline
        analysis = run_hybrid_pipeline(img_bgr, polygon)
        if analysis is None:
            continue
            
        detected_count += 1
        imf = analysis["imf_percentage"]
        glare = analysis["specular_pixels"]
        muscle_px = analysis["muscle_pixels"]
        fat_px = analysis["fat_pixels"]
        
        # Thai beef marbling score bracket (TAS standard reference)
        if imf < 10.0:
            est_grade = "Grade 1-2 (Lean)"
        elif imf < 20.0:
            est_grade = "Grade 3 (Slight)"
        elif imf < 30.0:
            est_grade = "Grade 4 (Modest)"
        elif imf < 40.0:
            est_grade = "Grade 5 (Moderate)"
        else:
            est_grade = "Grade 5+ (Abundant)"
            
        # Build 4-column visual montage
        # Panel 1: Original + Polygon Contour
        p1 = img_bgr.copy()
        cv2.polylines(p1, [polygon], isClosed=True, color=(0, 255, 255), thickness=3)
        cv2.putText(p1, f"YOLOv8-seg v2: {best_conf:.1%}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(p1, f"Latency: {lat_ms:.1f}ms", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                    
        # Panel 2: Isolated Muscle ROI (Subcutaneous fat removed)
        p2 = analysis["isolated_muscle"].copy()
        cv2.putText(p2, "1. Muscle ROI Isolated", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(p2, f"Area: {muscle_px:,} px", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
                    
        # Panel 3: Inpainting + CLAHE
        p3 = analysis["enhanced_bgr"].copy()
        cv2.putText(p3, "2. Inpaint + CLAHE", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(p3, f"Glare Inpainted: {glare:,} px", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                    
        # Panel 4: Otsu Marbling Segmentation (%IMF)
        p4 = analysis["fat_overlay"].copy()
        cv2.putText(p4, f"3. Otsu Marbling: {imf:.2f}%", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(p4, f"{est_grade}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
                    
        montage = np.hstack([p1, p2, p3, p4])
        montage_resized = cv2.resize(montage, (1600, 400))
        
        safe_name = sanitize_filename(fname)
        out_path = os.path.join(output_dir, f"test_{idx+1:02d}_{safe_name}.jpg")
        write_image_unicode(out_path, montage_resized)
        
        report_rows.append({
            "idx": idx + 1,
            "filename": fname,
            "status": "Detected",
            "resolution": f"{w}x{h}",
            "confidence": best_conf,
            "latency_ms": lat_ms,
            "muscle_pixels": muscle_px,
            "fat_pixels": fat_px,
            "imf_percentage": imf,
            "specular_pixels": glare,
            "grade": est_grade,
            "montage_path": out_path
        })
        
        print(f"[+] [{idx+1:02d}/{len(img_files)}] {fname[:25]:<25} | Conf: {best_conf:>5.1%} | %IMF: {imf:>5.2f}% | Lat: {lat_ms:>4.1f}ms | {est_grade}")
        
    print("\n" + "=" * 85)
    print("                    EVALUATION SUMMARY REPORT                        ")
    print("=" * 85)
    print(f"[*] Total Test Images Processed   : {len(img_files)}")
    print(f"[*] Successful Detections         : {detected_count} / {len(img_files)} ({detected_count / len(img_files):.1%})")
    
    if latencies:
        avg_lat = np.mean(latencies)
        print(f"[*] Average Latency (RTX 4060)   : {avg_lat:.2f} ms (~{1000.0 / avg_lat:.1f} FPS)")
        print(f"[*] Latency Range                : {np.min(latencies):.2f} ms - {np.max(latencies):.2f} ms")
        
    detected_rows = [r for r in report_rows if r["status"] == "Detected"]
    if detected_rows:
        confs = [r["confidence"] for r in detected_rows]
        imfs = [r["imf_percentage"] for r in detected_rows]
        print(f"[*] Mean Confidence Score        : {np.mean(confs):.1%} (Min: {np.min(confs):.1%}, Max: {np.max(confs):.1%})")
        print(f"[*] Mean %IMF (Marbling Ratio)   : {np.mean(imfs):.2f}% (Min: {np.min(imfs):.2f}%, Max: {np.max(imfs):.2f}%)")
        
    print(f"[*] Visual Pipeline Montages     : Saved {len(report_rows)} files in '{output_dir}'")
    
    # Save CSV Report
    csv_file = os.path.join(output_dir, "test_summary_v2_report.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("idx,filename,status,resolution,confidence,latency_ms,muscle_pixels,fat_pixels,imf_percentage,specular_pixels,grade,montage_path\n")
        for r in report_rows:
            f.write(f"{r['idx']},{r['filename']},{r['status']},{r['resolution']},{r['confidence']:.4f},{r['latency_ms']:.2f},{r['muscle_pixels']},{r['fat_pixels']},{r['imf_percentage']:.2f},{r['specular_pixels']},{r['grade']},{r['montage_path']}\n")
    print(f"[*] CSV Quantitative Report      : {csv_file}")
    print("=" * 85)

if __name__ == "__main__":
    main()
