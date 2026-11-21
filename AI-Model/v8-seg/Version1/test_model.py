"""
AI-WagyuSense: Comprehensive Model Testing & Pipeline Verification Script
Wat Khemaphirataram School - KIDE 2026 Project

This script tests the trained YOLOv8n-seg model on validation images,
benchmarks inference latency/FPS on the current GPU/CPU, and executes 
the end-to-end Computer Vision Preprocessing & Morphometry Pipeline:
  1. YOLOv8n-seg ROI Extraction (Longissimus dorsi)
  2. Specular Reflection Removal (HSV Inpainting)
  3. CLAHE Contrast Enhancement
  4. Otsu's Thresholding & %IMF (Intramuscular Fat) Calculation
"""

import os
import time
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def run_hybrid_pipeline(img_bgr, polygon_pts):
    """
    Executes the Hybrid CV pipeline on the detected muscle polygon.
    """
    h, w = img_bgr.shape[:2]
    
    # 1. Create binary mask for the Longissimus dorsi muscle
    muscle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(muscle_mask, [polygon_pts], 255)
    
    # Muscle area in pixels
    muscle_pixel_count = np.count_nonzero(muscle_mask)
    if muscle_pixel_count == 0:
        return None
        
    # Mask out non-muscle areas (subcutaneous fat & background)
    isolated_muscle = cv2.bitwise_and(img_bgr, img_bgr, mask=muscle_mask)
    
    # 2. Highlight Removal via HSV Inpainting
    hsv = cv2.cvtColor(isolated_muscle, cv2.COLOR_BGR2HSV)
    # Highlight pixels: Low Saturation (< 30) and High Value (> 220)
    specular_mask = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 30, 255]))
    specular_mask = cv2.bitwise_and(specular_mask, specular_mask, mask=muscle_mask)
    
    # Inpaint specular glare
    if np.count_nonzero(specular_mask) > 0:
        inpainted_bgr = cv2.inpaint(isolated_muscle, specular_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        inpainted_bgr = isolated_muscle.copy()
        
    # 3. CLAHE Contrast Enhancement on L-Channel (CIE L*a*b*)
    lab = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_bgr = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=muscle_mask)
    
    # 4. Mathematical Morphometry (%IMF via Otsu's Thresholding)
    # Extract grayscale only within muscle ROI
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    roi_pixels = gray[muscle_mask == 255]
    
    # Compute Otsu's threshold on muscle ROI pixels
    otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Binary fat mask: pixels in muscle exceeding Otsu threshold are fat
    _, fat_mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
    fat_mask = cv2.bitwise_and(fat_mask, fat_mask, mask=muscle_mask)
    
    fat_pixel_count = np.count_nonzero(fat_mask)
    imf_percentage = (fat_pixel_count / muscle_pixel_count) * 100.0
    
    # Colored visualization of fat mask (marbling in bright yellow/cyan)
    fat_vis = np.zeros_like(img_bgr)
    fat_vis[muscle_mask == 255] = [30, 30, 30] # dark background for muscle
    fat_vis[fat_mask == 255] = [0, 230, 255]   # yellow marbling
    
    return {
        "muscle_pixel_count": muscle_pixel_count,
        "fat_pixel_count": fat_pixel_count,
        "imf_percentage": imf_percentage,
        "otsu_thresh": otsu_thresh,
        "isolated_muscle": isolated_muscle,
        "enhanced_bgr": enhanced_bgr,
        "fat_vis": fat_vis,
        "specular_pixels": np.count_nonzero(specular_mask)
    }

def main():
    print("=" * 75)
    print("      AI-WagyuSense: Model Testing & Hybrid AI Pipeline Benchmark      ")
    print("=" * 75)
    
    weights_path = os.path.join("runs", "segment", "ai_wagyusense_segment", "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    print(f"[*] Loading Trained Model: {weights_path}")
    model = YOLO(weights_path)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Inference Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    val_images = glob.glob(os.path.join("Dataset1", "valid", "images", "*.*"))
    val_images = [f for f in val_images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[*] Found {len(val_images)} validation images for testing.")
    
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Warm-up GPU
    print("\n[*] Warming up GPU inference engine (5 passes)...")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(dummy_img, imgsz=640, device=device, verbose=False)
        
    # 2. Benchmark Inference Latency
    print("[*] Running Speed Benchmark over validation set...")
    inference_times = []
    for img_path in val_images:
        t0 = time.perf_counter()
        _ = model.predict(img_path, imgsz=640, device=device, verbose=False)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000.0) # in ms
        
    avg_latency = np.mean(inference_times)
    std_latency = np.std(inference_times)
    fps = 1000.0 / avg_latency
    print(f"[+] Benchmark Results on {torch.cuda.get_device_name(0)}:")
    print(f"    - Average Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"    - Frame Rate     : {fps:.1f} FPS")
    print("-" * 75)

    # 3. Full Pipeline Execution on all Validation Samples
    print("[*] Executing Full Hybrid AI Pipeline on Validation Images...")
    report_rows = []
    
    for idx, img_path in enumerate(val_images):
        img_bgr = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Inference
        results = model.predict(img_bgr, imgsz=640, conf=0.25, device=device, verbose=False)[0]
        
        if results.masks is None or len(results.masks) == 0:
            print(f"[-] Sample {idx+1}: {base_name[:20]} -> No muscle detected!")
            continue
            
        # Get highest confidence mask
        best_idx = 0
        conf = float(results.boxes.conf[best_idx].cpu().numpy())
        polygon = results.masks.xy[best_idx].astype(np.int32)
        
        # Run Preprocessing & Morphometry
        analysis = run_hybrid_pipeline(img_bgr, polygon)
        if analysis is None:
            continue
            
        imf = analysis["imf_percentage"]
        spec_px = analysis["specular_pixels"]
        
        # Visual Montage:
        # Col 1: Original + YOLO Segmentation Outline & Badge
        col1 = img_bgr.copy()
        cv2.polylines(col1, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(col1, f"YOLOv8n-seg Conf: {conf:.1%}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Col 2: Cropped Meat ROI (Background removed)
        col2 = analysis["isolated_muscle"].copy()
        cv2.putText(col2, "Extracted Muscle ROI", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
        # Col 3: Enhanced via Inpaint + CLAHE
        col3 = analysis["enhanced_bgr"].copy()
        cv2.putText(col3, f"CLAHE (Inpainted: {spec_px}px)", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
        # Col 4: Otsu Marbling Mask & %IMF
        col4 = analysis["fat_vis"].copy()
        cv2.putText(col4, f"Marbling %IMF: {imf:.2f}%", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
                    
        # Stitch horizontally
        montage = np.hstack([col1, col2, col3, col4])
        
        # Resize montage for easy viewing
        montage_resized = cv2.resize(montage, (1600, 400))
        
        out_filepath = os.path.join(output_dir, f"test_val_{idx+1:02d}_{base_name[:15]}.jpg")
        cv2.imwrite(out_filepath, montage_resized)
        
        report_rows.append({
            "sample": f"Val-{idx+1:02d}",
            "file": base_name[:20],
            "confidence": conf,
            "muscle_pixels": analysis["muscle_pixel_count"],
            "fat_pixels": analysis["fat_pixel_count"],
            "imf": imf,
            "glare_pixels": spec_px,
            "out_file": out_filepath
        })
        
    print(f"\n[+] Processing Completed! Saved {len(report_rows)} visual pipeline montages in '{output_dir}'.")
    print("=" * 85)
    print(f"{'Sample':<8} | {'YOLO Conf':<10} | {'Muscle Area':<12} | {'Fat Area':<10} | {'%IMF':<8} | {'Specular Plx':<12}")
    print("-" * 85)
    for r in report_rows:
        print(f"{r['sample']:<8} | {r['confidence']:>8.1%} | {r['muscle_pixels']:>10,} px | {r['fat_pixels']:>8,} px | {r['imf']:>6.2f}% | {r['glare_pixels']:>10,} px")
    print("=" * 85)
    
    # Save a CSV test report
    csv_out = os.path.join(output_dir, "validation_test_report.csv")
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("sample,file,confidence,muscle_pixels,fat_pixels,imf_percentage,specular_pixels,montage_path\n")
        for r in report_rows:
            f.write(f"{r['sample']},{r['file']},{r['confidence']:.4f},{r['muscle_pixels']},{r['fat_pixels']},{r['imf']:.2f},{r['glare_pixels']},{r['out_file']}\n")
    print(f"[*] Quantitative report saved to: {csv_out}")

if __name__ == "__main__":
    main()
