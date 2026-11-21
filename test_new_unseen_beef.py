"""
AI-WagyuSense: Dual-Model Testing on New Unseen Beef Images
Wat Khemaphirataram School - KIDE 2026 Project

Tests Model 1 (YOLOv8n-seg v4) + Preprocessing + Model 2 (Dual-Head EfficientNet-B0)
on 7 newly discovered diverse beef steak cuts:
  - 01: USDA Choice Bone-in Ribeye
  - 02: USDA Choice T-Bone Steak
  - 03: Japanese Kuroge Wagyu Ribeye (Extreme Marbling)
  - 04: Matsusaka Wagyu Slices (Fine Marbling)
  - 05: Standard Lean Beef Steak (European Rindersteak)
  - 06: Porterhouse Raw Steak
  - 07: Market Fresh Beef Steak
"""

import os
import sys
import glob
import time
import cv2
import numpy as np
import torch

# Paths
root_dir = os.path.dirname(os.path.abspath(__file__))
ai_model_dir = os.path.join(root_dir, "AI-Model")
sys.path.append(ai_model_dir)
from integrated_inference import AIWagyuSensePipeline, load_image_robust, write_image_unicode

def main():
    print("=" * 90)
    print(" AI-WAGYUSENSE: DUAL-MODEL EVALUATION ON NEW UNSEEN BEEF SPECIMENS")
    print("=" * 90)
    
    seg_model_path = os.path.join(ai_model_dir, "v8-seg", "Version4", "runs", "segment", "ai_wagyusense_segment_v4", "weights", "best.pt")
    cls_model_path = os.path.join(ai_model_dir, "model2-efficientnet", "best_model2_efficientnet.pth")
    test_input_dir = os.path.join(ai_model_dir, "new_test_web_beef")
    output_dir = os.path.join(ai_model_dir, "ผลการประเมิน_รูปเนื้อใหม่_2โมเดล")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Model 1 (YOLOv8n-seg)     : {seg_model_path}")
    print(f"[*] Model 2 (Dual-Head EffNet): {cls_model_path}")
    print(f"[*] Input Specimens           : {test_input_dir}")
    print(f"[*] Output Dashboards         : {output_dir}")
    
    pipeline = AIWagyuSensePipeline(seg_model_path, cls_model_path, device="cuda")
    
    files = sorted([f for f in os.listdir(test_input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))])
    print(f"[*] Discovered {len(files)} new test images.")
    print("-" * 90)
    
    summary_results = []
    
    for idx, fname in enumerate(files):
        fpath = os.path.join(test_input_dir, fname)
        img_bgr = load_image_robust(fpath)
        if img_bgr is None:
            print(f"[-] Failed to load: {fname}")
            continue
            
        t0 = time.perf_counter()
        dashboard, rec = pipeline.process_image(img_bgr)
        t_total = (time.perf_counter() - t0) * 1000.0
        
        if dashboard is None:
            print(f"[-] [{idx+1}/{len(files)}] {fname}: No muscle detected by Model 1.")
            continue
            
        # Save Dashboard
        out_name = f"eval_{idx+1:02d}_{os.path.splitext(fname)[0]}.jpg"
        out_path = os.path.join(output_dir, out_name)
        write_image_unicode(out_path, dashboard)
        
        grade = rec["tas_grade"]
        g_conf = rec["grade_confidence"]
        otsu_imf = rec["otsu_imf_percent"]
        deep_imf = rec["deep_imf_percent"]
        seg_conf = rec["seg_confidence"]
        lats = rec["latency_breakdown_ms"]
        
        print(f"[+] [{idx+1}/{len(files)}] {fname[:28]:<28} | Seg Conf: {seg_conf:.1%} | TAS Grade: {grade} ({g_conf:.1%}) | Otsu %IMF: {otsu_imf:5.2f}% | Deep %IMF: {deep_imf:5.2f}% | Lat: {lats['total']:.1f}ms")
        
        summary_results.append({
            "filename": fname,
            "seg_conf": seg_conf,
            "tas_grade": grade,
            "grade_conf": g_conf,
            "otsu_imf": otsu_imf,
            "deep_imf": deep_imf,
            "delta_imf": abs(otsu_imf - deep_imf),
            "lat_total": lats["total"],
            "glare_px": rec["glare_pixels"],
            "dashboard_path": out_path
        })
        
    print("\n" + "=" * 90)
    print("                         SUMMARY OF UNSEEN SPECIMEN BENCHMARK                        ")
    print("=" * 90)
    print(f"{'Specimen Name':<28} | {'YOLOv8':^8} | {'TAS Grade':^11} | {'Otsu IMF':^10} | {'Deep IMF':^10} | {'Δ Diff':^8} | {'Speed':^9}")
    print("-" * 90)
    for r in summary_results:
        print(f"{r['filename'][:28]:<28} | {r['seg_conf']*100:6.1f}% | Grade {r['tas_grade']} ({r['grade_conf']*100:4.1f}%) | {r['otsu_imf']:7.2f}%  | {r['deep_imf']:7.2f}%  | {r['delta_imf']:6.2f}% | {r['lat_total']:5.1f} ms")
    print("=" * 90)
    print(f"[+] All high-resolution 1600x900 Diagnostic Dashboards saved to:\n    {output_dir}")

if __name__ == "__main__":
    main()
