"""
AI-WagyuSense: Real-Time Live Camera Inference on NVIDIA Jetson Orin Nano
Wat Khemaphirataram School - KIDE 2026 Project

Usage:
  python live_camera_inference.py --source 0               # USB Camera / Chamber Camera
  python live_camera_inference.py --source test_steak.jpg  # Single Image Testing
  python live_camera_inference.py --source video.mp4       # Video Stream
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add AI-Model directory to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "AI-Model"))
from integrated_inference import AIWagyuSensePipeline, load_image_robust, write_image_unicode

def main():
    parser = argparse.ArgumentParser(description="AI-WagyuSense Live Inference on Jetson Orin Nano")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or path to image/video")
    parser.add_argument("--weights", type=str, default="weights/yolov8n_seg_best.pt", help="Path to YOLOv8n-seg weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output_dir", type=str, default="live_captures", help="Directory to save snapshots")
    parser.add_argument("--no-gui", action="store_true", help="Run without cv2 GUI windows (headless / SSH)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("   AI-WAGYUSENSE: REAL-TIME INFERENCE ON NVIDIA JETSON ORIN NANO")
    print("=" * 80)
    print(f"[*] Compute Device : {device.upper()}")
    if device == "cuda":
        print(f"[*] GPU Name       : {torch.cuda.get_device_name(0)}")
    print(f"[*] Model Weights  : {args.weights}")
    print(f"[*] Input Source   : {args.source}")
    print(f"[*] Output Dir     : {args.output_dir}")
    print("=" * 80)
    
    # Initialize Engine
    pipeline = AIWagyuSensePipeline(args.weights, device=device)
    
    # Check if source is image file
    is_image = False
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    if os.path.isfile(args.source) and args.source.lower().endswith(valid_exts):
        is_image = True
        
    if is_image:
        print(f"[*] Processing static image: {args.source}")
        img = load_image_robust(args.source)
        if img is None:
            print("[-] Error: Unable to read image.")
            return
            
        dashboard, rec = pipeline.process_image(img)
        if dashboard is not None:
            out_path = os.path.join(args.output_dir, f"capture_{os.path.basename(args.source)}")
            write_image_unicode(out_path, dashboard)
            print(f"[+] Assessment Complete: TAS Grade {rec['tas_grade']} | %IMF: {rec['otsu_imf_percent']:.2f}%")
            print(f"[+] Saved Diagnostic Dashboard: {out_path}")
            
            # Display result if GUI available
            if not args.no_gui:
                try:
                    cv2.namedWindow("AI-WagyuSense Diagnostic Dashboard", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("AI-WagyuSense Diagnostic Dashboard", 1280, 720)
                    cv2.imshow("AI-WagyuSense Diagnostic Dashboard", dashboard)
                    print("[*] Press any key in the GUI window to exit...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"[*] Note: GUI window display skipped ({e}). Dashboard saved to disk.")
        return

    # Video / Camera Stream
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    
    # Set camera resolution (1280x720 recommended for Chamber)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"[-] Error: Could not open video source {args.source}")
        return

    print("\n[+] Camera stream started successfully!")
    print("[*] Keyboard Controls:")
    print("    - [SPACE] or [S] : Capture & Save High-Res Diagnostic Dashboard")
    print("    - [Q] or [ESC]   : Exit Stream\n")

    fps_history = []
    capture_count = 0

    has_gui = not args.no_gui
    if has_gui:
        try:
            cv2.namedWindow("AI-WagyuSense Live Feed (Jetson)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AI-WagyuSense Live Feed (Jetson)", 1280, 720)
        except Exception:
            has_gui = False
            print("[*] GUI window not available. Running in headless console mode.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[-] End of stream or camera disconnected.")
            break
            
        t0 = time.perf_counter()
        # Fast streaming mode: lightweight polygon + marbling preview for live 30+ FPS
        _, rec = pipeline.process_image(frame, fast_mode=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        current_fps = 1000.0 / max(latency_ms, 1.0)
        fps_history.append(current_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        display_frame = frame.copy()

        if isinstance(rec, dict):
            # Real-time visual contour & marbling highlights
            polygon = rec.get("polygon")
            if polygon is not None:
                cv2.polylines(display_frame, [polygon], True, (0, 229, 255), 2, cv2.LINE_AA)
            fat_mask = rec.get("fat_mask")
            if fat_mask is not None and np.any(fat_mask):
                display_frame[fat_mask == 255] = [0, 230, 255] # Gold highlight on marbling

            grade = rec["tas_grade"]
            imf = rec["otsu_imf_percent"]
            
            # Top HUD Bar
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 70), (20, 24, 30), -1)
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 70), (0, 229, 255), 2)
            
            # Grade Badge
            badge_color = (0, 255, 180) if grade >= 4 else ((0, 229, 255) if grade >= 3 else (180, 180, 180))
            cv2.putText(display_frame, f"TAS GRADE {grade}", (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, badge_color, 3, cv2.LINE_AA)
            cv2.putText(display_frame, f"%IMF: {imf:.2f}%", (340, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"[SPACE] Capture Card", (600, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"{avg_fps:.1f} FPS ({latency_ms:.1f}ms)", (display_frame.shape[1] - 250, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 60), (20, 24, 30), -1)
            cv2.putText(display_frame, "Scanning for Longissimus dorsi muscle...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

        if has_gui:
            cv2.imshow("AI-WagyuSense Live Feed (Jetson)", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]: # Q or ESC
                break
            elif key in [ord(' '), ord('s'), ord('S')]: # SPACE or S
                capture_count += 1
                print(f"\n[*] Processing Full Diagnostic Certificate (Deep Metrology)...")
                dashboard, full_rec = pipeline.process_image(frame, fast_mode=False)
                if dashboard is not None and isinstance(full_rec, dict):
                    save_path = os.path.join(args.output_dir, f"jetson_capture_{capture_count:03d}.jpg")
                    write_image_unicode(save_path, dashboard)
                    print(f"[+] Capture #{capture_count} Complete: Grade {full_rec['tas_grade']} | %IMF: {full_rec['otsu_imf_percent']:.2f}%")
                    print(f"[+] Saved High-Res Diagnostic Card: {save_path}")
                    
                    # Visual Flash feedback on screen
                    flash = display_frame.copy()
                    cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (0, 255, 0), 12)
                    cv2.putText(flash, f"CAPTURED! Grade {full_rec['tas_grade']} | Saved: {os.path.basename(save_path)}",
                                (30, flash.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("AI-WagyuSense Live Feed (Jetson)", flash)
                    cv2.waitKey(400)
        else:
            # Console headless logging
            if len(fps_history) % 30 == 0:
                imf_str = f"{rec['otsu_imf_percent']:.1f}%" if isinstance(rec, dict) else "N/A"
                grade_str = f"Grade {rec['tas_grade']}" if isinstance(rec, dict) else "Scanning"
                print(f"[*] Stream: {avg_fps:.1f} FPS | Latency: {latency_ms:.1f}ms | Status: {grade_str} ({imf_str})")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Inference stream ended gracefully.")

if __name__ == "__main__":
    main()
