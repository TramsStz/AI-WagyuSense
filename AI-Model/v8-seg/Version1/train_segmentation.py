"""
AI-WagyuSense: Muscle Segmentation Training Script (YOLOv8n-seg)
Wat Khemaphirataram School - KIDE 2026 Project

This script fine-tunes YOLOv8n-seg to segment the Longissimus dorsi (meat_roi) muscle 
from Thai native beef cross-section images and exports the best model to ONNX for 
subsequent TensorRT deployment on NVIDIA Jetson Orin Nano.
"""

import os
import sys
import torch
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("      AI-WagyuSense: YOLOv8n-seg Training & Optimization Pipeline     ")
    print("=" * 70)

    # 1. Hardware & Environment Verification
    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else "cpu"
    
    print(f"[*] Python Version : {sys.version.split()[0]}")
    print(f"[*] PyTorch Version: {torch.__version__}")
    print(f"[*] CUDA Available : {cuda_available}")
    if cuda_available:
        print(f"[*] Compute Device : {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM Available : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[!] Warning: CUDA is not available. Training will proceed on CPU (slower).")
    print("-" * 70)

    # 2. Dataset Path Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(base_dir, "Dataset1", "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
    print(f"[*] Dataset Config: {data_yaml_path}")

    # 3. Model Initialization (YOLOv8n-seg: Nano size for Edge deployment)
    model_weight = "yolov8n-seg.pt"
    print(f"[*] Initializing model with base weights: {model_weight}")
    model = YOLO(model_weight)

    # 4. Training Hyperparameters
    # Designed specifically for Longissimus dorsi muscle segmentation
    train_args = {
        "data": data_yaml_path,
        "epochs": 100,
        "patience": 20,              # Early stopping if no improvement for 20 epochs
        "batch": 16,                 # Optimal batch size for 8GB RTX 4060
        "imgsz": 640,                # Standard input resolution
        "device": device,
        "workers": 4,                # Windows multiprocess workers
        "project": "runs/segment",
        "name": "ai_wagyusense_segment",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",         # Ultralytics auto-selects AdamW/SGD
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        # Augmentation settings tailored for meat samples:
        "fliplr": 0.5,               # Horizontal flip
        "flipud": 0.5,               # Vertical flip
        "degrees": 10.0,             # Slight rotation
        "hsv_h": 0.015,              # Conservative hue variation (preserve meat color tone)
        "hsv_s": 0.2,                # Moderate saturation
        "hsv_v": 0.2,                # Moderate value/brightness
    }

    print("[*] Starting Training with configuration:")
    for k, v in train_args.items():
        print(f"    - {k}: {v}")
    print("-" * 70)

    # 5. Run Training
    results = model.train(**train_args)
    print("\n" + "=" * 70)
    print("[*] Training Phase Completed!")
    print("=" * 70)

    # 6. Evaluation on Validation Set
    print("\n[*] Running final validation on best weights...")
    best_weights_path = os.path.join("runs", "segment", "ai_wagyusense_segment", "weights", "best.pt")
    best_model = YOLO(best_weights_path)
    val_results = best_model.val(data=data_yaml_path, split="val")

    print("\n[*] Validation Metrics Summary:")
    try:
        # Mask Metrics (Segmentation)
        mask_map50 = val_results.seg.map50
        mask_map = val_results.seg.map
        print(f"    - Mask mAP@0.50     : {mask_map50:.4f}")
        print(f"    - Mask mAP@0.50:0.95: {mask_map:.4f}")
        
        # Box Metrics (Localization)
        box_map50 = val_results.box.map50
        box_map = val_results.box.map
        print(f"    - Box mAP@0.50      : {box_map50:.4f}")
        print(f"    - Box mAP@0.50:0.95 : {box_map:.4f}")
    except Exception as e:
        print(f"    [!] Notice: Metrics parsing details: {e}")

    # 7. Model Export to ONNX (Ready for Jetson TensorRT FP16 compilation)
    print("\n" + "-" * 70)
    print("[*] Exporting best model to ONNX format (opset=12, simplify=True)...")
    try:
        exported_path = best_model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False
        )
        print(f"[*] ONNX Export Successful: {exported_path}")
        print(f"[*] This file is ready to be converted into TensorRT FP16 engine via 'trtexec' on Jetson Orin Nano.")
    except Exception as e:
        print(f"[!] Export warning: {e}")

    print("\n" + "=" * 70)
    print(f"[+] All processes completed successfully!")
    print(f"    - Best PyTorch Weights: {os.path.abspath(best_weights_path)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
