"""
AI-WagyuSense: Muscle Segmentation Training Script (YOLOv8n-seg - Version 4)
Wat Khemaphirataram School - KIDE 2026 Project

Fine-tunes YOLOv8n-seg on Dataset Version 4 (849 images: Train 810, Val 28, Test 11)
to segment the Longissimus dorsi (meat_roi) muscle and exports the best model to ONNX.
"""

import os
import sys
import torch
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("   AI-WagyuSense: YOLOv8n-seg Training Pipeline (Version 4)    ")
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
        print("[!] Warning: CUDA is not available. Training will proceed on CPU.")
    print("-" * 70)

    # 2. Dataset Path Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(base_dir, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
    print(f"[*] Dataset Config: {data_yaml_path}")

    # 3. Model Initialization (YOLOv8n-seg)
    model_weight = os.path.join(base_dir, "yolov8n-seg.pt")
    if not os.path.exists(model_weight):
        model_weight = "yolov8n-seg.pt"
    print(f"[*] Initializing model with base weights: {model_weight}")
    model = YOLO(model_weight)

    # 4. Training Hyperparameters
    train_args = {
        "data": data_yaml_path,
        "epochs": 100,
        "patience": 25,              # Early stopping if no improvement for 25 epochs
        "batch": 16,                 # Optimal batch size for 8GB RTX 4060
        "imgsz": 640,                # 640x640 resolution
        "device": device,
        "workers": 4,                # Windows multiprocess workers
        "project": os.path.join(base_dir, "runs", "segment"),
        "name": "ai_wagyusense_segment_v4",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
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
    best_weights_path = os.path.join(base_dir, "runs", "segment", "ai_wagyusense_segment_v4", "weights", "best.pt")
    print(f"\n[*] Running final validation on best weights: {best_weights_path}")
    best_model = YOLO(best_weights_path)
    
    val_results = best_model.val(data=data_yaml_path, split="val")
    print("\n[*] Validation Set (val) Metrics Summary:")
    try:
        mask_map50 = val_results.seg.map50
        mask_map = val_results.seg.map
        box_map50 = val_results.box.map50
        box_map = val_results.box.map
        print(f"    - Mask mAP@0.50     : {mask_map50:.4f}")
        print(f"    - Mask mAP@0.50:0.95: {mask_map:.4f}")
        print(f"    - Box mAP@0.50      : {box_map50:.4f}")
        print(f"    - Box mAP@0.50:0.95 : {box_map:.4f}")
    except Exception as e:
        print(f"    [!] Notice: Metrics parsing details: {e}")

    # Evaluate on Test Set as well
    print("\n[*] Running evaluation on Test Set (test split)...")
    try:
        test_results = best_model.val(data=data_yaml_path, split="test")
        print("\n[*] Test Set (test) Metrics Summary:")
        print(f"    - Mask mAP@0.50     : {test_results.seg.map50:.4f}")
        print(f"    - Mask mAP@0.50:0.95: {test_results.seg.map:.4f}")
        print(f"    - Box mAP@0.50      : {test_results.box.map50:.4f}")
        print(f"    - Box mAP@0.50:0.95 : {test_results.box.map:.4f}")
    except Exception as e:
        print(f"    [!] Test split evaluation notice: {e}")

    # 7. Model Export to ONNX
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
    except Exception as e:
        print(f"[!] Export warning: {e}")

    print("\n" + "=" * 70)
    print(f"[+] All processes completed successfully!")
    print(f"    - Best PyTorch Weights: {os.path.abspath(best_weights_path)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
