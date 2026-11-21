"""
AI-WagyuSense: Model 2 (Dual-Head EfficientNet-B0) Training Pipeline
Wat Khemaphirataram School - KIDE 2026 Project

Trains Dual-Head EfficientNet-B0 on the 224x224 Muscle ROI Dataset:
  - Task 1: TAS Beef Quality Grade Classification (5 classes: Grade 1 - 5)
  - Task 2: Intramuscular Fat Percentage (%IMF) Regression
  - Automatic Export to ONNX format for Jetson Orin Nano deployment
"""

import os
import sys
import time
import math
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from model2_architecture import DualHeadEfficientNetB0

class BeefROIDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        self.samples = []
        self.is_train = is_train
        
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "path": row["roi_path"],
                    "grade": int(row["tas_grade"]) - 1, # 1..5 mapped to 0..4
                    "imf": float(row["imf_percent"])
                })
                
        # Data Transforms
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["path"]).convert("RGB")
        tensor_img = self.transform(img)
        
        grade_label = torch.tensor(item["grade"], dtype=torch.long)
        imf_label = torch.tensor(item["imf"], dtype=torch.float32)
        
        return tensor_img, grade_label, imf_label

def train_epoch(model, dataloader, optimizer, criterion_cls, criterion_reg, device, lambda_reg=0.05):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    correct = 0
    total_samples = 0
    abs_errors = []
    
    for imgs, grade_labels, imf_labels in dataloader:
        imgs = imgs.to(device)
        grade_labels = grade_labels.to(device)
        imf_labels = imf_labels.to(device)
        
        optimizer.zero_grad()
        logits, pred_imf = model(imgs)
        
        loss_cls = criterion_cls(logits, grade_labels)
        loss_reg = criterion_reg(pred_imf, imf_labels)
        
        loss = loss_cls + (lambda_reg * loss_reg)
        loss.backward()
        optimizer.step()
        
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += loss_cls.item() * bs
        total_reg_loss += loss_reg.item() * bs
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == grade_labels).sum().item()
        total_samples += bs
        
        abs_errors.extend(torch.abs(pred_imf - imf_labels).detach().cpu().numpy())
        
    avg_loss = total_loss / total_samples
    avg_cls = total_cls_loss / total_samples
    avg_reg = total_reg_loss / total_samples
    acc = correct / total_samples
    mae = float(np.mean(abs_errors))
    
    return avg_loss, avg_cls, avg_reg, acc, mae

def evaluate(model, dataloader, criterion_cls, criterion_reg, device, lambda_reg=0.05):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    abs_errors = []
    all_preds_cls = []
    all_targets_cls = []
    
    with torch.no_grad():
        for imgs, grade_labels, imf_labels in dataloader:
            imgs = imgs.to(device)
            grade_labels = grade_labels.to(device)
            imf_labels = imf_labels.to(device)
            
            logits, pred_imf = model(imgs)
            
            loss_cls = criterion_cls(logits, grade_labels)
            loss_reg = criterion_reg(pred_imf, imf_labels)
            loss = loss_cls + (lambda_reg * loss_reg)
            
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == grade_labels).sum().item()
            total_samples += bs
            
            abs_errors.extend(torch.abs(pred_imf - imf_labels).cpu().numpy())
            all_preds_cls.extend(preds.cpu().numpy())
            all_targets_cls.extend(grade_labels.cpu().numpy())
            
    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    mae = float(np.mean(abs_errors))
    
    return avg_loss, acc, mae

def export_onnx(model, save_path, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    print(f"\n[*] Exporting Dual-Head EfficientNet-B0 to ONNX: {save_path}")
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input_roi"],
        output_names=["grade_logits", "imf_percent"],
        dynamic_axes={"input_roi": {0: "batch"}, "grade_logits": {0: "batch"}, "imf_percent": {0: "batch"}}
    )
    print(f"[+] ONNX Export successful! File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")

def main():
    print("=" * 75)
    print("  AI-WagyuSense: Model 2 (Dual-Head EfficientNet-B0) Training Pipeline")
    print("=" * 75)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset_roi")
    train_csv = os.path.join(dataset_dir, "metadata_train.csv")
    valid_csv = os.path.join(dataset_dir, "metadata_valid.csv")
    test_csv = os.path.join(dataset_dir, "metadata_test.csv")
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing {train_csv}. Run extract_roi_dataset.py first.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Datasets and Loaders
    train_dataset = BeefROIDataset(train_csv, is_train=True)
    valid_dataset = BeefROIDataset(valid_csv, is_train=False)
    test_dataset = BeefROIDataset(test_csv, is_train=False)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"[*] Dataset Statistics:")
    print(f"    - Train Samples: {len(train_dataset)}")
    print(f"    - Valid Samples: {len(valid_dataset)}")
    print(f"    - Test Samples : {len(test_dataset)}")
    print("-" * 75)
    
    # Model Initialization
    model = DualHeadEfficientNetB0(num_classes=5, pretrained=True).to(device)
    
    # Losses
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float("inf")
    best_weights_path = os.path.join(base_dir, "best_model2_efficientnet.pth")
    
    print(f"[*] Starting Dual-Head Training for {epochs} epochs...")
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^9} | {'Train MAE':^9} | {'Val Loss':^9} | {'Val Acc':^8} | {'Val MAE':^8} | {'LR':^9}")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_cls, tr_reg, tr_acc, tr_mae = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_reg, device, lambda_reg=0.05
        )
        val_loss, val_acc, val_mae = evaluate(
            model, valid_loader, criterion_cls, criterion_reg, device, lambda_reg=0.05
        )
        lr_curr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_mae": val_mae
            }, best_weights_path)
            mark = "★ BEST"
        else:
            mark = ""
            
        print(f"{epoch:^7d} | {tr_loss:^10.4f} | {tr_acc*100:^8.1f}% | {tr_mae:^9.2f} | {val_loss:^9.4f} | {val_acc*100:^7.1f}% | {val_mae:^8.2f} | {lr_curr:^9.2e} {mark}")
        
    print("=" * 80)
    print(f"[*] Training finished! Best weights saved to: {best_weights_path}")
    
    # Load Best Model for Final Test Evaluation
    checkpoint = torch.load(best_weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, test_mae = evaluate(model, test_loader, criterion_cls, criterion_reg, device, lambda_reg=0.05)
    print("\n" + "=" * 80)
    print("             MODEL 2 (EFFICIENTNET-B0) FINAL TEST RESULTS             ")
    print("=" * 80)
    print(f"Test Set Classification Accuracy (TAS Grade 1-5): {test_acc*100:.2f}%")
    print(f"Test Set %IMF Regression Mean Absolute Error (MAE): {test_mae:.2f}%")
    print("=" * 80)
    
    # Export to ONNX
    onnx_path = os.path.join(base_dir, "model2_dualhead.onnx")
    export_onnx(model, onnx_path, device)
    print("=" * 80)

if __name__ == "__main__":
    main()
