"""
AI-WagyuSense: Dataset Version 2 Verification Script
Checks image/label integrity across train, valid, and test sets, and generates visual previews.
"""

import os
import glob
import random
import cv2
import numpy as np

def read_image_unicode(file_path):
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def write_image_unicode(file_path, img):
    ext = os.path.splitext(file_path)[1]
    res, enc = cv2.imencode(ext, img)
    if res:
        enc.tofile(file_path)
        return True
    return False

def verify_split(dataset_dir, split_name):
    img_dir = os.path.join(dataset_dir, split_name, "images")
    lbl_dir = os.path.join(dataset_dir, split_name, "labels")
    
    img_files = glob.glob(os.path.join(img_dir, "*.*"))
    img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    
    print(f"[*] Split '{split_name}': {len(img_files)} images, {len(lbl_files)} label files.")
    
    # Check 1:1 match
    missing_labels = []
    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
        if not os.path.exists(lbl_path):
            missing_labels.append(base_name)
            
    if missing_labels:
        print(f"    [!] Warning: {len(missing_labels)} images missing label files!")
    else:
        print(f"    [+] Perfect 1:1 match: All {len(img_files)} images have corresponding label files.")
    return img_files

def generate_visual_previews(dataset_dir, output_dir, num_samples=6):
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(dataset_dir, "train", "images")
    lbl_dir = os.path.join(dataset_dir, "train", "labels")
    
    img_files = glob.glob(os.path.join(img_dir, "*.*"))
    img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    random.seed(42)
    selected_files = random.sample(img_files, min(num_samples, len(img_files)))
    print(f"\n[*] Generating {len(selected_files)} visual verification samples in '{output_dir}'...")
    
    for idx, img_path in enumerate(selected_files):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
        
        img = read_image_unicode(img_path)
        if img is None:
            continue
            
        h, w, c = img.shape
        overlay = img.copy()
        polygons_found = 0
        
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                pts = []
                for i in range(0, len(coords), 2):
                    px = int(coords[i] * w)
                    py = int(coords[i+1] * h)
                    pts.append([px, py])
                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                
                # Draw semi-transparent polygon (Cyan: B=255, G=200, R=0)
                cv2.fillPoly(overlay, [pts], color=(255, 200, 0))
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                
                centroid = pts.mean(axis=0)[0].astype(int)
                cv2.putText(img, f"meat_roi ({cls_id})", (max(0, centroid[0]-40), centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                polygons_found += 1
                
        alpha = 0.35
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        out_file = os.path.join(output_dir, f"sample_{idx+1}_{base_name[:25]}.jpg")
        write_image_unicode(out_file, blended)
        print(f"    - Sample {idx+1}: {os.path.basename(img_path)[:30]}... ({w}x{h}) -> Polygons: {polygons_found}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print("      AI-WagyuSense: Dataset Version 2 Verification & Audit        ")
    print("=" * 70)
    print(f"Dataset Root: {base_dir}")
    
    total_imgs = 0
    for split in ["train", "valid", "test"]:
        imgs = verify_split(base_dir, split)
        total_imgs += len(imgs)
        
    print(f"\n[+] Total Images across all splits in Version 2: {total_imgs}")
    
    preview_dir = os.path.join(base_dir, "preview_verification")
    generate_visual_previews(base_dir, preview_dir, num_samples=6)
    print("=" * 70)

if __name__ == "__main__":
    main()
