import os
import glob
import random
import cv2
import numpy as np

def visualize_dataset(dataset_dir="Dataset1", output_dir="preview_verification", num_samples=6):
    os.makedirs(output_dir, exist_ok=True)
    
    train_img_dir = os.path.join(dataset_dir, "train", "images")
    train_lbl_dir = os.path.join(dataset_dir, "train", "labels")
    
    img_files = glob.glob(os.path.join(train_img_dir, "*.*"))
    img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not img_files:
        print(f"Error: No images found in {train_img_dir}")
        return
        
    print(f"Total training images found: {len(img_files)}")
    
    # Pick fixed seed or random samples
    random.seed(42)
    selected_files = random.sample(img_files, min(num_samples, len(img_files)))
    
    summary = []
    
    for idx, img_path in enumerate(selected_files):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(train_lbl_dir, f"{base_name}.txt")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
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
                
                # Pair into (x, y)
                pts = []
                for i in range(0, len(coords), 2):
                    px = int(coords[i] * w)
                    py = int(coords[i+1] * h)
                    pts.append([px, py])
                    
                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                
                # Draw semi-transparent polygon (Cyan color: B=255, G=229, R=0 in BGR)
                cv2.fillPoly(overlay, [pts], color=(255, 200, 0))
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                
                # Label badge
                centroid = pts.mean(axis=0)[0].astype(int)
                cv2.putText(img, f"meat_roi ({cls_id})", (centroid[0]-40, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                polygons_found += 1
                
        # Alpha blend (0.6 original + 0.4 overlay)
        alpha = 0.35
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        out_file = os.path.join(output_dir, f"sample_{idx+1}_{base_name[:20]}.jpg")
        cv2.imwrite(out_file, blended)
        summary.append({
            "filename": os.path.basename(img_path),
            "resolution": f"{w}x{h}",
            "polygons": polygons_found,
            "preview_saved": out_file
        })
        
    print(f"Successfully generated {len(summary)} visual verification samples in '{output_dir}'.")
    for s in summary:
        print(f" - [{s['resolution']}] {s['filename'][:30]}... -> Polygons: {s['polygons']} -> Saved: {s['preview_saved']}")

if __name__ == "__main__":
    visualize_dataset()
