# AI-WagyuSense: High-Precision Beef Morphometry & Quality Assessment System
**Wat Khemaphirataram School — KIDE 2026 Project**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8-seg](https://img.shields.io/badge/YOLO-v8n--seg-00FFFF.svg)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano-76B900.svg)](https://developer.nvidia.com/embedded/jetson-orin)
[![TAS 6001-2014](https://img.shields.io/badge/Standard-TAS%206001--2014-orange.svg)](https://www.acfs.go.th/)

---

## 📌 Overview (ภาพรวมโครงการ)
**AI-WagyuSense** is an edge-deployable, physics-informed computer vision metrology system designed for automated, objective, and non-destructive grading of beef marbling (*Intramuscular Fat: %IMF*), specifically tailored for Thai Native Beef and Wagyu crosses according to Thai Agricultural Standard **TAS 6001-2014**.

Instead of relying on black-box neural networks that suffer from *Regression to the Mean* and *Resolution Bottlenecks*, AI-WagyuSense utilizes a **Pragmatic Hybrid Architecture**:
1. **Deep Learning (YOLOv8n-seg v4):** Extracts the primary *Longissimus dorsi* muscle region-of-interest (ROI).
2. **Physics-Informed Preprocessing:** Eliminates LED/moisture specular reflections via Telea Fast Marching Inpainting ($S < 20, V > 235$) and enhances capillary marbling contrast via Green Channel CLAHE ($\sim 540-580\text{ nm}$ Myoglobin absorption peak).
3. **Mathematical Morphometry & Metrology:** Computes exact, pixel-verifiable %IMF (via Calibrated Clamped Otsu $\ge 140$), marbling particle count, fineness density, $3 \times 3$ spatial dispersion uniformity ($CoV$), lean color spectrophotometry (CIE $L^*a^*b^*$), and deterministic TAS Grade 1–5 assignment.

---

## 🚀 Key Advantages & Performance (จุดเด่นและประสิทธิภาพ)
* **Zero Hallucination (100% White-Box XAI):** Every detected marbling pixel is physically auditable on the full-resolution cut.
* **Blazing Fast Real-Time Throughput:**
  - $\sim 11 - 16\text{ ms}$ total pipeline latency on NVIDIA RTX 4060 ($> 70\text{ FPS}$).
  - $\sim 25 - 35\text{ ms}$ on NVIDIA Jetson Orin Nano ($> 30\text{ FPS}$).
* **Ultra-Lightweight Repository:** Pre-packaged model weights are only **$6.8\text{ MB}$**, enabling instant cloning and deployment onto edge IoT hardware.

---

## 📂 Repository Structure (โครงสร้างไฟล์)
```
AI-WagyuSense/
├── AI-Model/
│   └── integrated_inference.py     # Core Production Pipeline Engine
├── weights/
│   └── yolov8n_seg_best.pt        # Trained YOLOv8n-seg v4 weights (6.8 MB)
├── live_camera_inference.py       # Live Webcam / Jetson Chamber Script
├── hybrid_preprocessing_pipeline.py # Calibrated CV & Morphometry Test Script
├── requirements.txt               # Python Dependencies
├── .gitignore                     # Git filter for lightweight sync
└── README.md                      # Documentation
```

---

## 🛠️ Installation & Setup on NVIDIA Jetson Orin Nano

### 1. Clone the Repository
On your Jetson terminal:
```bash
git clone https://github.com/<YOUR_USERNAME>/AI-WagyuSense.git
cd AI-WagyuSense
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Ensure PyTorch for Jetson JetPack with CUDA support is installed via NVIDIA's official Jetson PyTorch wheel).*

---

## 🎮 Running Inference (การเรียกใช้งาน)

### A. Live Camera Stream in Chamber (กล้องสดในตู้ Chamber)
```bash
python live_camera_inference.py --source 0
```
* **Keyboard Shortcuts:**
  - **`[SPACE]` or `[S]`**: Capture high-resolution $1600 \times 900$ Diagnostic Dashboard snapshot to `live_captures/`.
  - **`[Q]` or `[ESC]`**: Exit stream.

### B. Single Test Image Evaluation (ทดสอบภาพนิ่ง)
```bash
python live_camera_inference.py --source path/to/steak.jpg
```

### C. Run Full Batch Assessment
```bash
python AI-Model/integrated_inference.py
```

---

## 📊 Measured Quality Metrics (ตัวชี้วัดที่ระบบรายงาน)
| Metric | Description | Formula / Standard |
| :--- | :--- | :--- |
| **TAS Quality Grade** | Beef quality level (Grade 1 to 5) | TAS 6001-2014 mapping |
| **%IMF** | Intramuscular Fat Percentage | $\frac{N_{\text{fat}}}{N_{\text{muscle}}} \times 100\%$ |
| **Particle Count** | Total marbling islands ($\ge 2\text{ px}$) | Connected Components |
| **Fineness Density** | Particle density per area | $\frac{\text{Count}}{N_{\text{muscle}}} \times 1,000$ |
| **Spatial Dispersion CoV**| Regional distribution uniformity | $3 \times 3\text{ Grid } CoV = \frac{\text{SD}}{\text{Mean}} \times 100\%$ |
| **Meat Color** | Lean color spectrophotometry | CIE $L^*a^*b^*$ (Lightness & Redness) |

---

## 👨‍💻 Authors & Credits
* **Project:** AI-WagyuSense (KIDE 2026)
* **School:** Wat Khemaphirataram School, Nonthaburi, Thailand
