#!/bin/bash
# ==============================================================================
# AI-WagyuSense: TensorRT Engine Builder for NVIDIA Jetson Orin Nano
# ==============================================================================

echo "===================================================================="
echo "   AI-WAGYUSENSE: TENSORRT ENGINE COMPILATION (JETSON ORIN NANO)"
echo "===================================================================="

# 1. Unlock full performance
echo "[*] Setting Max Performance Power Mode (MAXN 15W)..."
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0 2>/dev/null || true
fi
if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks 2>/dev/null || true
fi

ONNX_MODEL="weights/yolov8n_seg_best.onnx"
PT_MODEL="weights/yolov8n_seg_best.pt"
ENGINE_OUTPUT="weights/yolov8n_seg_best.engine"

# Check for NVIDIA official trtexec compiler
TRTEXEC=""
if [ -f "/usr/src/tensorrt/bin/trtexec" ]; then
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
elif command -v trtexec &> /dev/null; then
    TRTEXEC="trtexec"
fi

if [ -n "$TRTEXEC" ] && [ -f "$ONNX_MODEL" ]; then
    echo "[+] Found NVIDIA TensorRT compiler: $TRTEXEC"
    echo "[*] Compiling high-speed FP16 TensorRT Engine from $ONNX_MODEL..."
    echo "[*] This takes about 2-3 minutes on Jetson Orin Nano..."
    $TRTEXEC --onnx=$ONNX_MODEL \
             --saveEngine=$ENGINE_OUTPUT \
             --fp16 \
             --memPoolSize=workspace:2048M
elif command -v yolo &> /dev/null; then
    echo "[*] Building TensorRT Engine via Ultralytics CLI..."
    yolo export model=$PT_MODEL format=engine half=True device=0
else
    echo "[-] Error: Neither trtexec nor yolo CLI found."
    echo "    Please verify JetPack TensorRT installation."
    exit 1
fi

if [ -f "$ENGINE_OUTPUT" ]; then
    echo ""
    echo "===================================================================="
    echo "[+] SUCCESS: TensorRT Engine built successfully!"
    echo "    Engine file: $ENGINE_OUTPUT"
    echo ""
    echo "    To run live camera stream at 60+ FPS, execute:"
    echo "    python3 live_camera_inference.py --weights $ENGINE_OUTPUT --source 0"
    echo "===================================================================="
else
    echo "[-] Build failed. Check the error output above."
fi
