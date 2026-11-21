@echo off
echo ======================================================================
echo    AI-WagyuSense: Training YOLOv8n-seg Version 3
echo ======================================================================
cd /d "%~dp0"
python train_segmentation.py
pause
