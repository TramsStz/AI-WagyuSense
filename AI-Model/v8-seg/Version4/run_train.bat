@echo off
chcp 65001 >nul
echo ======================================================================
echo    AI-WagyuSense: Training YOLOv8n-seg Version 4
echo    Wat Khemaphirataram School - KIDE 2026 Project
echo ======================================================================
cd /d "%~dp0"
python train_segmentation.py
pause
