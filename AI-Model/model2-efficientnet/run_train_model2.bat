@echo off
chcp 65001 >nul
echo ======================================================================
echo    AI-WagyuSense: Training Model 2 (Dual-Head EfficientNet-B0)
echo    Wat Khemaphirataram School - KIDE 2026 Project
echo ======================================================================
cd /d "%~dp0"
python train_model2.py
pause
