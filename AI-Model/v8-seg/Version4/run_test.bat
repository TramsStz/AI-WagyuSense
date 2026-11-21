@echo off
chcp 65001 >nul
echo ======================================================================
echo    AI-WagyuSense: Testing YOLOv8n-seg Version 4 & Pipeline
echo    Wat Khemaphirataram School - KIDE 2026 Project
echo ======================================================================
cd /d "%~dp0"
python test_model.py
pause
