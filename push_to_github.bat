@echo off
title AI-WagyuSense - Push to GitHub
echo ============================================================
echo   AI-WagyuSense: Pushing to GitHub (TramsStz/AI-WagyuSense)
echo ============================================================
echo.
cd /d "%~dp0"
git remote -v
echo.
echo Running: git push -u origin main ...
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo ============================================================
    echo   [SUCCESS] Code and Model pushed to GitHub successfully!
    echo ============================================================
) else (
    echo ============================================================
    echo   [FAILED / PENDING] Please check the error message above.
    echo ============================================================
)
echo.
pause
