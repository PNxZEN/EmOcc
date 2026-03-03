@echo off
echo ============================================================
echo Running Teacher vs Student Comparison on EXPW Dataset
echo ============================================================
echo.

cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python eval\compare_expw_performance.py

echo.
echo ============================================================
echo Evaluation Complete!
echo ============================================================
pause
