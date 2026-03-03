@echo off
REM Quick script to install UMAP dependencies and run visualization

echo ============================================================
echo Installing UMAP dependencies...
echo ============================================================
call ..\.venv\Scripts\activate.bat
pip install umap-learn matplotlib seaborn

echo.
echo ============================================================
echo Running UMAP visualization...
echo ============================================================
python visualize_embedding_space.py ^
    --teacher_path ../pretrained/FECNet.pt ^
    --student_path ../checkpoints/curriculum/student_best.pth ^
    --csv_path ../data/AffectNet/labels.csv ^
    --data_root ../data/AffectNet ^
    --max_samples 5000

echo.
echo ============================================================
echo Visualization complete! Check the umap folder for outputs.
echo ============================================================
pause
