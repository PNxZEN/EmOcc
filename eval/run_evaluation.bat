@echo off
REM Quick script to run FED-RO evaluation

echo ============================================================
echo Running FED-RO Evaluation (Teacher vs Student)
echo ============================================================

call ..\.venv\Scripts\activate.bat

python evaluate_fedro.py ^
    --teacher_path ../pretrained/FECNet.pt ^
    --student_path ../checkpoints/curriculum/student_best.pth ^
    --data_root ../data/FED-RO/FED-RO_crop ^
    --batch_size 32

echo.
echo ============================================================
echo Evaluation complete! Check the eval folder for outputs.
echo ============================================================
pause
