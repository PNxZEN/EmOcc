@echo off
echo ========================================
echo Teacher vs Student Performance Comparison
echo ========================================
echo.

REM Activate virtual environment
call ..\\.venv\Scripts\activate.bat

REM Run comparison
python compare_occluded_performance.py ^
    --teacher_path ../pretrained/FECNet.pt ^
    --student_path ../checkpoints/curriculum/student_best.pth ^
    --fedro_root ../data/FED-RO/FED-RO_crop ^
    --kdef_root ../data/KDEF/KDEF_Sorted_Resized ^
    --output_dir eval_comparison_results

echo.
echo ========================================
echo Comparison complete!
echo Results saved to: eval_comparison_results/
echo ========================================
pause
