@echo off
echo Starting PPE Detection Web Interface with NVIDIA GTX 1650...
echo.

set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Check if model exists
if not exist "models\best_ppe_model.pt" (
    echo Warning: Model file not found at models\best_ppe_model.pt
    echo The web interface will still work, but will use the default YOLOv8 model.
    echo For better PPE detection, please train the model first by running run_training.bat
    echo.
    timeout /t 5
)

REM Check if CUDA is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Starting web interface with reduced memory usage...
streamlit run web_app.py

echo.
echo Web interface closed.
echo.
echo If you encountered memory errors, please run fix_memory.bat for instructions
echo on how to increase your system's virtual memory.
pause 