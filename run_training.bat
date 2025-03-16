@echo off
echo Starting PPE Detection Model Training with NVIDIA GTX 1650...
echo.

set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Check if checkpoints exist
if exist "checkpoints\last.pt" (
    echo Previous training checkpoint found.
    echo.
    echo Options:
    echo 1. Resume training (you'll be asked about remaining epochs)
    echo 2. Start fresh training
    echo.
    set /p CHOICE="Enter your choice (1/2): "
    
    if "%CHOICE%"=="1" (
        set RESUME_FLAG=--resume
    ) else (
        set RESUME_FLAG=
        echo Starting fresh training...
    )
) else (
    set RESUME_FLAG=
)

REM Check if CUDA is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Starting training with reduced memory usage...
python train_ppe_model.py --data data.yaml --epochs 50 --batch 4 --img 416 --optimizer Adam --lr0 0.001 --augment --device 0 --save-period 5 %RESUME_FLAG%

REM If the above command fails, try with even smaller batch size and image size
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo First attempt failed. Trying with smaller batch size and image size...
    python train_ppe_model.py --data data.yaml --epochs 50 --batch 2 --img 320 --optimizer Adam --lr0 0.001 --augment --device 0 --save-period 5 %RESUME_FLAG%
)

REM If still fails, try with CPU
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo GPU training failed. Your system may not have enough GPU memory.
    echo Trying with CPU instead (this will be slower)...
    python train_ppe_model.py --data data.yaml --epochs 5 --batch 8 --img 640 --optimizer Adam --lr0 0.001 --augment --device cpu --save-period 1 %RESUME_FLAG%
)

echo.
echo Training completed. Check the runs/train directory for results.
echo.
echo If you encountered memory errors, please run fix_memory.bat for instructions
echo on how to increase your system's virtual memory.
pause 