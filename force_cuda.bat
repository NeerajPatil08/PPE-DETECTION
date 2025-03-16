@echo off
echo Forcing CUDA usage for your NVIDIA GTX 1650...
echo.

set CUDA_VISIBLE_DEVICES=0
python force_cuda.py

echo.
echo If CUDA is still not working, follow the instructions in CUDA_TROUBLESHOOTING.md
pause 