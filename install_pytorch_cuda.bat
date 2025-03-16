@echo off
echo Installing PyTorch with CUDA support for your NVIDIA GTX 1650...
echo.

pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installation completed. Run check_cuda.bat to verify CUDA is now available.
pause 