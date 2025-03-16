# CUDA Troubleshooting Guide for NVIDIA GTX 1650

This guide will help you resolve CUDA availability issues with your NVIDIA GTX 1650 GPU for the PPE Detection application.

## Quick Fix

Run the following batch files in order:

1. `check_cuda.bat` - Diagnose the current CUDA status
2. `install_pytorch_cuda.bat` - Install PyTorch with CUDA support
3. `check_cuda.bat` - Verify CUDA is now available

## Common Issues and Solutions

### 1. Paging File Too Small Error

**Symptoms:**
- Error message: `[WinError 1455] The paging file is too small for this operation to complete`
- Error loading CUDA DLLs like `cufft64_10.dll`

**Solution:**
1. Increase your system's virtual memory (paging file):
   - Right-click on "This PC" or "My Computer" and select "Properties"
   - Click on "Advanced system settings" on the left
   - In the System Properties window, go to the "Advanced" tab
   - Click on "Settings" under the "Performance" section
   - Go to the "Advanced" tab in the Performance Options window
   - Click on "Change" under the "Virtual memory" section
   - Uncheck "Automatically manage paging file size for all drives"
   - Select your system drive (usually C:)
   - Select "Custom size" and set:
     - Initial size: 8192 MB (8 GB)
     - Maximum size: 16384 MB (16 GB)
   - Click "Set" and then "OK" on all windows
   - Restart your computer

2. Alternatively, try reducing memory usage:
   - Use smaller batch sizes (2 or 4 instead of 8)
   - Use smaller image sizes (320 or 416 instead of 640)
   - Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

### 2. PyTorch Not Built with CUDA Support

**Symptoms:**
- `torch.cuda.is_available()` returns `False`
- PyTorch was installed without CUDA support

**Solution:**
```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Missing or Outdated NVIDIA Drivers

**Symptoms:**
- `nvidia-smi` command fails
- System doesn't recognize your GPU

**Solution:**
1. Download the latest drivers from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)
2. Select:
   - Product Type: GeForce
   - Product Series: GeForce GTX 16 Series
   - Product: GeForce GTX 1650
   - Operating System: Your OS version
3. Install the downloaded driver

### 4. Missing CUDA Toolkit

**Symptoms:**
- CUDA_PATH environment variable not set
- CUDA libraries not found

**Solution:**
1. Download CUDA Toolkit 11.8 from [NVIDIA's website](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Follow the installation instructions
3. Make sure the CUDA_PATH environment variable is set

### 5. Environment Variables Not Set Correctly

**Symptoms:**
- CUDA is installed but not found by PyTorch

**Solution:**
1. Add CUDA to your PATH environment variable:
   - Right-click on "This PC" or "My Computer"
   - Select "Properties"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Edit the "Path" variable and add:
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp`
2. Add CUDA_PATH environment variable:
   - Click "New" under System variables
   - Variable name: `CUDA_PATH`
   - Variable value: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

### 6. CUDA Version Mismatch

**Symptoms:**
- Error messages about CUDA version incompatibility

**Solution:**
- Make sure your PyTorch CUDA version matches your installed CUDA Toolkit
- For CUDA 11.8, use: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### 7. Not Enough GPU Memory

**Symptoms:**
- Out of memory errors during training
- CUDA initialization errors

**Solution:**
1. Reduce batch size (try 2 or 4 instead of 8)
2. Reduce image size (try 320 or 416 instead of 640)
3. Use a smaller model (like YOLOv8n instead of larger variants)
4. Set environment variable: `set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

## Verifying CUDA is Working

Run the following Python code to verify CUDA is working:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

## Still Having Issues?

If you're still experiencing problems after following these steps, try:

1. Restarting your computer
2. Checking for Windows updates
3. Updating your BIOS
4. Ensuring your laptop is using the NVIDIA GPU and not integrated graphics
5. Checking if your GPU is disabled in Device Manager 