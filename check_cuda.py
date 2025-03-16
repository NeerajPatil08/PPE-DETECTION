import torch
import sys
import subprocess
import os

def run_command(command):
    """Run a command and return its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               universal_newlines=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def check_cuda():
    """Check CUDA availability and print detailed information."""
    print("\n===== CUDA AVAILABILITY CHECK =====\n")
    
    # Check PyTorch version and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("\n===== TROUBLESHOOTING INFORMATION =====\n")
        
        # Check if CUDA is in the path
        cuda_path = os.environ.get('CUDA_PATH')
        print(f"CUDA_PATH environment variable: {cuda_path or 'Not set'}")
        
        # Check NVIDIA driver with nvidia-smi
        print("\nNVIDIA driver information:")
        nvidia_smi = run_command("nvidia-smi")
        print(nvidia_smi if nvidia_smi else "nvidia-smi not found or failed to run")
        
        # Check if PyTorch was built with CUDA
        print("\nPyTorch CUDA build information:")
        if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
            print(f"PyTorch CUDA version: {torch.version.cuda}")
        else:
            print("PyTorch was not built with CUDA support")
        
        # Check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        print(f"\nCUDA_VISIBLE_DEVICES: {cuda_visible or 'Not set'}")
        
        # Provide recommendations
        print("\n===== RECOMMENDATIONS =====\n")
        print("1. Make sure NVIDIA drivers are installed and up-to-date")
        print("2. Install PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Check if CUDA toolkit is installed")
        print("4. Make sure CUDA_PATH environment variable is set correctly")
        print("5. Try setting CUDA_VISIBLE_DEVICES=0 before running your script")

if __name__ == "__main__":
    check_cuda() 