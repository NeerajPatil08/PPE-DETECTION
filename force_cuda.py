import os
import sys
import torch
import subprocess

def set_cuda_env_vars():
    """Set environment variables to force CUDA usage."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Try to detect CUDA path
    potential_cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0',
    ]
    
    for path in potential_cuda_paths:
        if os.path.exists(path):
            os.environ['CUDA_PATH'] = path
            cuda_bin = os.path.join(path, 'bin')
            
            # Add to PATH if not already there
            if cuda_bin not in os.environ['PATH']:
                os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']
            
            print(f"Found CUDA at: {path}")
            break

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available and working."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               universal_newlines=True)
        print("NVIDIA GPU detected:")
        print(result.stdout)
        return True
    except:
        print("NVIDIA GPU not detected or nvidia-smi failed to run")
        return False

def check_pytorch_cuda():
    """Check if PyTorch can use CUDA."""
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Try to create a CUDA tensor to verify it works
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            print(f"Successfully created CUDA tensor: {x}")
            return True
        except Exception as e:
            print(f"Error creating CUDA tensor: {e}")
            return False
    else:
        return False

def main():
    """Main function to force CUDA usage."""
    print("\n===== FORCING CUDA USAGE =====\n")
    
    # Set environment variables
    set_cuda_env_vars()
    
    # Check NVIDIA GPU
    gpu_available = check_nvidia_gpu()
    
    # Check PyTorch CUDA
    cuda_working = check_pytorch_cuda()
    
    if gpu_available and cuda_working:
        print("\n✅ CUDA is working correctly with your NVIDIA GTX 1650!")
        print("You can now run the PPE detection application with GPU acceleration.")
    else:
        print("\n❌ CUDA is still not working correctly.")
        print("Please follow the instructions in CUDA_TROUBLESHOOTING.md")
        print("You may need to reinstall PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main() 