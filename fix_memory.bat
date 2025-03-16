@echo off
echo ===== VIRTUAL MEMORY FIX FOR CUDA OPERATIONS =====
echo.
echo Your system needs more virtual memory (paging file) to run CUDA operations.
echo.
echo Follow these steps to increase your virtual memory:
echo.
echo 1. Right-click on "This PC" or "My Computer" and select "Properties"
echo 2. Click on "Advanced system settings" on the left
echo 3. In the System Properties window, go to the "Advanced" tab
echo 4. Click on "Settings" under the "Performance" section
echo 5. Go to the "Advanced" tab in the Performance Options window
echo 6. Click on "Change" under the "Virtual memory" section
echo 7. Uncheck "Automatically manage paging file size for all drives"
echo 8. Select your system drive (usually C:)
echo 9. Select "Custom size" and set:
echo    - Initial size: 8192 MB (8 GB)
echo    - Maximum size: 16384 MB (16 GB)
echo 10. Click "Set" and then "OK" on all windows
echo 11. Restart your computer
echo.
echo After restarting, run the check_cuda.bat script again to verify CUDA is working.
echo.
pause 