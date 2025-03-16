@echo off
echo Fixing optree warning for PyTorch...
echo.

pip install --upgrade "optree>=0.13.0"

echo.
echo Optree has been updated. This should fix the warning:
echo "optree is installed but the version is too old to support PyTorch Dynamo in C++ pytree"
echo.
pause 