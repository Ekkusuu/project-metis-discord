@echo off
REM install_python_requirements.bat — install Python requirements and PyTorch CUDA 12.6
REM Usage: run from project root in cmd.exe. Activate your virtualenv first if needed.

echo NOTE: Activate your Python virtualenv before running this script (optional but recommended).
echo Running pip upgrade and installing packages from requirements.txt

python -m pip install --upgrade pip
IF %ERRORLEVEL% NEQ 0 (
    echo pip upgrade failed; continuing anyway...
)

echo Installing packages from requirements.txt
pip install -r "%~dp0requirements.txt"
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies from requirements.txt
    echo You can try running: pip install -r requirements.txt
    exit /b %ERRORLEVEL%
)

echo Installing PyTorch wheels for CUDA 12.6 (torch, torchvision, torchaudio)
echo This uses the official PyTorch index for cu126 wheels.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
IF %ERRORLEVEL% NEQ 0 (
    echo PyTorch install failed. If you don't have a compatible NVIDIA driver or CUDA, consider installing the CPU-only wheel or check the error message above.
    echo Suggested verification command:
    echo python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
    exit /b %ERRORLEVEL%
)

echo All Python packages installed. Verify PyTorch with:
echo python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

exit /b 0
