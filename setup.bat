@echo off
setlocal
cd /d "%~dp0"
set "CONDA_EXE_PATH=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE_PATH%" set "CONDA_EXE_PATH=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE_PATH%" (
    echo Anaconda/Miniconda not found in your user folder.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    call "%CONDA_EXE_PATH%" create -n youtube-translator-bootstrap python=3.11 --no-default-packages --override-channels -c conda-forge -y
    if errorlevel 1 goto :error
    call "%CONDA_EXE_PATH%" run -n youtube-translator-bootstrap python -m venv "%~dp0.venv"
    if errorlevel 1 goto :error
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :error
".venv\Scripts\python.exe" setup_runtime.py
if errorlevel 1 goto :error
echo Setup complete. Run run.bat to launch.
pause
exit /b 0
:error
echo Setup failed. See the error above.
pause
exit /b 1
