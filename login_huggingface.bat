@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\hf.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
echo Accept model access terms at https://huggingface.co/google/translategemma-12b-it
echo Then enter a Hugging Face read token below. Do not paste it into source files.
".venv\Scripts\hf.exe" auth login
pause
