@echo off
REM Creates a clean virtual environment and installs only the required packages.
REM Run this once from the project root before training.

set VENV_DIR=.venv

if exist "%VENV_DIR%" (
    echo Virtual environment already exists at %VENV_DIR%. Delete it to recreate.
    goto :activate
)

echo Creating virtual environment...
python -m venv %VENV_DIR%
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment. Ensure Python 3.11+ is installed.
    exit /b 1
)

:activate
echo Activating virtual environment and installing dependencies...
call "%VENV_DIR%\Scripts\activate.bat"
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete. To train the model, run:
echo   call .venv\Scripts\activate.bat
echo   python train\export_fusion_bundle.py --data-dir data --output model_artifacts\fusion_bundle.joblib
