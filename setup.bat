@echo off
echo ==========================================
echo Bird Counting System - Setup Script
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python not found. Please install Python 3.9+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo + Python %python_version% detected
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo ! Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo + Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo + Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo + Pip upgraded
echo.

REM Install requirements
echo Installing dependencies (this may take a few minutes)...
echo   - FastAPI, Uvicorn
echo   - YOLOv8 (Ultralytics)
echo   - OpenCV, NumPy, Pandas
echo   - PyTorch (CPU version)
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo X Installation failed
    pause
    exit /b 1
)
echo + All dependencies installed successfully
echo.

REM Create output directory
echo Creating output directory...
if not exist output mkdir output
echo + Output directory created
echo.

REM Download YOLOv8 model
echo Downloading YOLOv8n model...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" >nul 2>&1
echo + YOLOv8n model downloaded
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To start the API server:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run server: python main.py
echo   3. Access API docs: http://localhost:8000/docs
echo.
echo To test the API:
echo   python test_api.py
echo.
echo For more information, see README.md
echo.
pause