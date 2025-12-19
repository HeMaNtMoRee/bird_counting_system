#!/bin/bash

echo "=========================================="
echo "Bird Counting System - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✓ Python $python_version detected"
else
    echo "✗ Python 3.9+ required, found $python_version"
    exit 1
fi

echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠ Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"

echo ""

# Install requirements
echo "Installing dependencies (this may take a few minutes)..."
echo "  - FastAPI, Uvicorn"
echo "  - YOLOv8 (Ultralytics)"
echo "  - OpenCV, NumPy, Pandas"
echo "  - PyTorch (CPU version)"
echo ""

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed successfully"
else
    echo "✗ Installation failed"
    exit 1
fi

echo ""

# Create output directory
echo "Creating output directory..."
mkdir -p output
echo "✓ Output directory created"

echo ""

# Download YOLOv8 model
echo "Downloading YOLOv8n model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" > /dev/null 2>&1
echo "✓ YOLOv8n model downloaded"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start the API server:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run server: python main.py"
echo "  3. Access API docs: http://localhost:8000/docs"
echo ""
echo "To test the API:"
echo "  python test_api.py"
echo ""
echo "For more information, see README.md"
echo ""