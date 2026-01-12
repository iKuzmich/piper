#!/usr/bin/env bash
set -e

#######################################################################
# Build Piper TTS as a Native Linux Executable
#######################################################################
#
# This script builds Piper TTS into a standalone Linux executable
# that can be deployed to Azure AppService or any Linux environment.
#
# The resulting executable will include:
# - Python runtime
# - All Python dependencies (onnxruntime, nltk, etc.)
# - Native libraries (espeakbridge, espeak-ng)
# - espeak-ng-data directory
# - NLTK data
#
# Usage:
#   ./build_linux_executable.sh
#
# Requirements:
#   - Linux environment (Ubuntu 20.04+ recommended)
#   - Python 3.9+
#   - Build tools (gcc, cmake, ninja)
#
# For building from Windows, use Docker:
#   docker run -it --rm -v "%cd%:/work" -w /work python:3.12 bash
#   # Then run this script inside the container
#
#######################################################################

echo "=================================================="
echo "Building Piper TTS Native Linux Executable"
echo "=================================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "ERROR: This script must be run on Linux!"
    echo ""
    echo "If you're on Windows, use Docker:"
    echo "  docker run -it --rm -v \"\$(pwd):/work\" -w /work python:3.12 bash"
    echo "  # Then run this script inside the container"
    exit 1
fi

# Step 1: Install system dependencies
echo "Step 1: Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y build-essential cmake ninja-build python3-dev python3-pip python3.12-venv
elif command -v yum &> /dev/null; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y cmake ninja-build python3-devel python3-pip python3.12-venv
else
    echo "WARNING: Could not detect package manager. Please install manually:"
    echo "  - build-essential / gcc/g++"
    echo "  - cmake (3.18+)"
    echo "  - ninja-build"
    echo "  - python3-dev"
    echo ""
fi

# Step 2: Clean up any existing virtual environment
echo ""
echo "Step 2: Preparing build environment..."
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Step 3: Build the C++ components
echo ""
echo "Step 3: Building C++ components (libpiper, espeak-ng)..."
echo "This may take 5-10 minutes..."
./script/setup --dev
./script/dev_build

# Step 4: Package the Python wheel
echo ""
echo "Step 4: Packaging Python wheel..."
./script/package

# Step 5: Install the wheel
echo ""
echo "Step 5: Installing piper package..."
pip install dist/piper_tts-*.whl

# Step 6: Download NLTK data
echo ""
echo "Step 6: Downloading NLTK data..."
python3 setup_nltk_data.py

# Step 7: Install PyInstaller
echo ""
echo "Step 7: Installing PyInstaller..."
pip install pyinstaller

# Step 8: Build the executable
echo ""
echo "Step 8: Building executable with PyInstaller..."
echo "This may take a few minutes..."
pyinstaller piper.spec

# Step 9: Test the executable
echo ""
echo "Step 9: Testing the executable..."
if [ -f "dist/piper/piper" ]; then
    echo "✓ Executable built successfully!"
    echo ""
    echo "Location: dist/piper/piper"
    echo "Size: $(du -h dist/piper/piper | cut -f1)"
    echo ""

    # Quick test
    if ./dist/piper/piper --help &> /dev/null; then
        echo "✓ Executable runs successfully!"
    else
        echo "WARNING: Executable test failed!"
    fi
else
    echo "ERROR: Executable not found at dist/piper/piper"
    exit 1
fi

# Step 10: Create distribution archive
echo ""
echo "Step 10: Creating distribution archive..."
cd dist
tar -czf piper-linux-x64.tar.gz piper/
cd ..

echo ""
echo "=================================================="
echo "Build Complete!"
echo "=================================================="
echo ""
echo "Executable location: dist/piper/piper"
echo "Archive location: dist/piper-linux-x64.tar.gz"
echo ""
echo "The 'dist/piper' directory contains:"
echo "  - piper (executable)"
echo "  - All required libraries"
echo "  - espeak-ng-data"
echo "  - NLTK data"
echo ""
echo "Deploy the entire 'dist/piper' directory to your target system."
echo ""
echo "Usage example:"
echo "  cd dist/piper"
echo "  ./piper -m <model.onnx> -f output.wav 'Hello, world!'"
echo ""
echo "For Azure AppService deployment, see: AZURE_DEPLOYMENT.md"
echo ""
