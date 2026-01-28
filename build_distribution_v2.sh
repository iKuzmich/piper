#!/usr/bin/env bash
set -e

# Build Piper TTS

echo "=================================================="
echo "Building Piper TTS"
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
    apt-get install -y cmake ninja-build python3-dev python3-pip python3.12-venv
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

# Activate the virtual environment created by script/setup
echo ""
echo "Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated: $(which python3)"
else
    echo "ERROR: Virtual environment not found at .venv/bin/activate"
    exit 1
fi

# Step 4: Package the Python wheel
echo ""
echo "Step 4: Packaging Python wheel..."
./script/package

# Step 5: Install the wheel
echo ""
echo "Step 5: Installing piper package..."
pip install --upgrade dist/piper_tts-*.whl

# Step 6: Download NLTK data
echo ""
echo "Step 6: Downloading NLTK data..."
python3 setup_nltk_data.py

