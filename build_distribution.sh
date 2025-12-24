#!/bin/bash

# Build script for creating Piper TTS distribution package

echo "========================================"
echo "Piper TTS Distribution Builder"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

echo "[1/5] Installing build dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install build wheel scikit-build cmake ninja
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install build dependencies"
    exit 1
fi

echo ""
echo "[2/5] Cleaning previous builds..."
rm -rf build dist src/piper.egg-info

echo ""
echo "[3/5] Building wheel package..."
python3 -m build
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "[4/5] Setting up NLTK data..."
python3 setup_nltk_data.py
if [ $? -ne 0 ]; then
    echo "WARNING: NLTK data setup had issues"
    echo "You may need to run setup_nltk_data.py manually"
fi

echo ""
echo "[5/5] Build complete!"
echo ""
echo "Distribution package created in: dist/"
ls -lh dist/*.whl
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. Install the wheel package:"
echo "   pip install dist/piper_tts-1.3.1-*.whl"
echo ""
echo "2. Download a voice model from:"
echo "   https://github.com/rhasspy/piper/releases"
echo ""
echo "3. Test the installation:"
echo "   piper --version"
echo ""
echo "4. For .NET integration, see:"
echo "   DOTNET_INTEGRATION.md"
echo "========================================"
echo ""
