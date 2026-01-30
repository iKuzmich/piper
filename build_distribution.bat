@echo off
REM Build script for creating Piper TTS distribution package

echo ========================================
echo Piper TTS Distribution Builder
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

echo [1/5] Installing build dependencies...
python -m pip install --upgrade pip
python -m pip install build wheel scikit-build cmake ninja
if errorlevel 1 (
    echo ERROR: Failed to install build dependencies
    pause
    exit /b 1
)

echo.
echo [2/5] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist src\piper.egg-info rmdir /s /q src\piper.egg-info

echo.
echo [3/5] Building wheel package with CMake (this may take several minutes)...
echo This will compile the espeakbridge C extension and download espeak-ng...
python setup.py bdist_wheel
if errorlevel 1 (
    echo ERROR: Build failed
    echo.
    echo Common issues:
    echo - Visual Studio Build Tools not installed
    echo - CMake not in PATH
    echo - Git not available (needed to download espeak-ng)
    pause
    exit /b 1
)

echo.
echo [4/5] Setting up NLTK data...
python setup_nltk_data.py
if errorlevel 1 (
    echo WARNING: NLTK data setup had issues
    echo You may need to run setup_nltk_data.py manually
)

echo.
echo [5/5] Verifying build...
echo.
echo Checking for compiled extension in build output...
if exist "_skbuild\win*\cmake-build\espeakbridge.pyd" (
    echo [OK] espeakbridge.pyd was compiled successfully
) else (
    echo [WARNING] espeakbridge.pyd not found in build directory
)
echo.
echo Build complete!
echo.
echo Distribution package created in: dist\
dir dist\*.whl
echo.
echo ========================================
echo Next Steps:
echo ========================================
echo 1. Install the wheel package:
echo    pip install --force-reinstall dist\piper_tts-1.3.1-*.whl
echo.
echo    Note: The wheel should be platform-specific (e.g., win_amd64.whl)
echo    NOT py3-none-any.whl
echo.
echo 2. Download a voice model from:
echo    https://github.com/rhasspy/piper/releases
echo.
echo 3. Test the installation:
echo    piper --version
echo.
echo 4. For .NET integration, see:
echo    DOTNET_INTEGRATION.md
echo ========================================
echo.

pause
