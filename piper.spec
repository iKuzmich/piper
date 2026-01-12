# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for building Piper TTS CLI as a native Linux executable.

This spec file bundles:
- Python code (piper package)
- Native libraries (libpiper.so/espeakbridge, onnxruntime)
- espeak-ng-data directory
- NLTK data (punkt tokenizer)
- Tashkeel model data

Build Instructions:
===================

1. On Linux (or Linux VM/Docker container):

   # Install system dependencies
   sudo apt-get update
   sudo apt-get install -y build-essential cmake ninja-build python3-dev python3-pip

   # Build the project first (this compiles C++ extensions and downloads dependencies)
   ./script/setup --dev
   ./script/dev_build
   ./script/package

   # Install the built wheel
   pip install dist/piper_tts-*.whl

   # Download NLTK data
   python setup_nltk_data.py

   # Install PyInstaller
   pip install pyinstaller

   # Build the executable
   pyinstaller piper.spec

   # The executable will be in: dist/piper/piper

2. For cross-platform build (build Linux executable from Windows):
   Use Docker:

   docker run -it --rm -v "$(pwd):/work" -w /work python:3.12 bash
   # Then run the commands from step 1 above

The resulting executable can run standalone on Linux with all dependencies bundled.
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Find the piper package installation
piper_package = None
nltk_data_path = None

try:
    import piper
    piper_package = Path(piper.__file__).parent
    print(f"Found piper package at: {piper_package}")
except ImportError:
    print("ERROR: piper package not installed. Please install it first:")
    print("  1. Build the project: ./script/setup --dev && ./script/dev_build && ./script/package")
    print("  2. Install: pip install dist/piper_tts-*.whl")
    sys.exit(1)

try:
    import nltk
    # Get NLTK data paths
    nltk_data_paths = nltk.data.path
    for path in nltk_data_paths:
        punkt_path = Path(path) / 'tokenizers' / 'punkt'
        punkt_tab_path = Path(path) / 'tokenizers' / 'punkt_tab'
        if punkt_path.exists() and punkt_tab_path.exists():
            nltk_data_path = Path(path)
            print(f"Found NLTK data at: {nltk_data_path}")
            break

    if not nltk_data_path:
        print("WARNING: NLTK punkt data not found. Run: python setup_nltk_data.py")
        print("The executable will work but may fail on some text processing tasks.")
except ImportError:
    print("WARNING: NLTK not installed")

# Collect all piper package data
piper_datas = []

# Add espeak-ng-data directory (CRITICAL for phonemization)
espeak_data = piper_package / 'espeak-ng-data'
if espeak_data.exists():
    piper_datas.append((str(espeak_data), 'piper/espeak-ng-data'))
    print(f"✓ Adding espeak-ng-data from: {espeak_data}")
else:
    print(f"ERROR: espeak-ng-data not found at {espeak_data}")
    print("This is required! Make sure you've built the project correctly.")
    sys.exit(1)

# Add tashkeel model data (for Arabic diacritization)
tashkeel_dir = piper_package / 'tashkeel'
if tashkeel_dir.exists():
    for file in ['model.onnx', 'input_id_map.json', 'target_id_map.json', 'hint_id_map.json']:
        file_path = tashkeel_dir / file
        if file_path.exists():
            piper_datas.append((str(file_path), 'piper/tashkeel'))
    print(f"✓ Adding tashkeel data from: {tashkeel_dir}")

# Add other piper data files
for file in piper_package.glob('*.pyi'):
    piper_datas.append((str(file), 'piper'))

# Add NLTK data if found
if nltk_data_path:
    punkt_dir = nltk_data_path / 'tokenizers' / 'punkt'
    punkt_tab_dir = nltk_data_path / 'tokenizers' / 'punkt_tab'
    if punkt_dir.exists():
        piper_datas.append((str(punkt_dir), 'nltk_data/tokenizers/punkt'))
        print(f"✓ Adding NLTK punkt data")
    if punkt_tab_dir.exists():
        piper_datas.append((str(punkt_tab_dir), 'nltk_data/tokenizers/punkt_tab'))
        print(f"✓ Adding NLTK punkt_tab data")

# Collect hidden imports
hiddenimports = [
    'piper',
    'piper.__main__',
    'piper.voice',
    'piper.config',
    'piper.types',
    'piper.phonemize_espeak',
    'piper.phoneme_ids',
    'piper.audio_playback',
    'piper.tashkeel',
    'onnxruntime',
    'nltk',
    'nltk.tokenize',
    'nltk.tokenize.punkt',
    'wave',
    'json',
    'pathlib',
]

# Binaries to include (native libraries)
binaries = []

# Find espeakbridge library (CRITICAL - this is the phonemizer)
espeakbridge_patterns = ['espeakbridge*.so', 'espeakbridge*.pyd', '_espeakbridge*.so', 'espeakbridge*.dylib']
found_espeakbridge = False
for pattern in espeakbridge_patterns:
    for lib_file in piper_package.glob(pattern):
        binaries.append((str(lib_file), 'piper'))
        print(f"✓ Adding espeakbridge library: {lib_file}")
        found_espeakbridge = True

if not found_espeakbridge:
    print("ERROR: espeakbridge library not found!")
    print("This is required! Make sure you've built the project correctly:")
    print("  ./script/setup --dev && ./script/dev_build")
    sys.exit(1)

# Find onnxruntime libraries (for model inference)
try:
    import onnxruntime
    onnx_path = Path(onnxruntime.__file__).parent

    # Add onnxruntime shared libraries
    lib_patterns = ['libonnxruntime.so*', 'libonnxruntime*.dylib', 'onnxruntime.dll']
    for lib_pattern in lib_patterns:
        for lib_file in onnx_path.glob(lib_pattern):
            if lib_file.is_file() and not lib_file.is_symlink():
                binaries.append((str(lib_file), 'onnxruntime/capi'))
                print(f"✓ Adding onnxruntime library: {lib_file.name}")

    # Also check in capi subdirectory
    capi_path = onnx_path / 'capi'
    if capi_path.exists():
        for lib_pattern in lib_patterns:
            for lib_file in capi_path.glob(lib_pattern):
                if lib_file.is_file() and not lib_file.is_symlink():
                    binaries.append((str(lib_file), 'onnxruntime/capi'))
                    print(f"✓ Adding onnxruntime library: {lib_file.name}")

except ImportError:
    print("ERROR: onnxruntime not found!")
    print("Install it with: pip install onnxruntime")
    sys.exit(1)

a = Analysis(
    ['piper_cli.py'],
    pathex=[],
    binaries=binaries,
    datas=piper_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude large packages not needed for inference
        'torch',
        'tensorflow',
        'matplotlib',
        'IPython',
        'jupyter',
        'pandas',
        'scipy',
        'sklearn',
        'cv2',
        'PIL',
        # Exclude training modules
        'piper.train',
        'lightning',
        'tensorboard',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='piper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='piper',
)
