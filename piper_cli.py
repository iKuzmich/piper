#!/usr/bin/env python3
"""
Entry point script for PyInstaller build of Piper TTS CLI.

This script properly imports and runs the piper main function,
avoiding relative import issues that occur when PyInstaller
tries to run __main__.py directly.
"""

import sys

if __name__ == '__main__':
    # Import and run piper's main function
    from piper.__main__ import main
    main()
