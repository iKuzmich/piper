"""
Setup script to download and configure NLTK data for Piper TTS.
This ensures punkt tokenizer data is available for the library.
"""
import os
import sys
from pathlib import Path


def setup_nltk_data():
    """Download required NLTK data packages."""
    try:
        import nltk

        # Try to find punkt data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            print("NLTK punkt data already installed.")
            return True
        except LookupError:
            print("Downloading NLTK punkt data...")

        # Download punkt data
        success = True
        for package in ['punkt', 'punkt_tab']:
            try:
                nltk.download(package, quiet=False)
                print(f"Successfully downloaded {package}")
            except Exception as e:
                print(f"Warning: Failed to download {package}: {e}")
                success = False

        if success:
            print("\nNLTK data setup completed successfully!")
            print(f"Data installed to: {nltk.data.path[0]}")
        else:
            print("\nWarning: Some NLTK packages failed to download.")
            print("You may need to run this script again with internet connection.")

        return success

    except ImportError:
        print("Error: NLTK not installed. Please install piper-tts first:")
        print("  pip install piper-tts")
        return False
    except Exception as e:
        print(f"Error during NLTK setup: {e}")
        return False


if __name__ == "__main__":
    success = setup_nltk_data()
    sys.exit(0 if success else 1)
