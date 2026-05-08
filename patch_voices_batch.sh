#!/usr/bin/env bash
# Applies patch_voice_with_alignment.py to every .onnx file found under BASE_DIR.
# Usage: ./patch_voices_batch.sh <BASE_DIR> [PATCH_SCRIPT]
#   BASE_DIR     - root folder containing model subdirectories (e.g. /tmp/models)
#   PATCH_SCRIPT - path to patch_voice_with_alignment.py (default: script's own directory)

set -euo pipefail

BASE_DIR="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SCRIPT="${2:-"$SCRIPT_DIR/src/piper/patch_voice_with_alignment.py"}"

if [[ -z "$BASE_DIR" ]]; then
    echo "Usage: $(basename "$0") <BASE_DIR> [PATCH_SCRIPT]" >&2
    exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: directory not found: $BASE_DIR" >&2
    exit 1
fi

if [[ ! -f "$PATCH_SCRIPT" ]]; then
    echo "Error: patch script not found: $PATCH_SCRIPT" >&2
    exit 1
fi

mapfile -t ONNX_FILES < <(find "$BASE_DIR" -type f -name "*.onnx" | sort)

if [[ ${#ONNX_FILES[@]} -eq 0 ]]; then
    echo "No .onnx files found under $BASE_DIR"
    exit 0
fi

echo "Found ${#ONNX_FILES[@]} .onnx file(s) under $BASE_DIR"
echo

PASSED=0
FAILED=0
SKIPPED=0

for ONNX_FILE in "${ONNX_FILES[@]}"; do
    echo "Processing: $ONNX_FILE"
    if python3 -c "import sys; sys.argv = [sys.argv[1]] + sys.argv[2:]; exec(compile(open(sys.argv[0]).read(), sys.argv[0], 'exec'))" "$PATCH_SCRIPT" "$ONNX_FILE"; then
        echo "  OK"
        (( PASSED++ )) || true
    else
        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 1 ]]; then
            # patch_voice_with_alignment.py returns 1 for "already patched" or detection errors
            echo "  SKIPPED (already patched or detection error)"
            (( SKIPPED++ )) || true
        else
            echo "  FAILED (exit code $EXIT_CODE)" >&2
            (( FAILED++ )) || true
        fi
    fi
done

echo
echo "Done. Passed: $PASSED  Skipped: $SKIPPED  Failed: $FAILED"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
