#!/usr/bin/env python3
"""Convert Piper voice models (.onnx) to ORT-format (.ort) siblings for mmap-shared loading.

piper_create() in src/piper.cpp looks for a "<model>.ort" file next to a
requested "<model>.onnx" and, if present, loads it with
session.use_memory_mapped_ort_model + session.use_ort_model_bytes_for_initializers
so that sibling PiperWorker processes on the same AKS node loading the same
voice share the underlying RAM pages via the OS page cache instead of each
holding a private copy.

That sharing only survives if the .ort file is already fully optimized: at
load time piper.cpp sets GraphOptimizationLevel::ORT_DISABLE_ALL and
session.disable_prepacking=1, so no optimization or weight-repacking runs
per-process anymore -- it must all be baked into the file up front, which is
exactly what --optimization_style Fixed does here.

Usage:
    python -m pip install onnxruntime==1.28.0
    python convert_to_ort.py <path to .onnx file or directory of .onnx files>

Requires: onnxruntime installed (version should match ONNXRUNTIME_VERSION in
CMakeLists.txt -- the .ort flatbuffer format is not guaranteed compatible
across arbitrary ORT versions).
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path_or_dir",
        help="Path to a .onnx file, or a directory containing one or more .onnx voice models (searched recursively)",
    )
    parser.add_argument(
        "--target_platform",
        choices=["arm", "amd64"],
        default="amd64",
        help="Target platform the .ort file will run on (Skidbladnir's AKS nodes are amd64).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path_or_dir):
        print(f"error: path does not exist: {args.model_path_or_dir}", file=sys.stderr)
        sys.exit(1)

    # optimization_style=Fixed bakes the optimized graph into the .ort file itself, so
    # nothing needs to run again at PiperWorker session-creation time. The optimization
    # level applied is controlled by this env var (default "all" == ORT_ENABLE_ALL) --
    # set explicitly here so the behavior doesn't silently drift with the caller's shell.
    env = dict(os.environ)
    env.setdefault("ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL", "all")

    cmd = [
        sys.executable,
        "-m",
        "onnxruntime.tools.convert_onnx_models_to_ort",
        args.model_path_or_dir,
        "--optimization_style",
        "Fixed",
        "--target_platform",
        args.target_platform,
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

    print(
        "Done. Each <model>.onnx now has a sibling <model>.ort next to it -- "
        "stage both onto the node-cache tmpfs; piper_create() will prefer the .ort file automatically."
    )

if __name__ == "__main__":
    main()
