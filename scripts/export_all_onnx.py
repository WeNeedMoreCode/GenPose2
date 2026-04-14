"""
Batch export all ONNX models.
Skips models whose ONNX file already exists in the output directory.

Usage:
    python scripts/export_all_onnx.py
    python scripts/export_all_onnx.py --output_dir ./onnx_models
"""

import os
import subprocess
import sys
import argparse

# agent_type: output_filename
MODELS = [
    ("pointnet2_from_score", "pointnet2_from_score.onnx"),
    ("pointnet2_from_energy", "pointnet2_from_energy.onnx"),
    ("score", "scorenet.onnx"),
    ("energy", "energynet.onnx"),
    ("scale", "scalenet.onnx"),
    ("dinov2", "dinov2_vits14.onnx"),
]


def main():
    parser = argparse.ArgumentParser(description="Batch export all ONNX models")
    parser.add_argument("--output_dir", type=str, default="./onnx_models")
    parser.add_argument("--extra_args", nargs="*", default=[],
                        help="Extra args forwarded to export_onnx.py (e.g. --device cpu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    exported = 0
    skipped = 0

    for agent_type, filename in MODELS:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.isfile(filepath):
            print(f"[SKIP] {filename} already exists")
            skipped += 1
        else:
            print(f"\n>>> Exporting {agent_type}...")
            cmd = [
                sys.executable, "runners/export_onnx.py",
                "--agent_type", agent_type,
                "--output_dir", args.output_dir,
            ] + args.extra_args
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[FAIL] {agent_type} export failed")
            else:
                exported += 1

    print(f"\nDone. Exported: {exported}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
