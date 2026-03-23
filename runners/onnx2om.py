"""
GenPose2 ONNX to OM Conversion Script

Converts ONNX models to OM (Offline Model) format for NPU inference
using the Huawei ATC tool.

Usage:
    python runners/onnx2om.py
    python runners/onnx2om.py --batch_size 4
    python runners/onnx2om.py --soc_version Ascend310P
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def convert_onnx_to_om(
    onnx_path="./onnx_models/score_network.onnx",
    output_path="./onnx_models/score_network.om",
    batch_size=1,
    soc_version="Ascend310P"
):
    """
    Convert ONNX model to OM format using ATC tool.

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to output OM model
        batch_size: Batch size for ATC conversion (default: 1)
        soc_version: SoC version (default: Ascend310P)
    """
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)

    print("=" * 60)
    print("GenPose2 Score Network ONNX to OM Conversion")
    print("=" * 60)
    print()
    print("Parameters:")
    print(f"  ONNX model:     {onnx_path}")
    print(f"  OM output:      {output_path}")
    print(f"  Batch size:     {batch_size}")
    print(f"  SoC version:    {soc_version}")
    print()

    # Check if ONNX model exists
    if not onnx_path.exists():
        print(f"Error: ONNX model not found: {onnx_path}")
        print()
        print("Please run export first:")
        print("  python runners/export_onnx.py --agent_type score --output_dir ./onnx_models")
        return False

    # Check if ATC tool is available
    try:
        subprocess.run(["which", "atc"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Error: ATC tool not found!")
        print()
        print("Please source the environment first:")
        print("  source /usr/local/Ascend/ascend-toolkit/set_env.sh")
        print()
        print("Or if using a different path, adjust accordingly.")
        return False

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare ATC command
    # Input shapes (with dynamic batch dimension):
    #   pts_feat:     [batch_size, 1024]
    #   rgb_feat:     [batch_size, 384]
    #   sampled_pose: [batch_size, 9]
    #   t:            [batch_size, 1]

    input_shapes = {
        'pts_feat': f"{batch_size},1024",
        'rgb_feat': f"{batch_size},384",
        'sampled_pose': f"{batch_size},9",
        't': f"{batch_size},1"
    }
    input_shape_str = ";".join([f"{k}:{v}" for k, v in input_shapes.items()])

    atc_cmd = [
        "atc",
        "--framework=5",
        f"--model={onnx_path}",
        f"--output={output_path}",
        "--input_format=NCHW",
        f"--input_shape={input_shape_str}",
        "--log=error",
        f"--soc_version={soc_version}"
    ]

    # Run ATC conversion
    print("Running ATC conversion...")
    print("This may take a few minutes...")
    print()

    try:
        result = subprocess.run(
            atc_cmd,
            check=True,
            capture_output=True,
            text=True
        )

        # ATC doesn't output to stdout normally, check if file was created
        if output_path.exists():
            print()
            print("=" * 60)
            print("✓ OM conversion successful!")
            print(f"  Output: {output_path}")
            print()

            # Get file size
            file_size = output_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
            print()
            print("Next steps:")
            print("  1. Use the OM model in NPU inference (TODO: implement infer_om.py)")
            print("  2. Test with sample data to verify correctness")
            print("=" * 60)
            return True
        else:
            print()
            print("=" * 60)
            print("✗ OM conversion failed!")
            print("  Output file not created")
            print()
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            print("=" * 60)
            return False

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("✗ ATC conversion failed!")
        print(f"  Return code: {e.returncode}")
        print()

        if e.stderr:
            print("Error output:")
            print(e.stderr)
        print("=" * 60)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert GenPose2 ONNX models to OM format using ATC tool'
    )
    parser.add_argument('--onnx_path', type=str,
                        default='./onnx_models/score_network.onnx',
                        help='Path to input ONNX model')
    parser.add_argument('--output', type=str,
                        dest='output_path',
                        default='./onnx_models/score_network.om',
                        help='Path to output OM model')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for ATC conversion (default: 1)')
    parser.add_argument('--soc_version', type=str, default='Ascend310P',
                        help='SoC version (default: Ascend310P)')

    args = parser.parse_args()

    success = convert_onnx_to_om(
        onnx_path=args.onnx_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        soc_version=args.soc_version
    )

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
