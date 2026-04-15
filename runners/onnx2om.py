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
    output_dir=None,
    batch_size=1,
    soc_version="Ascend310P3"
):
    """
    Convert ONNX model to OM format using ATC tool.

    Args:
        onnx_path: Path to input ONNX model
        output_dir: Directory for output OM model (auto-detected if not specified)
        batch_size: Batch size for ATC conversion (default: 1)
        soc_version: SoC version (default: Ascend310P3)
    """
    onnx_path = Path(onnx_path)
    if output_dir is None:
        output_path = onnx_path.with_suffix('')
    else:
        output_dir = Path(output_dir)
        output_path = output_dir / onnx_path.stem

    # Detect model type from filename
    STEM_TO_TYPE = {
        'pointnet2_from_score': 'pointnet2',
        'pointnet2_from_energy': 'pointnet2',
        'scorenet': 'score',
        'energynet': 'energy',
        'scalenet': 'scale',
        'dinov2_vits14': 'dinov2',
    }
    model_type = STEM_TO_TYPE.get(onnx_path.stem)

    # Model-type default batch sizes
    DEFAULT_BATCH_SIZES = {
        'pointnet2': 16,
        'score': 800,
        'energy': 800,
        'scale': 16,
        'dinov2': 16,
    }

    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZES.get(model_type, 1)
        print(f"Using default batch_size for {model_type}: {batch_size}")

    print("=" * 60)
    print(f"GenPose2 {model_type.capitalize()} Network ONNX to OM Conversion")
    print("=" * 60)
    print()
    print("Parameters:")
    print(f"  Model type:     {model_type}")
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
        print("  python runners/export_onnx.py --agent_type [score|energy|scale|pointnet2_from_score|pointnet2_from_energy|dinov2] --output_dir ./onnx_models")
        return False

    # Check if ATC tool is available
    atc_check = subprocess.run(["which", "atc"], capture_output=True)
    if atc_check.returncode != 0:
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
    # Set input shapes based on model type
    if model_type == 'pointnet2':
        input_shapes = {
            'pointcloud': f"{batch_size},1024,387"
        }
    elif model_type in ['score', 'energy']:
        input_shapes = {
            'pts_feat': f"{batch_size},1024",
            'sampled_pose': f"{batch_size},9",
            't': f"{batch_size},1"
        }
    elif model_type == 'scale':
        input_shapes = {
            'pts_feat': f"{batch_size},1024",
            'axes': f"{batch_size},3,3"
        }
    elif model_type == 'dinov2':
        input_shapes = {
            'roi_rgb': f"{batch_size},3,224,224",
            'roi_xs': f"{batch_size},1024",
            'roi_ys': f"{batch_size},1024",
        }
    else:
        print(f"Error: Unknown model type '{model_type}'")
        return False
    input_shape_str = ";".join([f"{k}:{v}" for k, v in input_shapes.items()])

    atc_cmd = [
        "atc",
        "--framework=5",
        f"--model={onnx_path}",
        f"--output={output_path}",
        "--input_format=NCHW",
        f"--input_shape={input_shape_str}",
        "--log=error",
        f"--soc_version={soc_version}",
    ]

    # Run ATC conversion
    print("Running ATC conversion...")
    print("This may take a few minutes...")
    print()

    result = subprocess.run(atc_cmd, capture_output=True, text=True)
    # ATC creates output file with .om extension
    output_file = Path(str(output_path) + ".om")
    # Check if file was created
    if output_file.exists():
        print()
        print("=" * 60)
        print("✓ OM conversion successful!")
        print(f"  Output: {output_file}")
        print()

        # Get file size
        file_size = output_file.stat().st_size
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
        if result.stderr or result.stdout:
            print("Stderr:")
            print(result.stderr)
            print('Stdout:')
            print(result.stdout)
        print("=" * 60)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert GenPose2 ONNX models to OM format using ATC tool'
    )
    parser.add_argument('--onnx_path', type=str,
                        default='./onnx_models/scorenet.onnx',
                        help='Path to input ONNX model (scorenet.onnx, pointnet2_from_score.onnx, etc.)')
    parser.add_argument('--output', type=str,
                        dest='output_dir',
                        default='om_models',
                        help='Directory for output OM model (default: om_models)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for ATC conversion (default: model-type specific)')
    parser.add_argument('--soc_version', type=str, default='Ascend310P3',
                        help='SoC version (default: Ascend310P3)')

    args = parser.parse_args()

    success = convert_onnx_to_om(
        onnx_path=args.onnx_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        soc_version=args.soc_version
    )

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
