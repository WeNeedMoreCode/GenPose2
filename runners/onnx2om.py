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
    output_path=None,
    batch_size=None,  # None = auto-detect from metadata
    soc_version="Ascend310P3"
):
    """
    Convert ONNX model to OM format using ATC tool.

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to output OM model (auto-detected if not specified)
        batch_size: Batch size for ATC conversion (None = auto-detect from metadata)
        soc_version: SoC version (default: Ascend310P)
    """
    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_suffix('')
    else:
        output_path = Path(output_path)

    # Auto-detect batch_size from metadata
    metadata_path = onnx_path.parent / f"{onnx_path.stem}_metadata.json"
    detected_batch_size = None

    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        detected_batch_size = metadata.get('config', {}).get('om_batch_size', None)

    # Determine batch size to use
    if batch_size is None:
        if detected_batch_size is not None:
            batch_size = detected_batch_size
            print(f"Auto-detected batch_size from metadata: {batch_size}")
        else:
            batch_size = 1
            print(f"No batch_size found in metadata, using default: {batch_size}")
    else:
        # User specified batch_size explicitly
        if detected_batch_size is not None and batch_size != detected_batch_size:
            print(f"Warning: Specified batch_size ({batch_size}) differs from metadata ({detected_batch_size})")
        print(f"Using specified batch_size: {batch_size}")

    # Detect model type from filename
    if 'pointnet2_scorenet' in onnx_path.stem:
        model_type = 'pointnet2_scorenet'
    elif 'pointnet2_encoder' in onnx_path.stem:
        model_type = 'pointnet2_encoder'
    elif 'score' in onnx_path.stem:
        model_type = 'score'
    elif 'energy' in onnx_path.stem:
        model_type = 'energy'
    elif 'scale' in onnx_path.stem:
        model_type = 'scale'
    else:
        model_type = 'unknown'

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
        if model_type == 'score':
            print("  python runners/export_onnx.py --agent_type score --output_dir ./onnx_models")
        elif model_type == 'pointnet2_encoder':
            print("  python runners/export_onnx.py --agent_type pointnet2 --output_dir ./onnx_models")
        elif model_type == 'energy':
            print("  python runners/export_onnx.py --agent_type energy --output_dir ./onnx_models")
        elif model_type == 'scale':
            print("  python runners/export_onnx.py --agent_type scale --output_dir ./onnx_models")
        else:
            print("  python runners/export_onnx.py --agent_type [score|pointnet2|energy|scale] --output_dir ./onnx_models")
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
    # Input shapes:
    #   pointnet2_scorenet:
    #     pts:          [batch_size, 1024, 3] - raw point cloud
    #     rgb_feat:     [batch_size, 1024, 384] - DINOv2 features
    #     sampled_pose: [batch_size, 9]
    #     t:            [batch_size, 1]
    #   score_network (pointwise mode - rgb_feat already fused in pts_feat):
    #     pts_feat:     [batch_size, 1024] - contains RGB info
    #     sampled_pose: [batch_size, 9]
    #     t:            [batch_size, 1]
    #   energy_network:
    #     pts_feat:     [batch_size, 1024] - contains RGB info
    #     sampled_pose: [batch_size, 9]
    #     t:            [batch_size, 1]
    #   scale_network:
    #     pts_feat:     [batch_size, 1024] - contains RGB info
    #     axes:         [batch_size, 3, 3] - rotation matrices

    # Set input shapes based on model type
    if model_type == 'pointnet2_scorenet':
        input_shapes = {
            'pts': f"{batch_size},1024,3",
            'rgb_feat': f"{batch_size},1024,384",
            'sampled_pose': f"{batch_size},9",
            't': f"{batch_size},1"
        }
    elif model_type == 'pointnet2_encoder':
        input_shapes = {
            'pts': f"{batch_size},1024,3",
            'rgb_feat': f"{batch_size},1024,384"
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
        f"--soc_version={soc_version}"
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
                        default='./onnx_models/score_network.onnx',
                        help='Path to input ONNX model (score_network.onnx or energy_network.onnx)')
    parser.add_argument('--output', type=str,
                        dest='output_path',
                        default=None,
                        help='Path to output OM model (auto-detected from onnx_path if not specified)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for ATC conversion (default: auto-detect from metadata)')
    parser.add_argument('--soc_version', type=str, default='Ascend310P3',
                        help='SoC version (default: Ascend310P3)')

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
