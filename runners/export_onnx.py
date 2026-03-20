"""
GenPose2 ONNX Model Export Script

Exports PyTorch models to ONNX format for OM conversion.
Uses the decoupled ScoreNetworkWrapper for export.

Usage:
    python runners/export_onnx.py --agent_type score --output_dir ./onnx_models
    python runners/export_onnx.py --agent_type energy --output_dir ./onnx_models
    python runners/export_onnx.py --agent_type scale --output_dir ./onnx_models
"""

import sys
import os
import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs.config import get_config
from networks.score_wrapper import create_score_network

def dummy_remove_inplace_ops(graph, module):
    return graph

torch._C._jit_pass_remove_inplace_ops_for_onnx = dummy_remove_inplace_ops


def get_score_network_input_info(cfg):
    """
    Get input dimensions for ScoreNetworkWrapper.

    The ScoreNetworkWrapper expects:
        - pts_feat: [batch_size, 1024] - Point cloud features
        - rgb_feat: [batch_size, 384] - RGB features (DINOv2)
        - sampled_pose: [batch_size, 9] - Current pose estimate (rot_matrix format)
        - t: [batch_size, 1] - Diffusion timestep

    Returns:
        dict: Input information including shapes, dtypes, and names
    """
    return {
        'inputs': [
            {'name': 'pts_feat', 'shape': [1, 1024], 'dtype': 'float32', 'format': 'feature'},
            {'name': 'rgb_feat', 'shape': [1, 384], 'dtype': 'float32', 'format': 'feature'},
            {'name': 'sampled_pose', 'shape': [1, 9], 'dtype': 'float32', 'format': 'pose_rot_matrix'},
            {'name': 't', 'shape': [1, 1], 'dtype': 'float32', 'format': 'timestep'},
        ],
        'outputs': [
            {'name': 'score', 'shape': [1, 9], 'dtype': 'float32'},
        ],
        'metadata': {
            'export_type': 'score_network_wrapper',
            'pose_mode': cfg.pose_mode,
        }
    }


def export_score_network_to_onnx(checkpoint_path, output_dir, cfg, device='cpu'):
    """
    Export ScoreNetworkWrapper to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model and metadata
        cfg: Configuration object
        device: Device to load model on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting Score Network (ScoreNetworkWrapper) to ONNX")
    print(f"{'='*60}")

    # Load ScoreNetworkWrapper
    print(f"\nLoading ScoreNetworkWrapper...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")

    score_net = create_score_network(
        checkpoint_path=checkpoint_path,
        device=device
    )
    score_net.eval()

    # Get input info
    input_info = get_score_network_input_info(cfg)
    print(f"\nInput info:")
    for inp in input_info['inputs']:
        print(f"  {inp['name']}: {inp['shape']}, {inp['dtype']}")

    print(f"\nOutput info:")
    for out in input_info['outputs']:
        print(f"  {out['name']}: {out['shape']}, {out['dtype']}")

    # Prepare dummy inputs
    dummy_inputs = []
    for inp in input_info['inputs']:
        if inp['dtype'] == 'int64':
            dummy = torch.randint(0, 224, inp['shape'], dtype=torch.int64)
        else:
            dummy = torch.randn(inp['shape'], dtype=torch.float32)
        dummy_inputs.append(dummy)

    # Export to ONNX
    onnx_path = output_dir / "score_network.onnx"
    print(f"\nExporting to {onnx_path}...")

    dynamic_axes = {}
    input_names = [inp['name'] for inp in input_info['inputs']]
    output_names = [out['name'] for out in input_info['outputs']]

    # Add dynamic batch dimension
    for name in input_names:
        dynamic_axes[name] = {0: 'batch_size'}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    with torch.no_grad():
        torch.onnx.export(
            score_net,
            tuple(dummy_inputs),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            verbose=False,
            export_params=True,
            do_constant_folding=False,
            keep_initializers_as_inputs=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )
    print(f"✓ ONNX export successful: {onnx_path}")

    # Save metadata
    metadata_path = output_dir / "score_network_metadata.json"
    metadata = {
        'model_type': 'ScoreNetworkWrapper',
        'checkpoint_path': str(checkpoint_path),
        'onnx_path': str(onnx_path),
        'inputs': input_info['inputs'],
        'outputs': input_info['outputs'],
        'metadata': input_info['metadata'],
        'config': {
            'device': cfg.device,
            'pose_mode': cfg.pose_mode,
            'num_points': cfg.num_points,
            'dino': cfg.dino,
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Export GenPose2 Score Network to ONNX')
    parser.add_argument('--agent_type', type=str, default='score',
                        help='Agent type to export (currently only score is supported)')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                        help='Output directory for ONNX models')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint (auto-detected if not specified)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for export (default: cpu)')

    args = parser.parse_args()

    if args.agent_type != 'score':
        print(f"\nCurrently only score network export is supported.")
        print(f"Energy and scale network export will be added later.")
        return

    # Setup config
    sys.argv = [
        'export_onnx.py',
        '--data_path', './data',  # Dummy path, not used for export
        '--device', args.device,
    ]

    cfg = get_config()

    # Determine checkpoint path
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = getattr(cfg, 'pretrained_score_model_path',
                                    None) or './results/ckpts/ScoreNet/scorenet.pth'
        print(f"Auto-detected checkpoint: {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
        print(f"Please specify the correct path with --checkpoint_path")
        return

    # Export
    success = export_score_network_to_onnx(checkpoint_path, args.output_dir, cfg, args.device)

    if success:
        print(f"\n{'='*60}")
        print("Export completed successfully!")
        print(f"{'='*60}")
        print(f"\nExported files:")
        print(f"  - {args.output_dir}/score_network.onnx")
        print(f"  - {args.output_dir}/score_network_metadata.json")
        print(f"\nNext steps:")
        print(f"1. Convert ONNX to OM using ATC tool:")
        print(f"   atc --framework=5 --model={args.output_dir}/score_network.onnx \\")
        print(f"       --output={args.output_dir}/score_network.om \\")
        print(f"       --input_format=NCHW")
        print(f"\n2. Use the OM model in NPU inference (TODO: implement infer_om.py)")
    else:
        print("\nExport failed. Please check the error messages above.")


if __name__ == '__main__':
    main()
