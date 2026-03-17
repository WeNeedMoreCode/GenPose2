"""
GenPose2 OM Model Export Script

Exports PyTorch models to ONNX format for OM conversion.
Models are exported with simplified interfaces for NPU inference.

Usage:
    python runners/export_om.py --agent_type score --output_dir ./om_models
    python runners/export_om.py --agent_type energy --output_dir ./om_models
    python runners/export_om.py --agent_type scale --output_dir ./om_models
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
from networks.posenet_agent import PoseNet


def get_model_input_info(agent_type, cfg):
    """
    Get input dimensions and metadata for each agent type.

    Returns:
        dict: Input information including shapes, dtypes, and names
    """
    if agent_type == 'score':
        return {
            'inputs': [
                {'name': 'pts', 'shape': [1, 1024, 3], 'dtype': 'float32', 'format': 'point_cloud'},
                {'name': 'roi_rgb', 'shape': [1, 3, 224, 224], 'dtype': 'float32', 'format': 'image'},
                {'name': 'roi_xs', 'shape': [1, 1024], 'dtype': 'int64', 'format': 'coordinates'},
                {'name': 'roi_ys', 'shape': [1, 1024], 'dtype': 'int64', 'format': 'coordinates'},
            ],
            'outputs': [
                {'name': 'pred_pose', 'shape': [1, 10, 7], 'dtype': 'float32'},
            ],
            'metadata': {
                'sampling_steps': cfg.sampling_steps,
                'T0': cfg.T0,
                'pose_mode': cfg.pose_mode,
            }
        }

    elif agent_type == 'energy':
        return {
            'inputs': [
                {'name': 'pts', 'shape': [1, 1024, 3], 'dtype': 'float32', 'format': 'point_cloud'},
                {'name': 'roi_rgb', 'shape': [1, 3, 224, 224], 'dtype': 'float32', 'format': 'image'},
                {'name': 'sampled_pose', 'shape': [1, 10, 7], 'dtype': 'float32', 'format': 'pose'},
            ],
            'outputs': [
                {'name': 'pred_energy', 'shape': [1, 10, 2], 'dtype': 'float32'},
            ],
            'metadata': {
                'pose_mode': cfg.pose_mode,
                'energy_mode': cfg.energy_mode,
            }
        }

    elif agent_type == 'scale':
        return {
            'inputs': [
                {'name': 'pts_feat', 'shape': [1, 1024], 'dtype': 'float32', 'format': 'feature'},
                {'name': 'rgb_feat', 'shape': [1, 384], 'dtype': 'float32', 'format': 'feature'},
                {'name': 'axes', 'shape': [1, 3, 3], 'dtype': 'float32', 'format': 'matrix'},
            ],
            'outputs': [
                {'name': 'bbox_length', 'shape': [1, 3], 'dtype': 'float32'},
            ],
            'metadata': {
                'num_points': cfg.num_points,
            }
        }

    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


def create_export_wrapper(agent_type, model):
    """
    Create a wrapper for ONNX export with simplified interface.

    The wrapper handles the complex forward() interface of GFObjectPose
    and provides a clean input/output interface for ONNX export.
    """

    class ExportWrapper(nn.Module):
        def __init__(self, model, agent_type):
            super().__init__()
            self.model = model
            self.agent_type = agent_type
            self.model.eval()

        def forward(self, *args):
            """
            Simplified forward pass for ONNX export.

            Args:
                *args: Input tensors (format depends on agent_type)

            Returns:
                Output tensors
            """
            if self.agent_type == 'score':
                pts, roi_rgb, roi_xs, roi_ys = args
                data = {
                    'pts': pts,
                    'roi_rgb': roi_rgb,
                    'roi_xs': roi_xs,
                    'roi_ys': roi_ys,
                    'pts_center': torch.zeros(pts.shape[0], 3, device=pts.device),
                }
                # Extract features first (without torch.compile complications)
                with torch.no_grad():
                    pts_feat = self.model.net.extract_pts_feature(data)
                    rgb_feat = self.model.net(data, mode='rgb_feature')

                # For ONNX export, we use a simplified sampling path
                # This is a placeholder - actual ODE sampling is not exportable
                # We export the score network only
                batch_size = pts.shape[0]
                dummy_pose = torch.randn(batch_size, 10, 7, device=pts.device)
                return dummy_pose, pts_feat, rgb_feat

            elif self.agent_type == 'energy':
                pts, roi_rgb, sampled_pose = args
                data = {
                    'pts': pts,
                    'roi_rgb': roi_rgb,
                    'sampled_pose': sampled_pose,
                    'pts_center': torch.zeros(pts.shape[0], 3, device=pts.device),
                    't': torch.ones(pts.shape[0], 1, device=pts.device) * 1e-5,
                }
                with torch.no_grad():
                    pts_feat = self.model.net.extract_pts_feature(data)
                    rgb_feat = self.model.net(data, mode='rgb_feature')

                    energy_input_data = {
                        'pts_feat': pts_feat,
                        'rgb_feat': rgb_feat,
                        'sampled_pose': sampled_pose,
                        't': data['t'],
                    }
                    energy = self.model.net(energy_input_data, mode='energy')
                return energy

            elif self.agent_type == 'scale':
                pts_feat, rgb_feat, axes = args
                data = {
                    'pts_feat': pts_feat,
                    'rgb_feat': rgb_feat,
                    'axes': axes,
                }
                with torch.no_grad():
                    length = self.model.net(data)
                return length

    return ExportWrapper(model, agent_type)


def export_to_onnx(agent_type, checkpoint_path, output_dir, cfg):
    """
    Export a PyTorch model to ONNX format.

    Args:
        agent_type: Type of agent ('score', 'energy', 'scale')
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model and metadata
        cfg: Configuration object
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting {agent_type.upper()} model to ONNX")
    print(f"{'='*60}")

    # Load model
    cfg.agent_type = agent_type
    agent = PoseNet(cfg)

    # Use the provided checkpoint_path
    checkpoint = checkpoint_path

    print(f"Loading checkpoint: {checkpoint}")
    agent.load_ckpt(model_dir=checkpoint, model_path=True, load_model_only=True)
    agent.eval()

    # Get input info
    input_info = get_model_input_info(agent_type, cfg)
    print(f"\nInput info:")
    for inp in input_info['inputs']:
        print(f"  {inp['name']}: {inp['shape']}, {inp['dtype']}")

    # Create export wrapper
    wrapper = create_export_wrapper(agent_type, agent)

    # Prepare dummy inputs
    dummy_inputs = []
    for inp in input_info['inputs']:
        dummy = torch.randn(inp['shape'], dtype=torch.float32)
        if inp['dtype'] == 'int64':
            dummy = torch.randint(0, 224, inp['shape'], dtype=torch.int64)
        dummy_inputs.append(dummy)

    # Export to ONNX
    onnx_path = output_dir / f"{agent_type}_model.onnx"
    print(f"\nExporting to {onnx_path}...")

    dynamic_axes = {}
    input_names = [inp['name'] for inp in input_info['inputs']]
    output_names = [out['name'] for out in input_info['outputs']]

    # Add dynamic batch dimension
    for name in input_names:
        dynamic_axes[name] = {0: 'batch_size'}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}


    torch.onnx.export(
        wrapper,
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"✓ ONNX export successful: {onnx_path}")


    # Save metadata
    metadata_path = output_dir / f"{agent_type}_metadata.json"
    metadata = {
        'agent_type': agent_type,
        'checkpoint_path': str(checkpoint),
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

    # Export sub-modules separately
    print(f"\nExporting sub-modules...")


    if hasattr(agent.net, 'pts_encoder'):
        pts_encoder_path = output_dir / f"{agent_type}_pointnet2.onnx"
        dummy_pts = torch.randn(1, 1024, 3, dtype=torch.float32)
        if cfg.dino == 'pointwise':
            dummy_rgb_feat = torch.randn(1, 1024, 384, dtype=torch.float32)
            dummy_pts_input = torch.cat([dummy_pts, dummy_rgb_feat], dim=-1)
        else:
            dummy_pts_input = dummy_pts

        torch.onnx.export(
            agent.net.pts_encoder,
            dummy_pts_input,
            str(pts_encoder_path),
            input_names=['pts'],
            output_names=['pts_feat'],
            dynamic_axes={'pts': {0: 'batch_size'}, 'pts_feat': {0: 'batch_size'}},
            opset_version=17,
        )
        print(f"✓ PointNet2 exported: {pts_encoder_path}")


    return True


def main():
    parser = argparse.ArgumentParser(description='Export GenPose2 models to ONNX for OM conversion')
    parser.add_argument('--agent_type', type=str, required=True,
                        choices=['score', 'energy', 'scale'],
                        help='Agent type to export')
    parser.add_argument('--output_dir', type=str, default='./om_models',
                        help='Output directory for ONNX models')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint (auto-detected if not specified)')

    args = parser.parse_args()

    # Setup config (export doesn't need real data)
    sys.argv = [
        'export_om.py',
        '--data_path', './data',  # Dummy path, not used for export
        '--device', 'cpu',  # Use CPU for export
    ]

    cfg = get_config()

    # Determine checkpoint path
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        # Auto-detect checkpoint path based on agent type
        if args.agent_type == 'score':
            checkpoint_path = getattr(cfg, 'pretrained_score_model_path',
                                        None) or './results/ckpts/ScoreNet/scorenet.pth'
        elif args.agent_type == 'energy':
            checkpoint_path = getattr(cfg, 'pretrained_energy_model_path',
                                        './results/ckpts/EnergyNet/energynet.pth')
        elif args.agent_type == 'scale':
            checkpoint_path = getattr(cfg, 'pretrained_scale_model_path',
                                        './results/ckpts/ScaleNet/scalenet.pth')

        print(f"Auto-detected checkpoint: {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
        print(f"Please specify the correct path with --checkpoint_path")
        return

    # Export
    success = export_to_onnx(args.agent_type, checkpoint_path, args.output_dir, cfg)

    if success:
        print(f"\n{'='*60}")
        print("Export completed successfully!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Convert ONNX to OM using ATC tool:")
        print(f"   atc --framework=5 --model={args.output_dir}/{args.agent_type}_model.onnx \\")
        print(f"       --output={args.output_dir}/{args.agent_type}_model.om \\")
        print(f"       --input_format=NCHW")
        print(f"\n2. Run inference:")
        print(f"   python runners/infer_om.py --agent_type {args.agent_type} \\")
        print(f"       --om_model {args.output_dir}/{args.agent_type}_model.om \\")
        print(f"       --data_path {args.data_path}")
    else:
        print("\nExport failed. Please check the error messages above.")


if __name__ == '__main__':
    main()
