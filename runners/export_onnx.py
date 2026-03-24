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
from networks.posenet_agent import PoseNet
from networks.scalenet import ScaleNet


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




def export_energy_network_to_onnx(checkpoint_path, output_dir, cfg, device='cpu'):
    """
    Export Energy Network to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model and metadata
        cfg: Configuration object
        device: Device to load model on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting Energy Network to ONNX")
    print(f"{'='*60}")

    # Load PoseNet with energy checkpoint
    print(f"\nLoading Energy Network...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")

    energy_cfg = get_config()
    energy_cfg.agent_type = 'energy'
    energy_cfg.device = device
    energy_cfg.dino = 'pointwise'

    agent = PoseNet(energy_cfg)
    agent.load_ckpt(model_dir=checkpoint_path, model_path=True, load_model_only=True)
    net = agent.net
    net.eval()

    # Input info
    print(f"\nInput info:")
    print(f"  pts_feat: [1, 1024], float32")
    print(f"  sampled_pose: [1, 9], float32")
    print(f"  t: [1, 1], float32")

    print(f"\nOutput info:")
    print(f"  energy: [1], float32")

    # Prepare dummy inputs
    pts_feat = torch.randn(1, 1024, dtype=torch.float32)
    rgb_feat = None  # pointwise mode: already fused in pts_feat
    sampled_pose = torch.randn(1, 9, dtype=torch.float32)
    t = torch.randn(1, 1, dtype=torch.float32)

    # Export to ONNX
    onnx_path = output_dir / "energy_network.onnx"
    print(f"\nExporting to {onnx_path}...")

    # Minimal wrapper to set return_item='energy'
    class EnergyExportWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, pts_feat, sampled_pose, t):
            data = {
                'pts_feat': pts_feat,
                'rgb_feat': None,  # pointwise mode: already fused in pts_feat
                'sampled_pose': sampled_pose,
                't': t
            }
            return self.net(data, return_item='energy')

    energy_net = EnergyExportWrapper(net.pose_score_net)

    torch.onnx.export(
        energy_net,
        (pts_feat, sampled_pose, t),
        str(onnx_path),
        input_names=['pts_feat', 'sampled_pose', 't'],
        output_names=['energy'],
        dynamic_axes={
            'pts_feat': {0: 'batch_size'},
            'sampled_pose': {0: 'batch_size'},
            't': {0: 'batch_size'},
            'energy': {0: 'batch_size'},
        },
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    # Save metadata
    metadata_path = output_dir / "energy_network_metadata.json"
    metadata = {
        'model_type': 'PoseEnergyNet',
        'checkpoint_path': str(checkpoint_path),
        'onnx_path': str(onnx_path),
        'inputs': [
            {'name': 'pts_feat', 'shape': [1, 1024], 'dtype': 'float32'},
            {'name': 'sampled_pose', 'shape': [1, 9], 'dtype': 'float32'},
            {'name': 't', 'shape': [1, 1], 'dtype': 'float32'},
        ],
        'outputs': [
            {'name': 'energy', 'shape': [1], 'dtype': 'float32'},
        ],
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


def export_scale_network_to_onnx(checkpoint_path, output_dir, cfg, device='cpu'):
    """
    Export Scale Network to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model and metadata
        cfg: Configuration object
        device: Device to load model on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting Scale Network to ONNX")
    print(f"{'='*60}")

    # Load PoseNet with scale checkpoint
    print(f"\nLoading Scale Network...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")

    scale_cfg = get_config()
    scale_cfg.agent_type = 'scale'
    scale_cfg.device = device
    scale_cfg.dino = 'pointwise'

    agent = PoseNet(scale_cfg)
    agent.load_ckpt(model_dir=checkpoint_path, model_path=True, load_model_only=True)
    net = agent.net
    net.eval()

    # Input info
    print(f"\nInput info:")
    print(f"  pts_feat: [1, 1024], float32")
    print(f"  axes: [1, 3, 3], float32")

    print(f"\nOutput info:")
    print(f"  length: [1, 3], float32")

    # Prepare dummy inputs
    pts_feat = torch.randn(1, 1024, dtype=torch.float32)
    axes = torch.randn(1, 3, 3, dtype=torch.float32)

    # Export to ONNX
    onnx_path = output_dir / "scale_network.onnx"
    print(f"\nExporting to {onnx_path}...")

    # Wrapper: convert positional args to dict for ScaleNet
    class ScaleExportWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, pts_feat, axes):
            data = {'pts_feat': pts_feat, 'axes': axes}
            return self.net(data)

    scale_net = ScaleExportWrapper(net)

    torch.onnx.export(
        scale_net,
        (pts_feat, axes),
        str(onnx_path),
        input_names=['pts_feat', 'axes'],
        output_names=['length'],
        dynamic_axes={
            'pts_feat': {0: 'batch_size'},
            'axes': {0: 'batch_size'},
            'length': {0: 'batch_size'},
        },
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    # Save metadata
    metadata_path = output_dir / "scale_network_metadata.json"
    metadata = {
        'model_type': 'ScaleNet',
        'checkpoint_path': str(checkpoint_path),
        'onnx_path': str(onnx_path),
        'inputs': [
            {'name': 'pts_feat', 'shape': [1, 1024], 'dtype': 'float32'},
            {'name': 'axes', 'shape': [1, 3, 3], 'dtype': 'float32'},
        ],
        'outputs': [
            {'name': 'length', 'shape': [1, 3], 'dtype': 'float32'},
        ],
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


def export_pointnet2_scorenet_to_onnx(checkpoint_path, output_dir, cfg, device='cpu'):
    """
    Export PointNet2 + ScoreNet to ONNX format.

    This exports the complete pipeline for NPU acceleration:
        pts[bs,1024,3] + rgb_feat[bs,1024,384] → PointNet2 → ScoreNet → score[bs,9]

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model and metadata
        cfg: Configuration object
        device: Device to load model on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting PointNet2 + ScoreNet to ONNX")
    print(f"{'='*60}")

    # Load PoseNet with score checkpoint
    print(f"\nLoading PoseNet (PointNet2 + ScoreNet)...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")

    score_cfg = get_config()
    score_cfg.agent_type = 'score'
    score_cfg.device = device
    score_cfg.dino = 'pointwise'

    agent = PoseNet(score_cfg)
    agent.load_ckpt(model_dir=checkpoint_path, model_path=True, load_model_only=True)
    net = agent.net
    net.eval()

    # Input info
    print(f"\nInput info:")
    print(f"  pts: [1, 1024, 3], float32 - Raw point cloud")
    print(f"  rgb_feat: [1, 1024, 384], float32 - DINOv2 features")
    print(f"  sampled_pose: [1, 9], float32")
    print(f"  t: [1, 1], float32")

    print(f"\nOutput info:")
    print(f"  score: [1, 9], float32")

    # Prepare dummy inputs
    pts = torch.randn(1, 1024, 3, dtype=torch.float32)
    rgb_feat = torch.randn(1, 1024, 384, dtype=torch.float32)
    sampled_pose = torch.randn(1, 9, dtype=torch.float32)
    t = torch.randn(1, 1, dtype=torch.float32)

    # Export to ONNX
    onnx_path = output_dir / "pointnet2_scorenet.onnx"
    print(f"\nExporting to {onnx_path}...")

    # Wrapper: PointNet2 + ScoreNet
    class PointNet2ScoreNetExportWrapper(nn.Module):
        def __init__(self, pts_encoder, pose_score_net):
            super().__init__()
            self.pts_encoder = pts_encoder
            self.pose_score_net = pose_score_net
        def forward(self, pts, rgb_feat, sampled_pose, t):
            # Concatenate pts + rgb_feat (pointwise mode)
            pts_with_rgb = torch.cat([pts, rgb_feat], dim=-1)  # [bs, 1024, 387]
            # PointNet2 encoding
            pts_feat = self.pts_encoder(pts_with_rgb)  # [bs, 1024]
            # ScoreNet forward
            data = {
                'pts_feat': pts_feat,
                'rgb_feat': None,  # pointwise mode: already fused in pts_feat
                'sampled_pose': sampled_pose,
                't': t
            }
            return self.pose_score_net(data)

    pn2s_net = PointNet2ScoreNetExportWrapper(net.pts_encoder, net.pose_score_net)

    torch.onnx.export(
        pn2s_net,
        (pts, rgb_feat, sampled_pose, t),
        str(onnx_path),
        input_names=['pts', 'rgb_feat', 'sampled_pose', 't'],
        output_names=['score'],
        dynamic_axes={
            'pts': {0: 'batch_size'},
            'rgb_feat': {0: 'batch_size'},
            'sampled_pose': {0: 'batch_size'},
            't': {0: 'batch_size'},
            'score': {0: 'batch_size'},
        },
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    # Save metadata
    metadata_path = output_dir / "pointnet2_scorenet_metadata.json"
    metadata = {
        'model_type': 'PointNet2+ScoreNet',
        'checkpoint_path': str(checkpoint_path),
        'onnx_path': str(onnx_path),
        'inputs': [
            {'name': 'pts', 'shape': [1, 1024, 3], 'dtype': 'float32'},
            {'name': 'rgb_feat', 'shape': [1, 1024, 384], 'dtype': 'float32'},
            {'name': 'sampled_pose', 'shape': [1, 9], 'dtype': 'float32'},
            {'name': 't', 'shape': [1, 1], 'dtype': 'float32'},
        ],
        'outputs': [
            {'name': 'score', 'shape': [1, 9], 'dtype': 'float32'},
        ],
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
    parser = argparse.ArgumentParser(description='Export GenPose2 Networks to ONNX')
    parser.add_argument('--agent_type', type=str, default='score',
                        choices=['score', 'energy', 'scale', 'pointnet2_scorenet'],
                        help='Agent type to export: score, energy, scale, or pointnet2_scorenet')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                        help='Output directory for ONNX models')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint (auto-detected if not specified)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for export (default: cpu)')

    args = parser.parse_args()

    # Setup config
    sys.argv = [
        'export_onnx.py',
        '--device', args.device,
        '--dino', 'pointwise',  # Enable DINOv2 (must match checkpoint training mode)
    ]

    cfg = get_config()

    # Determine checkpoint path and export function based on agent_type
    if args.agent_type == 'score':
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

    elif args.agent_type == 'energy':
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = getattr(cfg, 'pretrained_energy_model_path',
                                        None) or './results/ckpts/EnergyNet/energynet.pth'
            print(f"Auto-detected checkpoint: {checkpoint_path}")

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
            print(f"Please specify the correct path with --checkpoint_path")
            return

        # Export
        success = export_energy_network_to_onnx(checkpoint_path, args.output_dir, cfg, args.device)

        if success:
            print(f"\n{'='*60}")
            print("Export completed successfully!")
            print(f"{'='*60}")
            print(f"\nExported files:")
            print(f"  - {args.output_dir}/energy_network.onnx")
            print(f"  - {args.output_dir}/energy_network_metadata.json")

    elif args.agent_type == 'scale':
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = getattr(cfg, 'pretrained_scale_model_path',
                                        None) or './results/ckpts/ScaleNet/scalenet.pth'
            print(f"Auto-detected checkpoint: {checkpoint_path}")

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
            print(f"Please specify the correct path with --checkpoint_path")
            return

        # Export
        success = export_scale_network_to_onnx(checkpoint_path, args.output_dir, cfg, args.device)

        if success:
            print(f"\n{'='*60}")
            print("Export completed successfully!")
            print(f"{'='*60}")
            print(f"\nExported files:")
            print(f"  - {args.output_dir}/scale_network.onnx")
            print(f"  - {args.output_dir}/scale_network_metadata.json")

    elif args.agent_type == 'pointnet2_scorenet':
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
        success = export_pointnet2_scorenet_to_onnx(checkpoint_path, args.output_dir, cfg, args.device)

        if success:
            print(f"\n{'='*60}")
            print("Export completed successfully!")
            print(f"{'='*60}")
            print(f"\nExported files:")
            print(f"  - {args.output_dir}/pointnet2_scorenet.onnx")
            print(f"  - {args.output_dir}/pointnet2_scorenet_metadata.json")

    if success:
        print(f"\nNext steps:")
        print(f"1. Convert ONNX to OM using ATC tool:")
        print(f"   python runners/onnx2om.py --onnx_path {args.output_dir}/{args.agent_type}_network.onnx")
        print(f"\n2. Use the OM model in NPU inference (TODO: implement infer_om.py)")
    else:
        print("\nExport failed. Please check the error messages above.")


if __name__ == '__main__':
    main()
