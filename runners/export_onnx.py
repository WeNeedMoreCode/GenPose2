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
import argparse
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs.config import get_config
from networks.om_wrappers import create_score_network
from networks.posenet_agent import PoseNet

# ONNX 导出时绕过 autograd Function 包装，直接调用底层函数
# autograd.Function.apply 在 ONNX trace 模式下会导致精度错误（如 FPS 输出全零）
import pointnet2_ops
from networks.pts_encoder.pointnet2_utils.pointnet2 import pointnet2_utils

pointnet2_utils.furthest_point_sample = lambda xyz, npoint: pointnet2_ops._furthest_point_sampling(xyz, npoint)
pointnet2_utils.gather_operation = lambda features, idx: pointnet2_ops._gather_points(features, idx)
pointnet2_utils.grouping_operation = lambda points, idx: pointnet2_ops._group_points(points, idx)
pointnet2_utils.ball_query = lambda radius, nsample, xyz, new_xyz: pointnet2_ops._ball_query(new_xyz, xyz, radius, nsample)


def resolve_checkpoint(cfg_attr, fallback_path, explicit_path=None):
    """Resolve checkpoint path: use explicit path, config attr, or fallback.
    Returns None if checkpoint not found."""
    if explicit_path is None:
        explicit_path = getattr(cfg, cfg_attr, None) or fallback_path
    if not os.path.exists(explicit_path):
        print(f"Checkpoint not found: {explicit_path}")
        print(f"Please specify with --checkpoint_path")
        return None
    return explicit_path


def log_export(model_name, om_batch_size, checkpoint_path, device, **extra):
    print(f"\n{'='*60}")
    print(f"Exporting {model_name} to ONNX")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  OM Batch Size: {om_batch_size}")
    for k, v in extra.items():
        print(f"  {k}: {v}")
    if checkpoint_path:
        print(f"\nLoading {model_name}...")
        print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")


def get_pointnet2_input_info(cfg, batch_size=16):
    """
    Get input dimensions for PointNet2 encoder.

    The PointNet2 encoder (Pointnet2ClsMSGFus) expects:
        - pointcloud: [batch_size, 1024, 387] - Concatenated pts + rgb_feat
          where pts = [bs, 1024, 3] and rgb_feat = [bs, 1024, 384]

    Args:
        cfg: Configuration object
        batch_size: Batch size for ONNX export (default: 16)

    Returns:
        dict: Input information including shapes, dtypes, and names
    """
    return {
        'inputs': [
            {'name': 'pointcloud', 'shape': [batch_size, 1024, 387], 'dtype': 'float32'},
        ],
        'outputs': [
            {'name': 'pts_feat', 'shape': [batch_size, 1024], 'dtype': 'float32'},
        ],
    }


def export_pointnet2_to_onnx(checkpoint_path, output_dir, cfg, device='cpu', om_batch_size=16, output_name='pointnet2.onnx'):
    """
    Export PointNet2 encoder to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (ScoreNet checkpoint containing PointNet2)
        output_dir: Directory to save ONNX model
        cfg: Configuration object
        device: Device to load model on
        om_batch_size: Fixed batch size for OM model (default: 16).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_export("PointNet2 Encoder", om_batch_size, checkpoint_path, device)

    score_cfg = get_config()
    score_cfg.agent_type = 'score'
    score_cfg.device = device
    score_cfg.dino = 'pointwise'

    agent = PoseNet(score_cfg)
    agent.load_ckpt(model_dir=checkpoint_path, model_path=True, load_model_only=True)
    agent.net.eval()

    # Extract PointNet2 encoder
    pts_encoder = agent.net.pts_encoder
    pts_encoder.eval()

    # Export the pts_encoder directly (no wrapper)
    # It expects concatenated input: pointcloud [bs, 1024, 387]
    # Caller should concatenate pts [bs, 1024, 3] and rgb_feat [bs, 1024, 384] before calling
    export_model = pts_encoder

    # Get input info with specified batch_size
    input_info = get_pointnet2_input_info(cfg, batch_size=om_batch_size)
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
    onnx_path = output_dir / output_name
    print(f"\nExporting to {onnx_path}...")

    input_names = [inp['name'] for inp in input_info['inputs']]
    output_names = [out['name'] for out in input_info['outputs']]

    torch.onnx.export(
        export_model,
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    return onnx_path


def get_score_network_input_info(cfg, batch_size=800):
    """
    Get input dimensions for ScoreNetworkWrapper (pointwise mode).

    Inputs:
        - pts_feat: [batch_size, 1024] - Point cloud features (already contains RGB info)
        - sampled_pose: [batch_size, 9] - Current pose estimate (rot_matrix format)
        - t: [batch_size, 1] - Diffusion timestep

    Args:
        cfg: Configuration object
        batch_size: Batch size for ONNX export (default: 800)

    Returns:
        dict: Input information including shapes, dtypes, and names
    """
    return {
        'inputs': [
            {'name': 'pts_feat', 'shape': [batch_size, 1024], 'dtype': 'float32', 'format': 'feature'},
            {'name': 'sampled_pose', 'shape': [batch_size, 9], 'dtype': 'float32', 'format': 'pose_rot_matrix'},
            {'name': 't', 'shape': [batch_size, 1], 'dtype': 'float32', 'format': 'timestep'},
        ],
        'outputs': [
            {'name': 'score', 'shape': [batch_size, 9], 'dtype': 'float32'},
        ],
    }


def get_energy_network_input_info(batch_size=800):
    return {
        'inputs': [
            {'name': 'pts_feat', 'shape': [batch_size, 1024], 'dtype': 'float32'},
            {'name': 'sampled_pose', 'shape': [batch_size, 9], 'dtype': 'float32'},
            {'name': 't', 'shape': [batch_size, 1], 'dtype': 'float32'},
        ],
        'outputs': [
            {'name': 'energy', 'shape': [batch_size, 2], 'dtype': 'float32'},
        ],
    }


def get_scale_network_input_info(batch_size=1):
    return {
        'inputs': [
            {'name': 'pts_feat', 'shape': [batch_size, 1024], 'dtype': 'float32'},
            {'name': 'axes', 'shape': [batch_size, 3, 3], 'dtype': 'float32'},
        ],
        'outputs': [
            {'name': 'length', 'shape': [batch_size, 3], 'dtype': 'float32'},
        ],
    }


def get_dinov2_input_info(batch_size=16, img_size=224, num_pts=1024):
    return {
        'inputs': [
            {'name': 'roi_rgb', 'shape': [batch_size, 3, img_size, img_size], 'dtype': 'float32'},
            {'name': 'roi_xs', 'shape': [batch_size, num_pts], 'dtype': 'int64'},
            {'name': 'roi_ys', 'shape': [batch_size, num_pts], 'dtype': 'int64'},
        ],
        'outputs': [
            {'name': 'rgb_feat', 'shape': [batch_size, num_pts, 384], 'dtype': 'float32'},
        ],
    }


def export_score_network_to_onnx(checkpoint_path, output_dir, cfg, device='cpu', om_batch_size=800):
    """
    Export ScoreNetworkWrapper to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model
        cfg: Configuration object
        device: Device to load model on
        om_batch_size: Fixed batch size for OM model (default: 800).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_export("Score Network", om_batch_size, checkpoint_path, device)

    score_net = create_score_network(
        checkpoint_path=checkpoint_path,
        device=device
    )
    score_net.eval()

    # Get input info with specified batch_size
    input_info = get_score_network_input_info(cfg, batch_size=om_batch_size)
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

    # Pointwise mode: ScoreNet.forward() expects 4 args (pts_feat, rgb_feat, sampled_pose, t)
    # but rgb_feat is always None in pointwise mode, so we wrap to expose 3 inputs
    class ScoreNetExportWrapperPointwise(nn.Module):
        def __init__(self, score_net):
            super().__init__()
            self.score_net = score_net

        def forward(self, pts_feat, sampled_pose, t):
            return self.score_net(pts_feat, rgb_feat=None, sampled_pose=sampled_pose, t=t)

    export_model = ScoreNetExportWrapperPointwise(score_net)
    export_model.eval()

    # Export to ONNX
    onnx_path = output_dir / "scorenet.onnx"
    print(f"\nExporting to {onnx_path}...")

    input_names = [inp['name'] for inp in input_info['inputs']]
    output_names = [out['name'] for out in input_info['outputs']]

    torch.onnx.export(
        export_model,
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    return onnx_path


def export_energy_network_to_onnx(checkpoint_path, output_dir, cfg, device='cpu', om_batch_size=800):
    """
    Export Energy Network to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model
        cfg: Configuration object
        device: Device to load model on
        om_batch_size: Fixed batch size for OM model (default: 800).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_export("Energy Network", om_batch_size, checkpoint_path, device)

    energy_cfg = get_config()
    energy_cfg.agent_type = 'energy'
    energy_cfg.device = device
    energy_cfg.dino = 'pointwise'

    agent = PoseNet(energy_cfg)
    agent.load_ckpt(model_dir=checkpoint_path, model_path=True, load_model_only=True)
    net = agent.net
    net.eval()

    input_info = get_energy_network_input_info(batch_size=om_batch_size)
    dummy_inputs = [torch.randn(inp['shape'], dtype=torch.float32) if inp['dtype'] != 'int64'
                   else torch.randint(0, 224, inp['shape'], dtype=torch.int64)
                   for inp in input_info['inputs']]
    input_names = [inp['name'] for inp in input_info['inputs']]
    output_names = [out['name'] for out in input_info['outputs']]

    onnx_path = output_dir / "energynet.onnx"

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
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    return onnx_path


def export_scale_network_to_onnx(checkpoint_path, output_dir, cfg, device='cpu', om_batch_size=1):
    """
    Export Scale Network to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model
        cfg: Configuration object
        device: Device to load model on
        om_batch_size: Fixed batch size for OM model (default: 1).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_export("Scale Network", om_batch_size, checkpoint_path, device)

    scale_cfg = get_config()
    scale_cfg.agent_type = 'scale'
    scale_cfg.device = device
    scale_cfg.dino = 'pointwise'

    agent = PoseNet(scale_cfg)
    agent.load_ckpt(model_dir=checkpoint_path, model_path=True, load_model_only=True)
    net = agent.net
    net.eval()

    input_info = get_scale_network_input_info(batch_size=om_batch_size)
    dummy_inputs = [torch.randn(inp['shape'], dtype=torch.float32) for inp in input_info['inputs']]
    input_names = [inp['name'] for inp in input_info['inputs']]
    output_names = [out['name'] for out in input_info['outputs']]

    onnx_path = output_dir / "scalenet.onnx"

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
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    return onnx_path



class DINOv2ExportWrapper(nn.Module):
    """
    ONNX-compatible wrapper for DINOv2 pointwise feature extraction.

    Replaces get_intermediate_layers() with forward_features() for ONNX tracing.
    Includes torch.gather for point-wise feature extraction.

    Input:
        roi_rgb:  [batch_size, 3, img_size, img_size]
        roi_xs:   [batch_size, num_pts] int64
        roi_ys:   [batch_size, num_pts] int64

    Output:
        rgb_feat: [batch_size, num_pts, 384]
    """

    def __init__(self, dinov2_model, dino_dim=384, img_size=224):
        super().__init__()
        self.dino = dinov2_model
        self.dino_dim = dino_dim
        self.patch_size = 14
        self.feat_size = img_size // self.patch_size  # 16 for 224

    def forward(self, roi_rgb, roi_xs, roi_ys):
        feat = self.dino.forward_features(roi_rgb)
        # forward_features already strips CLS token: x_norm_patchtokens [B, 256, 384]
        # Equivalent to get_intermediate_layers(x)[0]
        feat = feat['x_norm_patchtokens']  # [B, 256, 384]

        xs = roi_xs // self.patch_size
        ys = roi_ys // self.patch_size
        pos = xs * self.feat_size + ys  # [B, num_pts]
        pos = pos.unsqueeze(-1).expand(-1, -1, self.dino_dim)  # [B, num_pts, 384]

        rgb_feat = torch.gather(feat, 1, pos)  # [B, num_pts, 384]
        return rgb_feat


def export_dinov2_to_onnx(output_dir, device='cpu', img_size=224, num_pts=1024, om_batch_size=16):
    """
    Export DINOv2 (dinov2_vits14) to ONNX.

    Args:
        output_dir: Directory to save ONNX model
        device: Device to load model on (use 'cpu' for ONNX export)
        img_size: Input image size (default: 224)
        num_pts: Number of points (default: 1024)
        om_batch_size: Fixed batch size for OM model (default: 16).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_export("DINOv2 (dinov2_vits14)", om_batch_size, None, device,
               **{'Image size': img_size, 'Num points': num_pts,
                   'Feature dim': 384, 'Feature map': f'{img_size//14}x{img_size//14}'})

    # Load DINOv2
    import torch.hub
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino = dino.to(device)
    dino.requires_grad_(False)

    # Patch interpolate_pos_encoding: bicubic -> bilinear (Ascend 310P only supports bilinear)
    import torch.nn.functional as F, math, types
    def _no_resize_interpolate_pos(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        dtype = x.dtype
        pos_embed = self.pos_embed.float()
        cls_emb = pos_embed[:, :1]
        patch_emb = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = h0 = int(math.sqrt(N))
        tw = w // self.patch_size
        th = h // self.patch_size
        patch_emb = patch_emb.reshape(1, w0, h0, dim).permute(0, 3, 1, 2)
        patch_emb = F.interpolate(patch_emb, size=(th, tw), mode='bilinear', align_corners=False)
        patch_emb = patch_emb.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((cls_emb, patch_emb), dim=1).to(dtype)

    dino.interpolate_pos_encoding = types.MethodType(_no_resize_interpolate_pos, dino)
    print(f"  Patched interpolate_pos_encoding: bicubic -> bilinear")

    # Create wrapper
    export_model = DINOv2ExportWrapper(dino, dino_dim=384, img_size=img_size)
    export_model.eval()

    # Build dummy inputs from input_info
    input_info = get_dinov2_input_info(batch_size=om_batch_size, img_size=img_size, num_pts=num_pts)
    dummy_inputs = []
    for inp in input_info['inputs']:
        if inp['dtype'] == 'int64':
            dummy_inputs.append(torch.randint(0, img_size, inp['shape'], dtype=torch.int64).to(device))
        else:
            dummy_inputs.append(torch.randn(inp['shape'], dtype=torch.float32).to(device))
    input_names = [inp['name'] for inp in input_info['inputs']]
    output_names = [out['name'] for out in input_info['outputs']]

    # Export
    onnx_path = output_dir / "dinov2_vits14.onnx"

    torch.onnx.export(
        export_model,
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    # Verify and fix Resize nodes
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verification passed")

    for node in onnx_model.graph.node:
        if node.op_type == 'Resize':
            for attr in node.attribute:
                if attr.name == 'coordinate_transformation_mode' and attr.s == b'half_pixel':
                    attr.s = b'asymmetric'
    onnx.save(onnx_model, str(onnx_path))
    print(f"Resize fix applied (half_pixel -> asymmetric)")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export GenPose2 Networks to ONNX')
    parser.add_argument('--agent_type', type=str, default='score',
                        choices=['score', 'energy', 'scale', 'pointnet2_from_score', 'pointnet2_from_energy', 'dinov2'],
                        help='Agent type to export')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                        help='Output directory for ONNX models')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint (auto-detected if not specified)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for export (default: cpu)')
    parser.add_argument('--om_batch_size', type=int,
                        help='Fixed batch size for OM model. '
                             'Should typically be batch_size * eval_repeat_num (e.g., 16*50=800). '
                             'For best performance, match this to your inference configuration.')

    args = parser.parse_args()

    # Setup config
    sys.argv = [
        'export_onnx.py',
        '--device', args.device,
        '--dino', 'pointwise',  # Enable DINOv2 (must match checkpoint training mode)
    ]

    cfg = get_config()

    # Determine checkpoint path and export function based on agent_type
    output_file = None
    if args.agent_type == 'score':
        checkpoint_path = resolve_checkpoint('pretrained_score_model_path',
                                             './results/ckpts/ScoreNet/scorenet.pth', args.checkpoint_path)
        if checkpoint_path is None:
            return
        output_file = export_score_network_to_onnx(checkpoint_path, args.output_dir, cfg, args.device, args.om_batch_size)

    elif args.agent_type == 'pointnet2_from_score':
        checkpoint_path = resolve_checkpoint('pretrained_score_model_path',
                                             './results/ckpts/ScoreNet/scorenet.pth', args.checkpoint_path)
        if checkpoint_path is None:
            return
        args.om_batch_size = getattr(args, "om_batch_size", 16)
        output_file = export_pointnet2_to_onnx(checkpoint_path, args.output_dir, cfg, args.device, args.om_batch_size,
                                               output_name='pointnet2_from_score.onnx')

    elif args.agent_type == 'pointnet2_from_energy':
        checkpoint_path = resolve_checkpoint('pretrained_energy_model_path',
                                             './results/ckpts/EnergyNet/energynet.pth', args.checkpoint_path)
        if checkpoint_path is None:
            return
        args.om_batch_size = getattr(args, "om_batch_size", 16)
        output_file = export_pointnet2_to_onnx(checkpoint_path, args.output_dir, cfg, args.device, args.om_batch_size,
                                               output_name='pointnet2_from_energy.onnx')

    elif args.agent_type == 'energy':
        checkpoint_path = resolve_checkpoint('pretrained_energy_model_path',
                                             './results/ckpts/EnergyNet/energynet.pth', args.checkpoint_path)
        if checkpoint_path is None:
            return
        output_file = export_energy_network_to_onnx(checkpoint_path, args.output_dir, cfg, args.device, args.om_batch_size)

    elif args.agent_type == 'scale':
        checkpoint_path = resolve_checkpoint('pretrained_scale_model_path',
                                             './results/ckpts/ScaleNet/scalenet.pth', args.checkpoint_path)
        if checkpoint_path is None:
            return
        output_file = export_scale_network_to_onnx(checkpoint_path, args.output_dir, cfg, args.device, args.om_batch_size)

    elif args.agent_type == 'dinov2':
        img_size = getattr(cfg, 'img_size', 224)
        output_file = export_dinov2_to_onnx(args.output_dir, args.device, img_size=img_size)

    if output_file:
        print(f"\nExport completed: {output_file}")
        print(f"Next: python runners/onnx2om.py --onnx_path {output_file}")
    else:
        print("\nExport failed.")


if __name__ == '__main__':
    main()
