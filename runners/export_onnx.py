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
from networks.score_wrapper import create_score_network
from networks.posenet_agent import PoseNet
from networks.scalenet import ScaleNet
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSGFus

# ONNX 导出时绕过 autograd Function 包装，直接调用底层函数
# autograd.Function.apply 在 ONNX trace 模式下会导致精度错误（如 FPS 输出全零）
import pointnet2_ops
from networks.pts_encoder.pointnet2_utils.pointnet2 import pointnet2_utils

pointnet2_utils.furthest_point_sample = lambda xyz, npoint: pointnet2_ops._furthest_point_sampling(xyz, npoint)
pointnet2_utils.gather_operation = lambda features, idx: pointnet2_ops._gather_points(features, idx)
pointnet2_utils.grouping_operation = lambda points, idx: pointnet2_ops._group_points(points, idx)
pointnet2_utils.ball_query = lambda radius, nsample, xyz, new_xyz: pointnet2_ops._ball_query(new_xyz, xyz, radius, nsample)


def get_pointnet2_input_info(cfg, batch_size=1):
    """
    Get input dimensions for PointNet2 encoder.

    The PointNet2 encoder (Pointnet2ClsMSGFus) expects:
        - pointcloud: [batch_size, 1024, 387] - Concatenated pts + rgb_feat
          where pts = [bs, 1024, 3] and rgb_feat = [bs, 1024, 384]

    Args:
        cfg: Configuration object
        batch_size: Batch size for ONNX export (default: 1)

    Returns:
        dict: Input information including shapes, dtypes, and names
    """
    return {
        'inputs': [
            {'name': 'pointcloud', 'shape': [1, 1024, 387], 'dtype': 'float32', 'format': 'concatenated_input'},
        ],
        'outputs': [
            {'name': 'pts_feat', 'shape': [1, 1024], 'dtype': 'float32'},
        ],
        'metadata': {
            'export_type': 'pointnet2_encoder',
            'architecture': 'Pointnet2ClsMSGFus',
            'dino_mode': 'pointwise',
        }
    }


def export_pointnet2_to_onnx(checkpoint_path, output_dir, cfg, device='cpu', om_batch_size=1, output_name='pointnet2.onnx'):
    """
    Export PointNet2 encoder to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (ScoreNet checkpoint containing PointNet2)
        output_dir: Directory to save ONNX model and metadata
        cfg: Configuration object
        device: Device to load model on
        om_batch_size: Fixed batch size for OM model (default: 1).
                      Typically use batch_size from DataLoader (e.g., 16).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting PointNet2 Encoder to ONNX")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  OM Batch Size: {om_batch_size}")
    if om_batch_size > 1:
        print(f"  Note: Using fixed batch_size={om_batch_size} for OM deployment")
        print(f"  (Typically = DataLoader batch_size, e.g., 16)")

    # Load PoseNet to extract PointNet2 encoder
    print(f"\nLoading PointNet2 encoder from checkpoint...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")

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

    # Use dynamic_axes for dynamic batch size support
    # This allows OM conversion with --dynamic_batch_size
    dynamic_axes = {
        'pointcloud': {0: 'batch_size'},
        'pts_feat': {0: 'batch_size'},
    }

    torch.onnx.export(
        export_model,
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,  # Dynamic batch_size
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=False,
        keep_initializers_as_inputs=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    return True


def get_score_network_input_info(cfg, batch_size=1):
    """
    Get input dimensions for ScoreNetworkWrapper.

    In pointwise mode (dino='pointwise'):
        - pts_feat: [batch_size, 1024] - Point cloud features (already contains RGB info)
        - sampled_pose: [batch_size, 9] - Current pose estimate (rot_matrix format)
        - t: [batch_size, 1] - Diffusion timestep
        Note: rgb_feat is NOT used (dino_dim=0), so it's excluded from ONNX export

    In global mode (dino='global'):
        - pts_feat: [batch_size, 1024] - Point cloud features
        - rgb_feat: [batch_size, 384] - RGB features (DINOv2)
        - sampled_pose: [batch_size, 9] - Current pose estimate (rot_matrix format)
        - t: [batch_size, 1] - Diffusion timestep

    Args:
        cfg: Configuration object
        batch_size: Batch size for ONNX export (default: 1)

    Returns:
        dict: Input information including shapes, dtypes, and names
    """
    # In pointwise mode, rgb_feat is fused in pts_feat and NOT used by ScoreNet
    # (dino_dim=0, so ScoreNet takes the else branch without rgb_feat)
    use_rgb_feat = (cfg.dino == 'global')

    inputs = [
        {'name': 'pts_feat', 'shape': [1, 1024], 'dtype': 'float32', 'format': 'feature'},
    ]

    if use_rgb_feat:
        inputs.append({'name': 'rgb_feat', 'shape': [1, 384], 'dtype': 'float32', 'format': 'feature'})

    inputs.extend([
        {'name': 'sampled_pose', 'shape': [1, 9], 'dtype': 'float32', 'format': 'pose_rot_matrix'},
        {'name': 't', 'shape': [1, 1], 'dtype': 'float32', 'format': 'timestep'},
    ])

    return {
        'inputs': inputs,
        'outputs': [
            {'name': 'score', 'shape': [1, 9], 'dtype': 'float32'},
        ],
        'metadata': {
            'export_type': 'score_network_wrapper',
            'pose_mode': cfg.pose_mode,
            'dino_mode': cfg.dino,
            'use_rgb_feat': use_rgb_feat,
        }
    }


def export_score_network_to_onnx(checkpoint_path, output_dir, cfg, device='cpu', om_batch_size=1):
    """
    Export ScoreNetworkWrapper to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model and metadata
        cfg: Configuration object
        device: Device to load model on
        om_batch_size: Fixed batch size for OM model (default: 1).
                      Should match inference configuration (e.g., batch_size * eval_repeat_num).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting Score Network (ScoreNetworkWrapper) to ONNX")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  OM Batch Size: {om_batch_size}")
    print(f"  DINO Mode: {cfg.dino}")
    if om_batch_size > 1:
        print(f"  Note: Using fixed batch_size={om_batch_size} for OM deployment")
        print(f"  (Typically = batch_size * eval_repeat_num, e.g., 16 * 50 = 800)")
    if cfg.dino == 'pointwise':
        print(f"  Pointwise mode: rgb_feat is excluded (already fused in pts_feat)")

    # Load ScoreNetworkWrapper
    print(f"\nLoading ScoreNetworkWrapper...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")

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

    # Create export wrapper to handle variable number of inputs
    # ScoreNetworkWrapper.forward() always expects 4 args, but pointwise mode exports 3
    use_rgb_feat = input_info['metadata']['use_rgb_feat']

    if use_rgb_feat:
        # Global mode: 4 inputs - use score_net directly
        export_model = score_net
    else:
        # Pointwise mode: 3 inputs - wrap to accept (pts_feat, sampled_pose, t)
        class ScoreNetExportWrapperPointwise(nn.Module):
            def __init__(self, score_net):
                super().__init__()
                self.score_net = score_net

            def forward(self, pts_feat, sampled_pose, t):
                # Call original forward with rgb_feat=None
                return self.score_net(pts_feat, rgb_feat=None, sampled_pose=sampled_pose, t=t)

        export_model = ScoreNetExportWrapperPointwise(score_net)
        export_model.eval()

    # Export to ONNX
    onnx_path = output_dir / "scorenet.onnx"
    print(f"\nExporting to {onnx_path}...")

    input_names = [inp['name'] for inp in input_info['inputs']]
    output_names = [out['name'] for out in input_info['outputs']]

    dynamic_axes = {
        'pts_feat': {0: 'batch_size'},
        'sampled_pose': {0: 'batch_size'},
        't': {0: 'batch_size'},
        'score': {0: 'batch_size'}
    }

    # No dynamic_axes for fixed batch_size OM models
    torch.onnx.export(
        export_model,
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,  # Fixed batch_size for OM
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=False,  # Enable optimization
        keep_initializers_as_inputs=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

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
    onnx_path = output_dir / "scalenet.onnx"
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

    return True



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


def export_dinov2_to_onnx(output_dir, device='cpu', img_size=224, num_pts=1024):
    """
    Export DINOv2 (dinov2_vits14) to ONNX.

    No patching applied - if export fails, the full error trace will be visible
    so we can diagnose the exact issue on the remote server.

    Args:
        output_dir: Directory to save ONNX model
        device: Device to load model on (use 'cpu' for ONNX export)
        img_size: Input image size (default: 224)
        num_pts: Number of points (default: 1024)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting DINOv2 (dinov2_vits14) to ONNX")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  Image size:    {img_size}")
    print(f"  Num points:    {num_pts}")
    print(f"  Feature dim:   384")
    print(f"  Feature map:   {img_size//14}x{img_size//14}")
    print(f"  Device:        {device}")

    # Load DINOv2
    print(f"\nLoading DINOv2 model...")
    import torch.hub
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino = dino.to(device)
    dino.requires_grad_(False)

    # Patch interpolate_pos_encoding to skip F.interpolate(mode='bicubic')
    # which uses Resize op with mode='cubic' not supported by Ascend 310P.
    # For fixed 224x224 input, positional embedding interpolation is unnecessary.
    _orig_interpolate_pos = dino.interpolate_pos_encoding

    def _no_resize_interpolate_pos(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        # Sizes don't match: pos_embed is for 518x518 (37x37 patches),
        # input is 224x224 (16x16 patches). Use bilinear instead of bicubic
        # (Ascend 310P only supports nearest/linear/bilinear for Resize).
        # Returns only pos_embed (caller does x = x + result).
        import torch.nn.functional as F
        import math
        previous_dtype = x.dtype
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]          # [1, dim]
        patch_pos_embed = pos_embed[:, 1:]         # [1, N, dim]
        dim = x.shape[-1]
        w0 = h0 = int(math.sqrt(N))
        target_w = w // self.patch_size
        target_h = h // self.patch_size
        patch_pos_embed = patch_pos_embed.reshape(1, w0, h0, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(target_h, target_w),
                                        mode='bilinear', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    import types
    dino.interpolate_pos_encoding = types.MethodType(_no_resize_interpolate_pos, dino)
    print(f"  Patched interpolate_pos_encoding: bicubic -> linear")

    # Verify forward_features output
    print(f"\nVerifying forward_features output keys...")
    dummy_rgb = torch.randn(1, 3, img_size, img_size, dtype=torch.float32).to(device)
    feat_dict = dino.forward_features(dummy_rgb)
    print(f"  Keys: {list(feat_dict.keys())}")
    for k, v in feat_dict.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")

    # Create wrapper and test
    export_model = DINOv2ExportWrapper(dino, dino_dim=384, img_size=img_size)
    export_model.eval()

    dummy_xs = torch.randint(0, img_size, (1, num_pts), dtype=torch.int64).to(device)
    dummy_ys = torch.randint(0, img_size, (1, num_pts), dtype=torch.int64).to(device)
    with torch.no_grad():
        test_out = export_model(dummy_rgb, dummy_xs, dummy_ys)
    print(f"\nWrapper test: input {dummy_rgb.shape} -> output {test_out.shape}")

    # Export
    onnx_path = output_dir / "dinov2_vits14.onnx"
    print(f"\nExporting to {onnx_path}...")

    torch.onnx.export(
        export_model,
        (dummy_rgb, dummy_xs, dummy_ys),
        str(onnx_path),
        input_names=['roi_rgb', 'roi_xs', 'roi_ys'],
        output_names=['rgb_feat'],
        dynamic_axes={
            'roi_rgb': {0: 'batch_size'},
            'roi_xs': {0: 'batch_size'},
            'roi_ys': {0: 'batch_size'},
            'rgb_feat': {0: 'batch_size'},
        },
        opset_version=17,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"✓ ONNX export successful: {onnx_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verification passed")

    # Fix Resize nodes: Ascend ATC does not support coordinate_transformation_mode=half_pixel
    # Change to asymmetric (equivalent when this path is not actually reached at runtime)
    resize_fixed = False
    for node in onnx_model.graph.node:
        if node.op_type == 'Resize':
            for attr in node.attribute:
                if attr.name == 'coordinate_transformation_mode' and attr.s == b'half_pixel':
                    attr.s = b'asymmetric'
                    resize_fixed = True
                    print(f"  Fixed Resize node '{node.name}': half_pixel -> asymmetric")
    if resize_fixed:
        onnx.save(onnx_model, str(onnx_path))
        print(f"  ONNX model saved with Resize fix")
    else:
        print(f"  No Resize fix needed (no half_pixel mode found)")

    print(f"\nInput info:")
    print(f"  roi_rgb:  [batch_size, 3, {img_size}, {img_size}], float32")
    print(f"  roi_xs:   [batch_size, {num_pts}], int64")
    print(f"  roi_ys:   [batch_size, {num_pts}], int64")
    print(f"\nOutput info:")
    print(f"  rgb_feat: [batch_size, {num_pts}, 384], float32")

    return True


def main():
    parser = argparse.ArgumentParser(description='Export GenPose2 Networks to ONNX')
    parser.add_argument('--agent_type', type=str, default='score',
                        choices=['score', 'energy', 'scale', 'pointnet2_from_score', 'pointnet2_from_energy', 'pointnet2_scorenet', 'dinov2'],
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
        args.om_batch_size = getattr(args, "om_batch_size") or 800
        success = export_score_network_to_onnx(checkpoint_path, args.output_dir, cfg, args.device, args.om_batch_size)

        if success:
            print(f"\n{'='*60}")
            print("Export completed successfully!")
            print(f"{'='*60}")
            print(f"\nExported files:")
            print(f"  - {args.output_dir}/scorenet.onnx")

    elif args.agent_type == 'pointnet2_from_score':
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = getattr(cfg, 'pretrained_score_model_path',
                                        None) or './results/ckpts/ScoreNet/scorenet.pth'
            print(f"Auto-detected checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
            print(f"Please specify the correct path with --checkpoint_path")
            return

        args.om_batch_size = getattr(args, "om_batch_size") or 16
        success = export_pointnet2_to_onnx(checkpoint_path, args.output_dir, cfg, args.device, args.om_batch_size,
                                           output_name='pointnet2_from_score.onnx')

        if success:
            print(f"\n{'='*60}")
            print("Export completed successfully!")
            print(f"{'='*60}")
            print(f"\nExported files:")
            print(f"  - {args.output_dir}/pointnet2_from_score.onnx")

    elif args.agent_type == 'pointnet2_from_energy':
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = getattr(cfg, 'pretrained_energy_model_path',
                                        None) or './results/ckpts/EnergyNet/energynet.pth'
            print(f"Auto-detected checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
            print(f"Please specify the correct path with --checkpoint_path")
            return

        args.om_batch_size = getattr(args, "om_batch_size") or 16
        success = export_pointnet2_to_onnx(checkpoint_path, args.output_dir, cfg, args.device, args.om_batch_size,
                                           output_name='pointnet2_from_energy.onnx')

        if success:
            print(f"\n{'='*60}")
            print("Export completed successfully!")
            print(f"{'='*60}")
            print(f"\nExported files:")
            print(f"  - {args.output_dir}/pointnet2_from_energy.onnx")

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
            print(f"  - {args.output_dir}/scalenet.onnx")

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
        success = export_pointnet2_scorenet_to_onnx(
            checkpoint_path, args.output_dir, cfg, args.device, args.om_batch_size
        )

        if success:
            print(f"\n{'='*60}")
            print("Export completed successfully!")
            print(f"{'='*60}")
            print(f"\nExported files:")
            print(f"  - {args.output_dir}/pointnet2_scorenet.onnx")

    elif args.agent_type == 'dinov2':
        # DINOv2 doesn't need a checkpoint - loaded from torch.hub
        img_size = getattr(cfg, 'img_size', 224)
        success = export_dinov2_to_onnx(args.output_dir, args.device, img_size=img_size)

        if success:
            print(f"\n{'='*60}")
            print("Export completed successfully!")
            print(f"{'='*60}")
            print(f"\nExported files:")
            print(f"  - {args.output_dir}/dinov2_vits14.onnx")
            print(f"\nNext steps:")
            print(f"1. Convert ONNX to OM using ATC tool:")
            print(f"   python runners/onnx2om.py --onnx_path {args.output_dir}/dinov2_vits14.onnx")

    if success:
        print(f"\nNext steps:")
        print(f"1. Convert ONNX to OM using ATC tool:")
        print(f"   python runners/onnx2om.py --onnx_path {args.output_dir}/{args.agent_type}_network.onnx")
        print(f"\n2. Use the OM model in NPU inference (TODO: implement infer_om.py)")
    else:
        print("\nExport failed. Please check the error messages above.")


if __name__ == '__main__':
    main()
