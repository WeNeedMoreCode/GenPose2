"""
基于现有 export_onnx.py 逻辑，添加多输出导出功能

在 ONNX 推理时也能获取指定位置的中间输出
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from pathlib import Path
from configs.config import get_config
from networks.posenet_agent import PoseNet


class PointNet2WithIntermediates(nn.Module):
    """
    PointNet2 包装器，返回中间层输出（字典格式）

    直接复制原始 forward 逻辑，在关键位置保存中间输出
    """

    def __init__(self, pts_encoder):
        super().__init__()
        self.pts_encoder = pts_encoder
        # 复制必要的属性
        self.SA_modules = pts_encoder.SA_modules

        # 定义输出的顺序（用于 ONNX 导出）
        # 每个 SA_module 保存：执行前的 l_features[i] 和执行后的 li_features
        self.output_keys = []
        num_sa_modules = len(self.SA_modules)
        for i in range(num_sa_modules):
            self.output_keys.append(f'sa_{i}_input')   # SA_module 执行前的输入
            self.output_keys.append(f'sa_{i}_output')  # SA_module 执行后的输出
        self.output_keys.append('pts_feat')            # 最终输出

    def _break_up_pc(self, pc):
        """从原始 pts_encoder 复制的方法"""
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud):
        """
        复制原始 forward 逻辑，但保存每个 SA_module 后的特征

        Args:
            pointcloud: [bs, 1024, 387]

        Returns:
            dict: 包含最终输出和所有中间特征的字典
                {
                    'sa_0_input': [bs, F_0_in, 1024],   # SA_module[0] 执行前的输入
                    'sa_0_output': [bs, F_0, npoint_0],  # SA_module[0] 执行后的输出
                    'sa_1_input': [bs, F_1_in, npoint_0],
                    'sa_1_output': [bs, F_1, npoint_1],
                    ...
                    'pts_feat': [bs, 1024],
                }
        """
        # 复制原始 forward 的逻辑
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]

        # 用于存储所有输出的字典
        outputs = {}

        # first SA_module
        outputs['sa_0_input'] = l_features[0].clone()
        li_xyz, li_features, idx = self.SA_modules[0](l_xyz[0], l_features[0], return_idx=True)
        l_xyz.append(li_xyz)
        l_features.append(li_features)
        outputs['sa_0_output'] = li_features.clone()

        features = torch.gather(features, 2,
                    torch.unsqueeze(idx.type(torch.int64), 1).expand(-1, features.shape[1], -1))

        # middle SA_modules
        for i in range(1, len(self.SA_modules) - 1):
            l_features[i] = torch.concatenate([l_features[i], features], dim=1)
            outputs[f'sa_{i}_input'] = l_features[i].clone()
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i], return_idx=True)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            outputs[f'sa_{i}_output'] = li_features.clone()

            features = torch.gather(features, 2,
                        torch.unsqueeze(idx.type(torch.int64), 1).expand(-1, features.shape[1], -1))

        # last SA_module
        i += 1
        l_features[i] = torch.concatenate([l_features[i], features], dim=1)
        outputs[f'sa_{i}_input'] = l_features[i].clone()
        li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i], return_idx=True)
        l_xyz.append(li_xyz)
        l_features.append(li_features)
        outputs[f'sa_{i}_output'] = li_features.clone()

        # 最终输出
        output = l_features[-1].squeeze(-1)
        outputs['pts_feat'] = output

        return outputs

    def forward_as_tuple(self, pointcloud):
        """
        返回元组格式（用于 ONNX 导出）

        ONNX 不支持字典输出，因此提供这个方法将字典转换为有序元组
        元组的顺序与 self.output_keys 一致

        Args:
            pointcloud: [bs, 1024, 387]

        Returns:
            tuple: (sa_0_input, sa_0_output, sa_1_input, sa_1_output, ..., pts_feat)
        """
        outputs_dict = self.forward(pointcloud)
        return tuple(outputs_dict[key] for key in self.output_keys)


def export_pointnet2_with_intermediates(
    checkpoint_path='./results/ckpts/ScoreNet/scorenet.pth',
    output_dir='./onnx_models',
    device='cpu',
    om_batch_size=1
):
    """
    导出带中间层输出的 PointNet2 ONNX

    基于现有 export_onnx.py 的逻辑，添加多输出支持
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting PointNet2 with Intermediate Outputs")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  OM Batch Size: {om_batch_size}")

    # ========== 完全复制 export_onnx.py 的加载逻辑 ==========
    print(f"\nLoading PointNet2 encoder from ScoreNet checkpoint...")
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

    print(f"  ✓ pts_encoder type: {type(pts_encoder).__name__}")
    print(f"  ✓ pts_encoder.training: {pts_encoder.training}")

    # 打印第一层权重确认加载正确
    first_param = list(pts_encoder.parameters())[0]
    print(f"  ✓ First param: shape={first_param.shape}, mean={first_param.mean().item():.6f}")
    # ==========================================================

    # 创建多输出包装器
    num_sa_modules = len(pts_encoder.SA_modules)
    print(f"\n✓ Found {num_sa_modules} SA_modules")

    export_model = PointNet2WithIntermediates(pts_encoder)
    export_model.eval()

    # 准备输入（与 export_onnx.py 相同）
    batch_size = 1  # ONNX 导出时通常用 1
    pointcloud = torch.randn(batch_size, 1024, 387, dtype=torch.float32)

    print(f"\nInput info:")
    print(f"  pointcloud: {pointcloud.shape}, float32")

    # 从模型获取输出键（已经定义好的顺序）
    output_keys = export_model.output_keys

    print(f"\nOutput info:")
    print(f"  Total outputs: {len(output_keys)}")
    for key in output_keys:
        print(f"    - {key}")

    # 导出 ONNX
    onnx_path = output_dir / "pointnet2_with_intermediates.onnx"
    print(f"\nExporting to {onnx_path}...")
    print("This may take a few minutes...")

    input_names = ['pointcloud']
    output_names = output_keys  # 使用相同的键作为输出名称

    # dynamic_axes 需要包含所有输出
    dynamic_axes = {'pointcloud': {0: 'batch_size'}}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    try:
        # 用适配器模块包装，将 dict 输出转为 tuple（ONNX 要求）
        class _TupleAdapter(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, pointcloud):
                return self.inner.forward_as_tuple(pointcloud)

        adapter = _TupleAdapter(export_model)

        torch.onnx.export(
            adapter,
            pointcloud,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            verbose=False,
            export_params=True,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
        )

        print(f"✓ ONNX export successful: {onnx_path}")
        print(f"  File size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")

        # 保存 metadata
        import json
        metadata_path = output_dir / "pointnet2_intermediates_metadata.json"
        metadata = {
            'model_type': 'PointNet2WithIntermediates',
            'num_sa_modules': num_sa_modules,
            'checkpoint_path': str(checkpoint_path),
            'device': device,
            'outputs': output_names,
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}")

        return str(onnx_path)

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_onnx_with_intermediates(
    onnx_path='./onnx_models/pointnet2_with_intermediates.onnx',
    pointcloud_path=None  # 可选：加载保存的输入
):
    """
    测试带中间输出的 ONNX 模型
    """
    import onnxruntime as ort
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Testing ONNX with Intermediate Outputs")
    print(f"{'='*60}")

    # 加载模型
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # 打印输入输出信息
    print(f"\nModel Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}, {inp.type}")

    print(f"\nModel Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}, {out.type}")

    # 准备输入
    if pointcloud_path and os.path.exists(pointcloud_path):
        pointcloud = torch.load(pointcloud_path).numpy().astype(np.float32)
        print(f"\n✓ Loaded input from: {pointcloud_path}")
    else:
        pointcloud = np.random.randn(1, 1024, 387).astype(np.float32)
        print(f"\n✓ Using random input")

    print(f"  Input shape: {pointcloud.shape}")
    print(f"  Input mean: {pointcloud.mean():.6f}")

    # 运行推理
    print(f"\nRunning inference...")
    outputs = session.run(None, {'pointcloud': pointcloud})

    output_names = [o.name for o in session.get_outputs()]
    print(f"\n✓ Got {len(outputs)} outputs:")
    for name, out in zip(output_names, outputs):
        print(f"  {name}: shape={out.shape}, mean={out.mean():.6f}, std={out.std():.6f}")

    return outputs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export PointNet2 with intermediate outputs')
    parser.add_argument('--mode', type=str, default='export',
                        choices=['export', 'test'],
                        help='Mode: export or test')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./results/ckpts/ScoreNet/scorenet.pth',
                        help='Path to ScoreNet checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default='./onnx_models',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--onnx_path', type=str,
                        default='./onnx_models/pointnet2_with_intermediates.onnx',
                        help='Path to ONNX model (for test mode)')
    parser.add_argument('--pointcloud_path', type=str, default=None,
                        help='Path to saved pointcloud (for test mode)')

    args = parser.parse_args()

    if args.mode == 'export':
        export_pointnet2_with_intermediates(
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            device=args.device
        )
    elif args.mode == 'test':
        test_onnx_with_intermediates(
            onnx_path=args.onnx_path,
            pointcloud_path=args.pointcloud_path
        )
