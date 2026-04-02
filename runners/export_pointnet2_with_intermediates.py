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
    PointNet2 包装器，返回中间层输出

    在原始 pts_encoder 的基础上，添加钩子捕获 SA_modules 的输出
    """

    def __init__(self, pts_encoder):
        super().__init__()
        self.pts_encoder = pts_encoder
        self.intermediates = []

        # 注册前向钩子来捕获 SA_modules 的输出
        self.hooks = []
        for i, sa_module in enumerate(self.pts_encoder.SA_modules):
            hook = sa_module.register_forward_hook(
                lambda module, input, output, idx=i: self.intermediates.append(output)
            )
            self.hooks.append(hook)

    def forward(self, pointcloud):
        """
        Args:
            pointcloud: [bs, 1024, 387]

        Returns:
            output: [bs, 1024] - 最终输出
            intermediate_0, intermediate_1, ...: 每个 SA_module 的输出
        """
        # 清空之前的中间结果
        self.intermediates.clear()

        # 调用原始 forward
        output = self.pts_encoder(pointcloud)

        # 返回最终输出 + 所有中间输出
        return output, *self.intermediates


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

    # 准备输出名称
    output_names = ['pts_feat']  # 最终输出
    for i in range(num_sa_modules):
        output_names.append(f'sa_{i}_output')

    print(f"\nOutput info:")
    print(f"  Total outputs: {len(output_names)}")
    for name in output_names:
        print(f"    - {name}")

    # 导出 ONNX
    onnx_path = output_dir / "pointnet2_with_intermediates.onnx"
    print(f"\nExporting to {onnx_path}...")
    print("This may take a few minutes...")

    input_names = ['pointcloud']

    # dynamic_axes 需要包含所有输出
    dynamic_axes = {'pointcloud': {0: 'batch_size'}}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    try:
        torch.onnx.export(
            export_model,
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
            'description': 'Each SA_module output is included as a separate ONNX output'
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

    print(f"\n✓ Got {len(outputs)} outputs:")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape={out.shape}, mean={out.mean():.6f}, std={out.std():.6f}")

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
