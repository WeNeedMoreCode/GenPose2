"""
PointNet2 精度对比测试脚本

对比 PyTorch 和 ONNX 两种格式的精度，支持逐 SA_module 的 input/output 对比
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

import torch
import onnxruntime as ort

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from runners.export_pointnet2_with_intermediates import PointNet2WithIntermediates


class TestLogger:
    """日志记录器，同时输出到控制台和文件"""

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.file_handle = open(log_file, 'w', encoding='utf-8')
        else:
            self.file_handle = None

    def print(self, *args, **kwargs):
        """同时输出到控制台和文件"""
        print(*args, **kwargs)
        if self.file_handle:
            end = kwargs.get('end', '\n')
            sep = kwargs.get('sep', ' ')
            text = sep.join(str(arg) for arg in args) + end
            self.file_handle.write(text)
            self.file_handle.flush()

    def close(self):
        """关闭文件"""
        if self.file_handle:
            self.file_handle.close()


def run_pytorch_with_intermediates(checkpoint_path: str, pointcloud: torch.Tensor, device: str = 'cpu', logger=None):
    """
    使用 PointNet2WithIntermediates 运行 PyTorch 模型，获取每个 SA_module 的输入输出

    Returns:
        dict: {'sa_0_input': ..., 'sa_0_output': ..., ..., 'pts_feat': ...}
    """
    from configs.config import get_config
    from networks.posenet_agent import PoseNet

    score_cfg = get_config()
    score_cfg.agent_type = 'score'
    score_cfg.device = device
    score_cfg.dino = 'pointwise'

    agent = PoseNet(score_cfg)
    agent.load_ckpt(model_dir=checkpoint_path, model_path=True, load_model_only=True)
    agent.net.eval()

    pts_encoder = agent.net.pts_encoder
    pts_encoder.eval()

    model = PointNet2WithIntermediates(pts_encoder)
    model.eval()

    with torch.no_grad():
        pc = pointcloud.to(device).float()
        outputs = model(pc)

    # 转为 numpy
    results = {}
    for key, val in outputs.items():
        results[key] = val.cpu().numpy()

    if logger:
        logger.print(f"\nPyTorch 推理结果:")
        for key in model.output_keys:
            arr = results[key]
            logger.print(f"  {key}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")

    return results, model.output_keys


def run_onnx_with_intermediates(onnx_path: str, pointcloud: np.ndarray, logger=None):
    """
    运行带中间输出的 ONNX 模型

    Returns:
        dict: {'sa_0_input': ..., 'sa_0_output': ..., ..., 'pts_feat': ...}
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    if logger:
        logger.print(f"\nONNX 模型信息:")
        logger.print(f"  输入:")
        for inp in session.get_inputs():
            logger.print(f"    {inp.name}: {inp.shape}, {inp.type}")
        logger.print(f"  输出:")
        for out in session.get_outputs():
            logger.print(f"    {out.name}: {out.shape}, {out.type}")

    outputs = session.run(None, {'pointcloud': pointcloud.astype(np.float32)})
    output_names = [o.name for o in session.get_outputs()]

    results = {}
    for name, arr in zip(output_names, outputs):
        results[name] = arr

    if logger:
        logger.print(f"\nONNX 推理结果:")
        for name in output_names:
            arr = results[name]
            logger.print(f"  {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")

    return results, output_names


def compare_intermediates(pt_results: dict, onnx_results: dict, output_keys: list, logger=None):
    """
    逐层对比 PyTorch 和 ONNX 的中间输出
    """
    logger.print(f"\n{'='*80}")
    logger.print(f"逐层精度对比 (PyTorch vs ONNX)")
    logger.print(f"{'='*80}")

    for key in output_keys:
        pt = pt_results[key]
        onnx = onnx_results[key]

        if pt.shape != onnx.shape:
            logger.print(f"\n  {key}: *** SHAPE 不匹配 *** PyTorch={pt.shape} vs ONNX={onnx.shape}")
            continue

        # fps_idx 是 int64 索引，用精确匹配判断
        if 'fps_idx' in key:
            match = np.array_equal(pt, onnx)
            diff_count = np.sum(pt != onnx)
            logger.print(f"\n  {key}: shape={pt.shape} (int64 indices)")
            logger.print(f"    完全匹配: {'YES' if match else 'NO'}")
            if not match:
                logger.print(f"    不匹配元素数: {diff_count} / {pt.size}")
                # 打印前几个不匹配的位置
                mismatches = np.where(pt != onnx)
                n_show = min(5, len(mismatches[0]))
                for j in range(n_show):
                    b, k = mismatches[0][j], mismatches[1][j]
                    logger.print(f"      [{b},{k}] PyTorch={pt[b,k]} vs ONNX={onnx[b,k]}")
            continue

        diff = np.abs(pt - onnx)
        rel_err = diff.mean() / (np.abs(pt).mean() + 1e-9)

        logger.print(f"\n  {key}: shape={pt.shape}")
        logger.print(f"    PyTorch: mean={pt.mean():.6f}, std={pt.std():.6f}, min={pt.min():.6f}, max={pt.max():.6f}")
        logger.print(f"    ONNX:    mean={onnx.mean():.6f}, std={onnx.std():.6f}, min={onnx.min():.6f}, max={onnx.max():.6f}")
        logger.print(f"    Diff:    mean={diff.mean():.8f}, max={diff.max():.8f}, rel_err={rel_err:.6f}")

        if rel_err < 0.01:
            logger.print(f"    结果: PASS")
        elif rel_err < 0.05:
            logger.print(f"    结果: WARN (rel_err={rel_err:.4f})")
        else:
            logger.print(f"    结果: FAIL (rel_err={rel_err:.4f})")


def main():
    parser = argparse.ArgumentParser(description='PointNet2 逐层精度对比: PyTorch vs ONNX')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./results/ckpts/ScoreNet/scorenet.pth')
    parser.add_argument('--onnx_path', type=str,
                        default='./onnx_models/pointnet2_with_intermediates.onnx')
    parser.add_argument('--pointcloud_path', type=str,
                        default='./test_pointcloud.pth',
                        help='保存的 pointcloud 输入文件')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_file', type=str, default='./logs/pointnet2_accuracy_test.log')

    args = parser.parse_args()

    logger = TestLogger(args.log_file)

    logger.print(f"\n{'='*60}")
    logger.print(f"PointNet2 逐层精度对比 (PyTorch vs ONNX)")
    logger.print(f"{'='*60}")
    logger.print(f"  checkpoint: {args.checkpoint_path}")
    logger.print(f"  onnx:       {args.onnx_path}")
    logger.print(f"  input:      {args.pointcloud_path}")
    logger.print(f"  device:     {args.device}")

    # 加载输入
    if os.path.exists(args.pointcloud_path):
        data = torch.load(args.pointcloud_path)
        if isinstance(data, dict) and 'pointcloud' in data:
            pointcloud = data['pointcloud']
        else:
            pointcloud = data
        logger.print(f"\n  pointcloud: shape={pointcloud.shape}, mean={pointcloud.mean():.6f}")
    else:
        logger.print(f"\n  *** 输入文件不存在: {args.pointcloud_path} ***")
        logger.close()
        return

    # PyTorch 推理
    logger.print(f"\n{'='*60}")
    logger.print(f"运行 PyTorch 模型...")
    logger.print(f"{'='*60}")
    pt_results, output_keys = run_pytorch_with_intermediates(
        args.checkpoint_path, pointcloud, device=args.device, logger=logger
    )

    # ONNX 推理
    logger.print(f"\n{'='*60}")
    logger.print(f"运行 ONNX 模型...")
    logger.print(f"{'='*60}")
    onnx_results, onnx_output_names = run_onnx_with_intermediates(
        args.onnx_path, pointcloud.cpu().numpy(), logger=logger
    )

    # 逐层对比
    compare_intermediates(pt_results, onnx_results, output_keys, logger=logger)

    logger.print(f"\n{'='*60}")
    logger.print(f"测试完成，日志已保存到: {args.log_file}")
    logger.print(f"{'='*60}")

    logger.close()


if __name__ == '__main__':
    main()
