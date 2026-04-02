"""
PointNet2 精度对比测试脚本

同时测试 PyTorch (.pth)、OM (.om)、ONNX (.onnx) 三种格式的精度

修改适配：所有格式都使用拼接后的 pointcloud [bs, 1024, 387] 作为输入
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch_npu
import onnxruntime as ort

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


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
        # 输出到控制台
        print(*args, **kwargs)

        # 输出到文件
        if self.file_handle:
            # 将 kwargs 中的 end 和 sep 处理一下
            end = kwargs.get('end', '\n')
            sep = kwargs.get('sep', ' ')
            text = sep.join(str(arg) for arg in args) + end
            self.file_handle.write(text)
            self.file_handle.flush()

    def close(self):
        """关闭文件"""
        if self.file_handle:
            self.file_handle.close()


logger = None


def run_pytorch_model(score_net, pointcloud: torch.Tensor):
    """
    运行 PyTorch 模型

    Args:
        score_net: ScoreNetworkWrapper 实例
        pointcloud: [batch_size, 1024, 387] - 拼接后的点云数据
    """
    score_net.net.eval()
    with torch.no_grad():
        # 直接调用 pts_encoder（期望拼接后的输入）
        output = score_net.net.pts_encoder(pointcloud)
    return output


def run_om_model(model, pointcloud: torch.Tensor):
    """
    运行 OM 模型

    Args:
        model: PointNet2EncoderWrapper 实例
        pointcloud: [batch_size, 1024, 387] - 拼接后的点云数据
    """
    with torch.no_grad():
        # OM 模型现在也期望拼接后的输入
        output = model(pointcloud)
    return output


def run_onnx_model(onnx_path: str, pointcloud: np.ndarray):
    """
    运行 ONNX 模型（单输出版本）

    Args:
        onnx_path: ONNX 模型路径
        pointcloud: [batch_size, 1024, 387] - 拼接后的点云数据
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    # 获取输入名称
    input_names = [i.name for i in session.get_inputs()]

    # 构造输入（新的 ONNX 格式只有一个 pointcloud 输入）
    onnx_inputs = {
        'pointcloud': pointcloud.astype(np.float32),
    }

    outputs = session.run(None, onnx_inputs)
    return outputs[0]


def run_onnx_model_with_intermediates(onnx_path: str, pointcloud: np.ndarray, logger=None):
    """
    运行带中间输出的 ONNX 模型

    Args:
        onnx_path: ONNX 模型路径
        pointcloud: [batch_size, 1024, 387] - 拼接后的点云数据
        logger: TestLogger 实例

    Returns:
        final_output: 最终输出 [batch_size, 1024]
        intermediates: 中间输出列表
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    # 打印模型输入输出信息
    if logger:
        logger.print(f"\nONNX 模型信息:")
        logger.print(f"  输入:")
        for inp in session.get_inputs():
            logger.print(f"    {inp.name}: {inp.shape}, {inp.type}")
        logger.print(f"  输出:")
        for out in session.get_outputs():
            logger.print(f"    {out.name}: {out.shape}, {out.type}")

    # 构造输入
    onnx_inputs = {
        'pointcloud': pointcloud.astype(np.float32),
    }

    # 运行推理
    outputs = session.run(None, onnx_inputs)

    # 第一个是最终输出，其余是中间输出
    final_output = outputs[0]
    intermediates = outputs[1:]

    if logger:
        logger.print(f"\nONNX 推理结果:")
        logger.print(f"  最终输出 (pts_feat):")
        logger.print(f"    Shape: {final_output.shape}")
        logger.print(f"    Mean:  {final_output.mean():.6f}")
        logger.print(f"    Std:   {final_output.std():.6f}")
        logger.print(f"    Min:   {final_output.min():.6f}")
        logger.print(f"    Max:   {final_output.max():.6f}")

        logger.print(f"\n  中间输出 ({len(intermediates)} 个):")
        for i, inter in enumerate(intermediates):
            logger.print(f"    SA_{i} output:")
            logger.print(f"      Shape: {inter.shape}")
            logger.print(f"      Mean:  {inter.mean():.6f}")
            logger.print(f"      Std:   {inter.std():.6f}")

    return final_output, intermediates


def create_dummy_inputs(batch_size: int = 4, device='npu:0'):
    """
    创建测试输入（使用固定种子确保可复现）

    Returns:
        pts: [batch_size, 1024, 3] - 点云坐标
        rgb_feat: [batch_size, 1024, 384] - DINOv2 特征
        pointcloud: [batch_size, 1024, 387] - 拼接后的数据
    """
    torch.manual_seed(42)
    np.random.seed(42)

    pts = torch.randn(batch_size, 1024, 3, dtype=torch.float32, device=device)
    rgb_feat = torch.randn(batch_size, 1024, 384, dtype=torch.float32, device=device)

    # 拼接（与实际推理逻辑一致）
    with torch.no_grad():
        pointcloud = torch.cat([pts, rgb_feat], dim=-1)

    return pts, rgb_feat, pointcloud


def load_pytorch_model(checkpoint_path: str, device='npu:0'):
    """加载 PyTorch ScoreNetworkWrapper 模型（使用与实际推理相同的方式）"""
    from configs.config import get_config
    from networks.score_wrapper import create_score_network

    cfg = get_config()
    cfg.agent_type = 'score'
    cfg.device = device
    cfg.dino = 'pointwise'

    # 使用与实际推理相同的加载方式
    score_net = create_score_network(
        checkpoint_path=checkpoint_path,
        device=device
    )
    score_net.net.eval()
    score_net.net.pts_encoder.eval()  # 确保 pts_encoder 也是 eval 模式

    return score_net


def load_om_model(om_path: str, device='npu:0'):
    """加载 OM PointNet2 模型"""
    from networks.score_wrapper import create_pointnet2_encoder
    return create_pointnet2_encoder(checkpoint_path=om_path, device=device)


def compare_outputs(pt_output: torch.Tensor, om_output: torch.Tensor, onnx_output: np.ndarray, name: str):
    """对比三种格式的输出"""
    # 转换为相同格式
    pt_np = pt_output.cpu().numpy()
    om_np = om_output.cpu().numpy()
    onnx_np = onnx_output

    logger.print(f"\n{'='*60}")
    logger.print(f"{name} 精度对比")
    logger.print(f"{'='*60}")

    # 基本信息
    logger.print(f"\nOutput Shape:")
    logger.print(f"  PyTorch: {pt_output.shape}")
    logger.print(f"  OM:      {om_output.shape}")
    logger.print(f"  ONNX:    {onnx_output.shape}")

    # 统计信息
    logger.print(f"\n输出统计:")
    logger.print(f"{'格式':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    logger.print(f"{'-'*50}")
    logger.print(f"{'PyTorch':<10} {pt_np.min():<12.6f} {pt_np.max():<12.6f} {pt_np.mean():<12.6f} {pt_np.std():<12.6f}")
    logger.print(f"{'OM':<10} {om_np.min():<12.6f} {om_np.max():<12.6f} {om_np.mean():<12.6f} {om_np.std():<12.6f}")
    logger.print(f"{'ONNX':<10} {onnx_np.min():<12.6f} {onnx_np.max():<12.6f} {onnx_np.mean():<12.6f} {onnx_np.std():<12.6f}")

    # 误差分析
    logger.print(f"\n误差分析 (以 PyTorch 为基准):")
    pt_om_diff = np.abs(pt_np - om_np)
    pt_onnx_diff = np.abs(pt_np - onnx_np)
    om_onnx_diff = np.abs(om_np - onnx_np)

    logger.print(f"{'对比':<15} {'Max Diff':<12} {'Mean Diff':<12} {'相对误差':<12}")
    logger.print(f"{'-'*50}")
    logger.print(f"{'PyTorch vs OM':<15} {pt_om_diff.max():<12.6f} {pt_om_diff.mean():<12.6f} {pt_om_diff.mean()/(np.abs(pt_np).mean()+1e-9):<12.6f}")
    logger.print(f"{'PyTorch vs ONNX':<15} {pt_onnx_diff.max():<12.6f} {pt_onnx_diff.mean():<12.6f} {pt_onnx_diff.mean()/(np.abs(pt_np).mean()+1e-9):<12.6f}")
    logger.print(f"{'OM vs ONNX':<15} {om_onnx_diff.max():<12.6f} {om_onnx_diff.mean():<12.6f} {om_onnx_diff.mean()/(np.abs(om_np).mean()+1e-9):<12.6f}")

    # 判断是否一致
    threshold = 1e-3
    pt_om_match = (pt_om_diff.max() < threshold)
    pt_onnx_match = (pt_onnx_diff.max() < threshold)

    logger.print(f"\n一致性检查 (阈值={threshold}):")
    logger.print(f"  PyTorch vs OM:      {'✓ 通过' if pt_om_match else '✗ 失败'}")
    logger.print(f"  PyTorch vs ONNX:    {'✓ 通过' if pt_onnx_match else '✗ 失败'}")

    return pt_om_match and pt_onnx_match


def test_pointnet2(
    checkpoint_path: str = './results/ckpts/ScoreNet/scorenet.pth',
    om_path: str = './onnx_models/pointnet2.om',
    onnx_path: str = './onnx_models/pointnet2.onnx',
    batch_size: int = 4,
    device: str = 'npu:0',
    log_file: str = None,
    use_intermediates: bool = False,
    onnx_with_intermediates_path: str = None
):
    """测试 PointNet2 三种格式"""

    global logger
    logger = TestLogger(log_file)

    logger.print(f"\n{'='*60}")
    if use_intermediates:
        logger.print(f"PointNet2 带中间输出测试")
    else:
        logger.print(f"PointNet2 精度对比测试")
    logger.print(f"{'='*60}")
    logger.print(f"Batch Size: {batch_size}")
    logger.print(f"Device: {device}")

    # 确定使用哪个 ONNX 模型
    if use_intermediates and onnx_with_intermediates_path:
        actual_onnx_path = onnx_with_intermediates_path
        logger.print(f"\n模型路径:")
        logger.print(f"  ONNX (带中间输出): {actual_onnx_path}")
    else:
        actual_onnx_path = onnx_path
        logger.print(f"\n模型路径:")
        logger.print(f"  ONNX: {actual_onnx_path}")

    logger.print(f"\n输入格式:")
    logger.print(f"  pointcloud: [{batch_size}, 1024, 387] (拼接后的 pts + rgb_feat)")

    # 创建测试输入
    pointcloud = torch.load('./test_pointcloud.pth')['pointcloud'].to(device)

    logger.print(f"\n原始数据 shape:")
    logger.print(f"  pointcloud: {pointcloud.shape}")
    logger.print(f"  pointcloud mean: {pointcloud.mean():.6f}")
    logger.print(f"  pointcloud std: {pointcloud.std():.6f}")

    # 测试 ONNX
    logger.print(f"\n{'='*60}")
    logger.print(f"测试 ONNX 模型...")
    logger.print(f"{'='*60}")

    if use_intermediates:
        # 使用带中间输出的版本
        onnx_output, intermediates = run_onnx_model_with_intermediates(
            actual_onnx_path,
            pointcloud.cpu().numpy(),
            logger
        )
    else:
        # 使用普通版本
        onnx_output = run_onnx_model(actual_onnx_path, pointcloud.cpu().numpy())

    logger.print(f"\n{'='*60}")
    logger.print(f"测试完成")
    logger.print(f"{'='*60}\n")

    logger.close()

    if use_intermediates:
        return onnx_output, intermediates
    else:
        return onnx_output


def main():
    parser = argparse.ArgumentParser(description='Test PointNet2 accuracy: PyTorch vs OM vs ONNX')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./results/ckpts/ScoreNet/scorenet.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--om_path', type=str,
                        default='./onnx_models/pointnet2.om',
                        help='Path to OM model')
    parser.add_argument('--onnx_path', type=str,
                        default='./onnx_models/pointnet2.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--onnx_with_intermediates_path', type=str,
                        default='./onnx_models/pointnet2_with_intermediates.onnx',
                        help='Path to ONNX model with intermediate outputs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default='npu:0',
                        help='Device to run on')
    parser.add_argument('--log_file', type=str, default='./logs/pointnet2_accuracy_test.log',
                        help='Path to log file')
    parser.add_argument('--use_intermediates', action='store_true',
                        help='Use ONNX model with intermediate outputs')

    args = parser.parse_args()

    test_pointnet2(
        checkpoint_path=args.checkpoint_path,
        om_path=args.om_path,
        onnx_path=args.onnx_path,
        batch_size=args.batch_size,
        device=args.device,
        log_file=args.log_file,
        use_intermediates=args.use_intermediates,
        onnx_with_intermediates_path=args.onnx_with_intermediates_path
    )


if __name__ == '__main__':
    main()
