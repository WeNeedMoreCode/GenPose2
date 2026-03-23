"""
ONNX Model Testing Script

Tests the exported ONNX model using ONNXRuntime CPU provider.

Usage:
    python runners/test_onnx.py
    python runners/test_onnx.py --onnx_path ./onnx_models/score_network.onnx
    python runners/test_onnx.py --batch_size 4
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
from pathlib import Path

import onnxruntime as ort

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def run_onnxruntime_cpu(
    onnx_model_path: str, inputs: dict[str, np.ndarray]
) -> tuple[list[np.ndarray], list[str], float]:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(
        str(onnx_model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    input_names = [i.name for i in session.get_inputs()]

    onnx_inputs = {name: inputs[name] for name in input_names}

    _ = session.run(None, onnx_inputs)

    num_runs = 10
    start = time.perf_counter()
    for _ in range(num_runs):
        outputs = session.run(None, onnx_inputs)
    elapsed = (time.perf_counter() - start) / num_runs

    return [np.asarray(output) for output in outputs], input_names, elapsed


def create_dummy_inputs(batch_size: int = 1) -> dict[str, np.ndarray]:
    np.random.seed(42)

    inputs = {
        'pts_feat': np.random.randn(batch_size, 1024).astype(np.float32),
        'rgb_feat': np.random.randn(batch_size, 384).astype(np.float32),
        'sampled_pose': np.random.randn(batch_size, 9).astype(np.float32),
        't': np.ones((batch_size, 1), dtype=np.float32) * 0.5,
    }

    return inputs


def test_onnx_model(onnx_path: str, batch_size: int = 1, compare_with_pytorch: bool = True):
    onnx_path = Path(onnx_path)
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX: {onnx_path}, {file_size_mb:.2f} MB, batch={batch_size}")

    inputs = create_dummy_inputs(batch_size=batch_size)

    outputs, input_names, elapsed = run_onnxruntime_cpu(str(onnx_path), inputs)
    print(f"ONNX inference: {elapsed*1000:.2f} ms, {batch_size/elapsed:.1f} samples/sec")
    print(f"Output: shape={outputs[0].shape}, min={outputs[0].min():.6f}, max={outputs[0].max():.6f}")

    if compare_with_pytorch:
        from networks.score_wrapper import create_score_network
        from configs.config import get_config

        cfg = get_config()
        checkpoint_path = getattr(cfg, 'pretrained_score_model_path',
                                   './results/ckpts/ScoreNet/scorenet.pth')

        score_net = create_score_network(
            checkpoint_path=checkpoint_path,
            device='cpu'
        )
        score_net.eval()

        torch_inputs = {
            'pts_feat': torch.from_numpy(inputs['pts_feat']),
            'rgb_feat': torch.from_numpy(inputs['rgb_feat']),
            'sampled_pose': torch.from_numpy(inputs['sampled_pose']),
            't': torch.from_numpy(inputs['t']),
        }

        with torch.no_grad():
            pytorch_output = score_net(
                torch_inputs['pts_feat'],
                torch_inputs['rgb_feat'],
                torch_inputs['sampled_pose'],
                torch_inputs['t']
            )

        pytorch_output_np = pytorch_output.cpu().numpy()
        onnx_output = outputs[0]

        max_diff = np.max(np.abs(pytorch_output_np - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output_np - onnx_output))

        print(f"PyTorch vs ONNX: max_diff={max_diff:.10f}, mean_diff={mean_diff:.10f}")

        if max_diff < 1e-5:
            print("✓ Outputs match (diff < 1e-5)")
        elif max_diff < 1e-3:
            print("⚠ Outputs mostly match (diff < 1e-3)")
        else:
            print("✗ Outputs differ significantly!")

        print(f"PyTorch: {pytorch_output_np[0, :5]}")
        print(f"ONNX:    {onnx_output[0, :5]}")


def main():
    parser = argparse.ArgumentParser(description='Test ONNX model with ONNXRuntime')
    parser.add_argument('--onnx_path', type=str,
                        default='./onnx_models/score_network.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--no_compare', action='store_true',
                        help='Skip comparison with PyTorch model')

    args = parser.parse_args()

    test_onnx_model(
        onnx_path=args.onnx_path,
        batch_size=args.batch_size,
        compare_with_pytorch=not args.no_compare
    )


if __name__ == '__main__':
    main()
