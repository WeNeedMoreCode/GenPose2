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


def test_onnx_model(onnx_path: str, batch_size: int = 1):
    onnx_path = Path(onnx_path)
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX: {onnx_path}, {file_size_mb:.2f} MB, batch={batch_size}")

    inputs = create_dummy_inputs(batch_size=batch_size)

    outputs, input_names, elapsed = run_onnxruntime_cpu(str(onnx_path), inputs)
    print(f"ONNX inference: {elapsed*1000:.2f} ms, {batch_size/elapsed:.1f} samples/sec")
    print(f"Output: shape={outputs[0].shape}, min={outputs[0].min():.6f}, max={outputs[0].max():.6f}")


def main():
    parser = argparse.ArgumentParser(description='Test ONNX model with ONNXRuntime')
    parser.add_argument('--onnx_path', type=str,
                        default='./onnx_models/score_network.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for testing')

    args = parser.parse_args()

    test_onnx_model(
        onnx_path=args.onnx_path,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
