"""
Checkpoint Diagnosis Script

Analyzes the checkpoint file to understand why ONNX export
produces a much smaller file than the original checkpoint.

Usage:
    python runners/diagnose_checkpoint.py
"""

import sys
import os
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def diagnose_checkpoint(checkpoint_path):
    """Analyze checkpoint structure and parameter count."""

    print("=" * 70)
    print("Checkpoint Diagnosis Report")
    print("=" * 70)
    print()

    # Check file exists
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    # File size
    file_size = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"1. FILE INFO")
    print(f"   Path: {ckpt_path}")
    print(f"   Size: {file_size:.2f} MB")
    print()

    # Load checkpoint
    print(f"2. LOADING CHECKPOINT...")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"   Loaded successfully")
        print()
    except Exception as e:
        print(f"   Error loading: {e}")
        return

    # Top-level keys
    print(f"3. TOP-LEVEL KEYS")
    top_keys = list(ckpt.keys())
    print(f"   Keys: {top_keys}")
    print()

    # Find model state dict
    model_state = None
    if 'model' in ckpt:
        model_state = ckpt['model']
        print(f"4. MODEL STATE DICT (under 'model' key)")
    elif 'state_dict' in ckpt:
        model_state = ckpt['state_dict']
        print(f"4. MODEL STATE DICT (under 'state_dict' key)")
    elif 'net' in ckpt:
        model_state = ckpt['net']
        print(f"4. MODEL STATE DICT (under 'net' key)")
    else:
        # Check if it's a direct state dict
        if any('score' in k or 'pose' in k for k in top_keys):
            model_state = ckpt
            print(f"4. MODEL STATE DICT (direct state dict)")
        else:
            print(f"4. No model state dict found!")
            return

    # Analyze model structure
    print(f"   Total keys: {len(model_state)}")
    print()

    # Group by module
    print(f"5. MODULE BREAKDOWN")
    modules = {}
    total_params = 0

    for key, value in model_state.items():
        if isinstance(value, torch.Tensor):
            param_count = value.numel()
            total_params += param_count

            # Extract module name
            parts = key.split('.')
            if 'score_agent' in parts:
                module = 'score_agent'
            elif 'pose_score_net' in parts:
                module = 'pose_score_net'
            elif 'pts_encoder' in parts or 'pointnet' in parts.lower():
                module = 'pts_encoder'
            elif 'dino' in parts.lower():
                module = 'dino_encoder'
            elif 'rgb_encoder' in parts:
                module = 'rgb_encoder'
            else:
                module = parts[0] if parts else 'other'

            if module not in modules:
                modules[module] = {'count': 0, 'params': 0, 'keys': []}
            modules[module]['count'] += 1
            modules[module]['params'] += param_count
            modules[module]['keys'].append(key)

    # Print module breakdown
    for module, info in sorted(modules.items(), key=lambda x: -x[1]['params']):
        params_mb = info['params'] * 4 / (1024 * 1024)
        print(f"   {module:25s}: {info['count']:3d} tensors, {info['params']:10,} params ({params_mb:6.2f} MB)")

    print()
    print(f"6. SUMMARY")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Expected size (float32): {total_params * 4 / (1024 * 1024):.2f} MB")
    print(f"   Actual file size: {file_size:.2f} MB")
    print()

    # Expected ONNX size should be close to parameter size
    expected_onnx_mb = total_params * 4 / (1024 * 1024)
    print(f"7. ONNX EXPORT EXPECTATION")
    print(f"   Expected ONNX size: ~{expected_onnx_mb:.2f} MB")
    print(f"   Your ONNX export: 4.5 MB")
    print()

    if expected_onnx_mb > 50:
        print("   WARNING: Expected ONNX size is much larger than 4.5 MB!")
        print("   This means the ONNX export is MISSING most of the model.")
        print()
        print("   Possible causes:")
        print("   1. ScoreNetworkWrapper not loading full checkpoint")
        print("   2. Only pose_score_net exported, missing encoders")
        print("   3. load_ckpt() not finding matching keys")
        print()
        print("   Recommend checking:")
        print("   - Does ScoreNetworkWrapper load pts_encoder?")
        print("   - Does ScoreNetworkWrapper load rgb_encoder?")

    print()
    print("=" * 70)

    # Show largest layers
    print(f"8. LARGEST LAYERS (for debugging)")
    all_layers = []
    for key, value in model_state.items():
        if isinstance(value, torch.Tensor):
            all_layers.append((key, value.numel(), list(value.shape)))

    all_layers.sort(key=lambda x: -x[1])
    for key, count, shape in all_layers[:10]:
        print(f"   {key:50s}: {shape} = {count:,} params")

    print()
    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose checkpoint file structure')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./results/ckpts/ScoreNet/scorenet.pth',
                        help='Path to checkpoint file')

    args = parser.parse_args()

    diagnose_checkpoint(args.checkpoint_path)


if __name__ == '__main__':
    main()
