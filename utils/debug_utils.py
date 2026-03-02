"""
调试工具模块 - 用于对比NPU和GPU版本
在评估流程的关键位置输出统计信息
"""

import torch
import os

# 控制调试输出的开关
ENABLE_DEBUG = os.environ.get('GENPOSE_DEBUG', '1') == '1'

def debug_print(stage, name, tensor, extra_info=None):
    """在指定阶段输出tensor的统计信息"""
    if not ENABLE_DEBUG:
        return

    if tensor is None:
        print(f"[DEBUG {stage}] {name}: None")
        return

    if isinstance(tensor, (list, tuple)):
        print(f"[DEBUG {stage}] {name}: list/tuple with {len(tensor)} elements")
        return

    # 获取基本统计信息
    info = {
        'stage': stage,
        'name': name,
        'dtype': str(tensor.dtype),
        'shape': list(tensor.shape),
        'device': str(tensor.device),
        'min': float(tensor.min()),
        'max': float(tensor.max()),
        'mean': float(tensor.mean()),
        'std': float(tensor.std()),
        'has_nan': bool(torch.isnan(tensor).any()),
        'has_inf': bool(torch.isinf(tensor).any()),
    }

    # 格式化输出
    print(f"[DEBUG {stage}] {name}:")
    print(f"  dtype={info['dtype']}, shape={info['shape']}, device={info['device']}")
    print(f"  min={info['min']:.6f}, max={info['max']:.6f}, mean={info['mean']:.6f}, std={info['std']:.6f}")
    if info['has_nan']:
        print(f"  WARNING: Contains NaN!")
    if info['has_inf']:
        print(f"  WARNING: Contains Inf!")
    if extra_info is not None:
        print(f"  extra: {extra_info}")


def debug_print_batch(stage, data_dict, keys=None):
    """批量输出dict中多个tensor的调试信息"""
    if not ENABLE_DEBUG:
        return

    if keys is None:
        keys = list(data_dict.keys())

    print(f"\n{'='*60}")
    print(f"[DEBUG BATCH] Stage: {stage}")
    for key in keys:
        if key in data_dict:
            debug_print(stage, key, data_dict[key])
    print(f"{'='*60}\n")
