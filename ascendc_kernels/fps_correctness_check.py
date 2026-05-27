"""
FPS correctness test with real point cloud data.
Tests 4 implementations against each other.
"""
import sys
import os
import torch
import torch_npu  # noqa
torch_npu.npu.set_compile_mode(jit_compile=False)

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "GenPosePlus"))
sys.path.insert(0, os.path.join(project_root, "GenPose2"))
sys.path.insert(0, os.path.join(project_root, "GenPose2", "fps_kernellaunch"))

# Load real point cloud data
data_path = os.path.join(project_root, "GenPose2", "test_group_data.pth")
if not os.path.exists(data_path):
    data_path = "./test_group_data.pth"
test_data = torch.load(data_path)
xyz = test_data["xyz"].to('npu:0')  # [B, N, 3]

B, N, C = xyz.shape
npoints = 512

print(f"=== FPS Correctness Test (real data) ===")
print(f"xyz shape: {xyz.shape}, dtype: {xyz.dtype}, device: {xyz.device}")
print(f"npoints: {npoints}\n")

# --- 1. Project call chain: pointnet2_utils.furthest_point_sample ---
from networks.pts_encoder.pointnet2_utils.pointnet2 import pointnet2_utils as pn2_utils

idx1 = pn2_utils.furthest_point_sample(xyz, npoints)
print(f"1. pointnet2_utils.furthest_point_sample:")
print(f"   [:10] = {idx1[0, :10].tolist()}")

# --- 2. Direct call: pointnet2_ops._furthest_point_sampling ---
import pointnet2_ops

idx2 = pointnet2_ops._furthest_point_sampling(xyz, npoints)
print(f"\n2. pointnet2_ops._furthest_point_sampling:")
print(f"   [:10] = {idx2[0, :10].tolist()}")

# --- 3. AscendC 1 core ---
from fps_wrapper import FurthestPointSamplingAscendC

fps_1c = FurthestPointSamplingAscendC()
idx3 = fps_1c(xyz, npoints)
print(f"\n3. AscendC (1 core):")
print(f"   [:10] = {idx3[0, :10].tolist()}")

# --- 4. AscendC 8 cores ---
from fps_wrapper import FurthestPointSamplingMultiCore

fps_8c = FurthestPointSamplingMultiCore()
idx4 = fps_8c(xyz, npoints)
print(f"\n4. AscendC (8 cores):")
print(f"   [:10] = {idx4[0, :10].tolist()}")

# --- Compare ---
print(f"\n=== Comparison ===")
pairs = [
    ("1 vs 2 (pn2_utils vs pn2_ops)", idx1, idx2),
    ("1 vs 3 (pn2_utils vs Ascend1c)", idx1, idx3),
    ("1 vs 4 (pn2_utils vs Ascend8c)", idx1, idx4),
    ("2 vs 3 (pn2_ops vs Ascend1c)", idx2, idx3),
    ("2 vs 4 (pn2_ops vs Ascend8c)", idx2, idx4),
    ("3 vs 4 (Ascend1c vs Ascend8c)", idx3, idx4),
]

for name, a, b in pairs:
    a_i64 = a.long()
    b_i64 = b.long()
    match = torch.equal(a_i64, b_i64)
    diff = (a_i64 != b_i64).sum().item()
    print(f"  {name}: {'MATCH' if match else f'MISMATCH ({diff}/{npoints})'}")
