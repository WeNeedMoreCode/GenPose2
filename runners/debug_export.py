"""
Debug ONNX export issues by testing individual components.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from configs.config import get_config
from networks.posenet_agent import PoseNet

def test_component(name, model, dummy_input):
    """Test if a component can be exported to ONNX."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        # Try JIT trace first
        print("  Step 1: Testing JIT trace...")
        traced = torch.jit.trace(model, dummy_input)
        print("  ✓ JIT trace passed")

        # Try ONNX export with minimal settings
        print("  Step 2: Testing ONNX export...")
        torch.onnx.export(
            model,
            dummy_input,
            f"/tmp/test_{name.replace('/', '_')}.onnx",
            opset_version=17,
            verbose=False,
            do_constant_folding=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        print(f"  ✓ {name} ONNX export passed")
        return True
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        return False

def main():
    # Load model
    print("Loading PoseNet model...")
    cfg = get_config()
    cfg.agent_type = 'score'
    cfg.device = 'cpu'
    cfg.dino = 'pointwise'

    checkpoint_path = './results/ckpts/ScoreNet/scorenet.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    agent = PoseNet(cfg)
    agent.load_ckpt(model_dir=checkpoint_path, model_path=True, load_model_only=True)
    net = agent.net
    net.eval()

    B, N, M = 1, 1024, 512

    # Test 1: PointNet2 encoder only
    pts_with_rgb = torch.randn(B, N, 387)
    if not test_component("pts_encoder", net.pts_encoder, pts_with_rgb):
        print("\n❌ PointNet2 encoder failed, stopping here")
        return

    # Test 2: ScoreNet only
    pts_feat = torch.randn(B, N)
    sampled_pose = torch.randn(B, 9)
    t = torch.randn(B, 1)
    data = {'pts_feat': pts_feat, 'rgb_feat': None, 'sampled_pose': sampled_pose, 't': t}

    class ScoreNetWrapper(nn.Module):
        def __init__(self, score_net):
            super().__init__()
            self.score_net = score_net
        def forward(self, pts_feat, sampled_pose, t):
            data = {'pts_feat': pts_feat, 'rgb_feat': None, 'sampled_pose': sampled_pose, 't': t}
            return self.score_net(data)

    score_wrapper = ScoreNetWrapper(net.pose_score_net)
    if not test_component("pose_score_net", score_wrapper, (pts_feat, sampled_pose, t)):
        print("\n❌ ScoreNet failed, stopping here")
        return

    # Test 3: End-to-end
    print(f"\n{'='*60}")
    print("Testing: End-to-End (PointNet2 + ScoreNet)")
    print(f"{'='*60}")

    class EndToEndWrapper(nn.Module):
        def __init__(self, pts_encoder, pose_score_net):
            super().__init__()
            self.pts_encoder = pts_encoder
            self.pose_score_net = pose_score_net
        def forward(self, pts, rgb_feat, sampled_pose, t):
            pts_with_rgb = torch.cat([pts, rgb_feat], dim=-1)
            pts_feat = self.pts_encoder(pts_with_rgb)
            data = {'pts_feat': pts_feat, 'rgb_feat': None, 'sampled_pose': sampled_pose, 't': t}
            return self.pose_score_net(data)

    pts = torch.randn(B, N, 3)
    rgb_feat = torch.randn(B, N, 384)

    e2e_wrapper = EndToEndWrapper(net.pts_encoder, net.pose_score_net)
    test_component("pointnet2_scorenet_endtoend", e2e_wrapper, (pts, rgb_feat, sampled_pose, t))

    print(f"\n{'='*60}")
    print("Debug complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
