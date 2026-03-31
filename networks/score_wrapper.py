"""
Score Network Wrapper for ODE Sampling Decoupling

This module provides a clean interface for using the Score Network
outside of the integrated ODE sampler, enabling:
1. ONNX export of Score Network only
2. External ODE sampling implementation (PyTorch or C++)
3. Call Score Network as a standalone function
4. Support for both PyTorch (.pth) and OM (.om) models
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from configs.config import get_config
from networks.posenet_agent import PoseNet


class ScoreNetworkWrapper(nn.Module):
    """
    Wrapper for PoseNet that exposes only the Score Network forward pass.

    This allows the Score Network to be:
    - Exported to ONNX independently (ScoreNet only)
    - Called from external ODE sampling loops
    - Used with different sampling strategies
    - Loaded from either PyTorch (.pth) or OM (.om) models

    PyTorch Mode Input:
        pts_feat: [batch_size, 1024] - Point cloud features from PointNet2
        rgb_feat: [batch_size, 384] - RGB features from DINOv2
        sampled_pose: [batch_size, 9] - Current pose estimate
        t: [batch_size, 1] - Timestep

    OM Mode Input (end-to-end with PointNet2):
        pts: [batch_size, 1024, 3] - Raw point cloud
        rgb_feat: [batch_size, 1024, 384] - DINOv2 features
        sampled_pose: [batch_size, 9] - Current pose estimate
        t: [batch_size, 1] - Timestep

    Output:
        score: [batch_size, 9] - Score/gradient for the given pose
    """

    def __init__(self, checkpoint_path, device='npu:0', pointnet2_om_path=None):
        """
        Initialize ScoreNetworkWrapper with a trained checkpoint.

        Args:
            checkpoint_path: Path to ScoreNet checkpoint (.pth or .om)
            device: Device to load model on
            pointnet2_om_path: Optional path to PointNet2 OM model for unified pts_feat extraction
        """
        super().__init__()

        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.is_om = self.checkpoint_path.suffix.lower() == '.om'
        self.pointnet2_om = None  # Will be loaded if pointnet2_om_path provided

        if self.is_om:
            self._load_om_model()
        else:
            self._load_pytorch_model()

        # Optionally load PointNet2 OM for unified pts_feat extraction
        if pointnet2_om_path is not None:
            self.pointnet2_om = create_pointnet2_encoder(pointnet2_om_path, device)

    def _load_pytorch_model(self):
        """Load PyTorch model from .pth checkpoint."""
        # Load config and model
        cfg = get_config()
        cfg.agent_type = 'score'
        cfg.device = self.device
        cfg.dino = 'pointwise'  # Enable DINOv2 (must match checkpoint training mode)

        # Load ScoreNet
        self.score_agent = PoseNet(cfg)
        self.score_agent.load_ckpt(
            model_dir=str(self.checkpoint_path),
            model_path=True,
            load_model_only=True
        )
        self.score_agent.eval()

        # Store references
        self.net = self.score_agent.net
        self.pts_encoder = self.net.pts_encoder  # PointNet2
        self.pose_score_net = self.net.pose_score_net  # ScoreNet
        self.cfg = cfg

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad_(False)

    def _load_om_model(self):
        """Load OM model using ais_bench InferSession.

        Note: OM model should contain PointNet2 + ScoreNet end-to-end.
        Input: pts, rgb_feat, sampled_pose, t
        Output: score
        """
        try:
            from ais_bench.infer.interface import InferSession
        except ImportError:
            raise ImportError(
                "OM model requires 'ais_bench' package. "
                "Install with: pip install ais_bench"
            )

        # Determine device ID
        if isinstance(self.device, str) and 'npu:' in self.device:
            device_id = int(self.device.split(':')[1])
        else:
            device_id = 0

        # Load metadata
        metadata_path = metadata_path = self.checkpoint_path.parent / f"{self.checkpoint_path.stem}_metadata.json"
        if not metadata_path.exists():
            metadata_path = self.checkpoint_path.parent / 'score_network_metadata.json'

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Loaded metadata from: {metadata_path}")
        else:
            print(f"Warning: Metadata not found, using defaults")
            self.metadata = None

        # Load OM model
        print(f"Loading OM model: {self.checkpoint_path}")
        self.om_session = InferSession(device_id, str(self.checkpoint_path))
        print(f"✓ OM model loaded successfully")

        # Store dummy cfg for compatibility
        self.cfg = get_config()
        self.cfg.pose_mode = 'rot_matrix'

    def forward(self, pts_feat, rgb_feat, sampled_pose, t):
        """
        Forward pass of Score Network.

        PyTorch Mode:
            Args:
                pts_feat: [batch_size, 1024] - Point cloud features from PointNet2
                rgb_feat: [batch_size, 384] - RGB features from DINOv2 (may be None)
                sampled_pose: [batch_size, 9] - Current pose estimate
                t: [batch_size, 1] - Diffusion timestep
            Returns:
                score: [batch_size, 9]

        OM Mode (end-to-end):
            Args:
                pts_feat: [batch_size, 1024, 3] - Raw point cloud
                rgb_feat: [batch_size, 1024, 384] - DINOv2 features (may be unused)
                sampled_pose: [batch_size, 9] - Current pose estimate
                t: [batch_size, 1] - Diffusion timestep
            Returns:
                score: [batch_size, 9]
        """
        if self.is_om:
            # OM model inference (end-to-end: PointNet2 + ScoreNet)
            # Convert torch tensors to numpy
            inputs = [
                pts_feat.cpu().numpy().astype(np.float32),  # Actually raw pts
            ]

            # Check if rgb_feat is used (based on metadata)
            use_rgb_feat = True
            if hasattr(self, 'metadata') and self.metadata is not None:
                use_rgb_feat = self.metadata.get('metadata', {}).get('use_rgb_feat', True)

            if use_rgb_feat and rgb_feat is not None:
                inputs.append(rgb_feat.cpu().numpy().astype(np.float32))

            inputs.extend([
                sampled_pose.cpu().numpy().astype(np.float32),
                t.cpu().numpy().astype(np.float32)
            ])

            # Run OM inference
            outputs = self.om_session.infer(inputs, mode="dymbatch")

            # Convert back to torch tensor
            if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
                score = torch.from_numpy(outputs[0])
            else:
                score = torch.from_numpy(outputs)

            return score.to(self.device)

        else:
            # PyTorch model inference (ScoreNet only, pts_feat already extracted)
            # Prepare data dict (matching PoseScoreNet.forward format)
            data = {
                'pts_feat': pts_feat,
                'rgb_feat': rgb_feat,
                'sampled_pose': sampled_pose,
                't': t
            }

            # Call score network
            with torch.no_grad():
                score = self.pose_score_net(data)

            return score

    def get_score(self, data):
        """
        Alternative interface that accepts data dict (for compatibility).

        Args:
            data: Dict with keys 'pts_feat', 'rgb_feat', 'sampled_pose', 't'

        Returns:
            score: Score/gradient
        """
        return self.forward(
            data['pts_feat'],
            data['rgb_feat'],
            data['sampled_pose'],
            data['t']
        )

    def extract_pts_feat(self, pts, rgb_feat):
        """
        Extract point cloud features using PointNet2 encoder.

        This method provides a unified interface that internally uses:
        - PyTorch PointNet2 (if self.is_om=False or no OM model loaded)
        - OM PointNet2 (if pointnet2_om is available)

        Args:
            pts: [batch_size, 1024, 3] - Point cloud coordinates
            rgb_feat: [batch_size, 1024, 384] - DINOv2 features

        Returns:
            pts_feat: [batch_size, 1024] - Encoded point cloud features
        """
        if self.pointnet2_om is not None:
            # Use OM PointNet2 encoder
            return self.pointnet2_om(pts, rgb_feat)
        else:
            # Use PyTorch PointNet2 encoder
            with torch.no_grad():
                pointcloud = torch.cat([pts, rgb_feat], dim=-1)
                return self.pts_encoder(pointcloud)


class ODESamplerExternal:
    """
    External ODE sampler that calls ScoreNetworkWrapper.

    This maintains the same ODE sampling logic as cond_ode_sampler
    but allows the Score Network to be provided externally (e.g., ONNX model).

    Usage:
        # Create score network (PyTorch or ONNX)
        score_net = ScoreNetworkWrapper(checkpoint_path)

        # Create ODE sampler
        sampler = ODESamplerExternal(score_net)

        # Run sampling
        final_pose, trajectory = sampler.sample(
            pts_feat=pts_feat,
            rgb_feat=rgb_feat,
            T0=0.55,
            rtol=1e-5,
            atol=1e-5
        )
    """

    def __init__(self, score_network, prior_fn, sde_coeff, device='npu:0'):
        """
        Initialize ODE sampler.

        Args:
            score_network: ScoreNetworkWrapper instance (or compatible callable)
            prior_fn: Prior sampling function (from SDE)
            sde_coeff: SDE coefficient function (from SDE)
            device: Device to run on
        """
        self.score_network = score_network
        self.prior_fn = prior_fn
        self.sde_coeff = sde_coeff
        self.device = device

    def score_eval_wrapper(self, data):
        """
        Wrapper for score network that matches the interface expected by ode_func.

        Args:
            data: Dict with keys 'pts_feat', 'rgb_feat', 'sampled_pose', 't'

        Returns:
            score: Score as numpy array
        """
        with torch.no_grad():
            if hasattr(self.score_network, 'get_score'):
                score = self.score_network.get_score(data)
            else:
                score = self.score_network(
                    data['pts_feat'],
                    data['rgb_feat'],
                    data['sampled_pose'],
                    data['t']
                )
        return score.cpu().numpy().reshape((-1,))

    def sample(self, pts_feat, rgb_feat, batch_size, pose_dim,
               eps=1e-5, T=1.0, rtol=1e-5, atol=1e-5, denoise=True, init_x=None, pts_center=None):
        """
        Run ODE sampling using SciPy RK45 solver.

        Args:
            pts_feat: [batch_size, 1024] - Point cloud features
            rgb_feat: [batch_size, 384] - RGB features
            batch_size: Batch size
            pose_dim: Pose dimension (e.g., 7 for quat_wxyz)
            eps: End time (default 1e-5)
            T: Start time (default 1.0, use 0.55 for faster inference)
            rtol: Relative tolerance
            atol: Absolute tolerance
            denoise: Whether to apply denoising step
            init_x: Initial pose (optional)

        Returns:
            trajectory: [num_steps, batch_size, pose_dim] - Sampling trajectory
            final_pose: [batch_size, pose_dim] - Final sampled pose
        """
        import numpy as np
        from scipy import integrate

        # Initialize
        # If init_x is provided, use it directly (don't add noise since we pre-generated all noise)
        # Otherwise generate random noise
        init_x = self.prior_fn((batch_size, pose_dim), T=T).to(self.device) if init_x is None else init_x

        shape = init_x.shape
        data = {
            'pts_feat': pts_feat,
            'rgb_feat': rgb_feat,
        }

        def ode_func(t, x):
            """ODE function for use by the ODE solver."""
            x_tensor = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=self.device)
            time_steps = torch.ones(batch_size, device=self.device).unsqueeze(-1) * t
            drift, diffusion = self.sde_coeff(torch.tensor(t))
            drift = drift.cpu().numpy()
            diffusion = diffusion.cpu().numpy()

            data['sampled_pose'] = x_tensor
            data['t'] = time_steps

            score = self.score_eval_wrapper(data)
            return drift - 0.5 * (diffusion**2) * score

        # Run ODE solver
        res = integrate.solve_ivp(
            ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(),
            rtol=rtol, atol=atol, method='RK45'
        )

        # Extract results
        xs = torch.tensor(res.y, device=self.device, dtype=torch.float32).T.view(-1, batch_size, pose_dim)
        x = torch.tensor(res.y[:, -1], device=self.device, dtype=torch.float32).reshape(shape)

        # Denoising step (if requested)
        if denoise:
            # Reverse diffusion predictor for denoising (same as original cond_ode_sampler:221)
            vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
            drift, diffusion = self.sde_coeff(vec_eps)
            data['sampled_pose'] = x.float()
            data['t'] = vec_eps
            grad = self.score_network.get_score(data)  # Returns tensor, not numpy
            drift = drift - diffusion**2 * grad
            mean_x = x + drift * ((1 - eps) / 1000)
            x = mean_x

        # Normalize rotation (same as original cond_ode_sampler:226-232)
        from utils.misc import normalize_rotation
        pose_mode = self.score_network.cfg.pose_mode

        num_steps = xs.shape[0]
        xs = xs.reshape(batch_size * num_steps, -1)
        xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
        xs = xs.reshape(num_steps, batch_size, -1)
        if pts_center is not None:
            xs[:, :, -3:] += pts_center.unsqueeze(0).repeat(xs.shape[0], 1, 1)

        x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
        if pts_center is not None:
            x[:, -3:] += pts_center

        return xs.permute(1, 0, 2), x


def create_score_network(checkpoint_path, device='npu:0', pointnet2_om_path=None):
    """
    Factory function to create ScoreNetworkWrapper.

    Args:
        checkpoint_path: Path to ScoreNet checkpoint
        device: Device to load model on
        pointnet2_om_path: Optional path to PointNet2 OM model for unified pts_feat extraction

    Returns:
        ScoreNetworkWrapper instance
    """
    return ScoreNetworkWrapper(checkpoint_path, device, pointnet2_om_path)


def create_ode_sampler(score_network, sde, device='npu:0'):
    """
    Factory function to create ODE sampler with score network.

    Args:
        score_network: ScoreNetworkWrapper instance
        sde: SDE object or dict containing prior_fn and sde_fn/sde_coeff
        device: Device to run on

    Returns:
        ODESamplerExternal instance
    """
    # Handle both dict and object inputs
    if isinstance(sde, dict):
        prior_fn = sde['prior_fn']
        sde_coeff = sde.get('sde_coeff', sde.get('sde_fn'))
    else:
        prior_fn = sde.prior_fn
        sde_coeff = sde.sde_fn if hasattr(sde, 'sde_fn') else sde.sde_coeff

    return ODESamplerExternal(
        score_network=score_network,
        prior_fn=prior_fn,
        sde_coeff=sde_coeff,
        device=device
    )


class PointNet2EncoderWrapper(nn.Module):
    """
    Wrapper for PointNet2 encoder OM model.

    This wrapper loads a PointNet2 encoder exported to OM format
    and provides a simple forward interface.

    Args:
        checkpoint_path: Path to OM model (.om file)
        device: Device to run inference on (e.g., 'npu:0')
    """

    def __init__(self, checkpoint_path, device='npu:0'):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.is_om = self.checkpoint_path.suffix.lower() == '.om'

        if not self.is_om:
            raise ValueError(f"PointNet2EncoderWrapper only supports OM models, got {checkpoint_path}")

        # Load metadata
        metadata_path = self.checkpoint_path.parent / f"{self.checkpoint_path.stem}_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Loaded PointNet2 metadata from: {metadata_path}")
        else:
            print(f"Warning: PointNet2 metadata not found")
            self.metadata = None

        self._load_om_model()

    def _load_om_model(self):
        """Load PointNet2 OM model using ais_bench InferSession."""
        try:
            from ais_bench.infer.interface import InferSession
        except ImportError:
            raise ImportError(
                "OM model requires 'ais_bench' package. "
                "Install with: pip install ais_bench"
            )

        # Determine device ID
        if isinstance(self.device, str) and 'npu:' in self.device:
            device_id = int(self.device.split(':')[1])
        else:
            device_id = 0

        # Load OM model
        print(f"Loading PointNet2 OM model: {self.checkpoint_path}")
        self.om_session = InferSession(device_id, str(self.checkpoint_path))
        print(f"✓ PointNet2 OM model loaded successfully")

    def forward(self, pts, rgb_feat):
        """
        Forward pass of PointNet2 encoder.

        Args:
            pts: [batch_size, 1024, 3] - Point cloud coordinates
            rgb_feat: [batch_size, 1024, 384] - DINOv2 features

        Returns:
            pts_feat: [batch_size, 1024] - Encoded point cloud features
        """
        # Convert torch tensors to numpy
        inputs = [
            pts.cpu().numpy().astype(np.float32),
            rgb_feat.cpu().numpy().astype(np.float32),
        ]

        # Run OM inference
        outputs = self.om_session.infer(inputs, mode="dymbatch")

        # Convert back to torch tensor
        if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
            pts_feat = torch.from_numpy(outputs[0])
        else:
            pts_feat = torch.from_numpy(outputs)

        return pts_feat.to(self.device)


def create_pointnet2_encoder(checkpoint_path, device='npu:0'):
    """
    Factory function to create PointNet2EncoderWrapper.

    Args:
        checkpoint_path: Path to PointNet2 OM checkpoint
        device: Device to load model on

    Returns:
        PointNet2EncoderWrapper instance
    """
    return PointNet2EncoderWrapper(checkpoint_path, device)
