"""
OM Backend for GenPose2

Provides a wrapper around ais_bench InferSession that mimics nn.Module interface.
This allows seamless switching between PyTorch and OM models.

Usage:
    # In posenet_agent.py:
    if checkpoint_path.endswith('.om'):
        self.net = OMInferSession(om_model_path, metadata_path)
    else:
        self.net = GFObjectPose(...)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn


class OMInferSession(nn.Module):
    """
    Wrapper for ais_bench InferSession that provides nn.Module-like interface.

    This allows OM models to be used as drop-in replacements for PyTorch models.
    """

    def __init__(self, om_model_path: str, metadata_path: Optional[str] = None, device_id: int = 0):
        """
        Initialize OM inference session.

        Args:
            om_model_path: Path to .om model file
            metadata_path: Path to metadata JSON file with input/output info
            device_id: NPU device ID (default: 0)
        """
        super().__init__()

        self.om_model_path = Path(om_model_path)
        self.device_id = device_id

        if not self.om_model_path.exists():
            raise FileNotFoundError(f"OM model not found: {om_model_path}")

        # Load metadata
        self.metadata = {}
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}

        if metadata_path:
            self._load_metadata(metadata_path)
        else:
            # Try to find metadata file next to OM model
            metadata_guess = self.om_model_path.with_suffix('.json')
            if metadata_guess.exists():
                self._load_metadata(str(metadata_guess))
            else:
                # Use default naming convention
                self.input_names = ['input']
                self.output_names = ['output']

        # Load OM model
        try:
            from ais_bench.infer.interface import InferSession
        except ImportError:
            raise ImportError(
                "ais_bench is required for OM inference. "
                "Install with: pip install ais_bench"
            )

        print(f"Loading OM model: {om_model_path}")
        self.session = InferSession(self.device_id, str(self.om_model_path))
        print(f"✓ OM model loaded successfully")

        # Store device for compatibility
        self.device = torch.device(f'npu:{device_id}')

    def _load_metadata(self, metadata_path: str):
        """Load input/output metadata from JSON file."""
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        if 'inputs' in self.metadata:
            self.input_names = [inp['name'] for inp in self.metadata['inputs']]
            self.input_shapes = {inp['name']: inp['shape'] for inp in self.metadata['inputs']}
        if 'outputs' in self.metadata:
            self.output_names = [out['name'] for out in self.metadata['outputs']]
            self.output_shapes = {out['name']: out['shape'] for out in self.metadata['outputs']}

    def forward(self, data: Dict[str, torch.Tensor], mode: str = 'default', **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass - mimics GFObjectPose.forward() interface.

        Args:
            data: Dict with input tensors
            mode: Forward mode ('score', 'energy', 'pts_feature', 'rgb_feature', etc.)
            **kwargs: Additional arguments (T0, init_x, etc.)

        Returns:
            Output tensor(s) as torch.Tensor(s) on CPU
        """
        # Prepare inputs for OM model
        om_inputs = self._prepare_inputs(data, mode)

        # Run inference
        outputs = self.session.infer(om_inputs)

        # Convert outputs back to torch tensors
        return self._process_outputs(outputs, mode, data)

    def _prepare_inputs(self, data: Dict[str, torch.Tensor], mode: str) -> List[np.ndarray]:
        """Convert torch tensors to numpy arrays for OM inference."""
        inputs = []

        if mode == 'pts_feature':
            # Extract point cloud features
            pts = data['pts'].cpu().numpy().astype(np.float32)
            roi_rgb = data.get('roi_rgb')
            if roi_rgb is not None:
                roi_rgb = roi_rgb.cpu().numpy().astype(np.float32)
            # For OM, we need specific inputs based on model export format
            inputs = [pts]
            if roi_rgb is not None:
                inputs.append(roi_rgb)

        elif mode == 'rgb_feature':
            # RGB features are handled differently - might need separate model
            # For now, return dummy or skip
            return [np.zeros((1, 384), dtype=np.float32)]

        elif mode == 'ode_sample' or mode == 'score_sample':
            # Score sampling with ODE solver - this is complex for OM
            # For now, run the score network directly
            inputs = self._prepare_score_inputs(data)

        elif mode == 'energy':
            # Energy computation
            inputs = self._prepare_energy_inputs(data)

        else:
            # Default: use all inputs from metadata
            for name in self.input_names:
                if name in data:
                    tensor = data[name]
                    if tensor.dtype in [torch.float32, torch.float64]:
                        inputs.append(tensor.cpu().numpy().astype(np.float32))
                    elif tensor.dtype in [torch.int64, torch.int32]:
                        inputs.append(tensor.cpu().numpy().astype(np.int64))
                    else:
                        inputs.append(tensor.cpu().numpy())

        return inputs

    def _prepare_score_inputs(self, data: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Prepare inputs for score network."""
        inputs = []
        if 'pts' in data:
            inputs.append(data['pts'].cpu().numpy().astype(np.float32))
        if 'roi_rgb' in data and data['roi_rgb'] is not None:
            inputs.append(data['roi_rgb'].cpu().numpy().astype(np.float32))
        return inputs

    def _prepare_energy_inputs(self, data: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Prepare inputs for energy network."""
        inputs = []
        if 'pts' in data:
            inputs.append(data['pts'].cpu().numpy().astype(np.float32))
        if 'roi_rgb' in data and data['roi_rgb'] is not None:
            inputs.append(data['roi_rgb'].cpu().numpy().astype(np.float32))
        if 'sampled_pose' in data:
            inputs.append(data['sampled_pose'].cpu().numpy().astype(np.float32))
        return inputs

    def _process_outputs(self, outputs: List[np.ndarray], mode: str, input_data: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert numpy outputs back to torch tensors."""
        torch_outputs = []

        for output in outputs:
            torch_outputs.append(torch.from_numpy(output))

        # Return based on expected output format for each mode
        if len(torch_outputs) == 1:
            return torch_outputs[0]
        else:
            return torch_outputs

    def __call__(self, *args, **kwargs):
        """Allow direct calling like model(data)."""
        if len(args) == 1 and isinstance(args[0], dict):
            return self.forward(args[0], **kwargs)
        return super().__call__(*args, **kwargs)


def create_om_backend(om_model_path: str, metadata_path: Optional[str] = None, device_id: int = 0) -> OMInferSession:
    """
    Factory function to create OM backend.

    Args:
        om_model_path: Path to .om model file
        metadata_path: Path to metadata JSON file
        device_id: NPU device ID

    Returns:
        OMInferSession instance
    """
    return OMInferSession(om_model_path, metadata_path, device_id)


def is_om_model(model_path: str) -> bool:
    """Check if model path is an OM model."""
    return Path(model_path).suffix.lower() == '.om'
