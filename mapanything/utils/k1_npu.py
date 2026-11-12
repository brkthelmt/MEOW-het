"""Explicit K1 native entry point; loading never changes the default pipeline."""
from pathlib import Path

import torch


def load_k1_library(path):
    """Load a trusted library registering meow_k1::pointmap_to_normals."""
    library = Path(path).expanduser().resolve(strict=True)
    torch.ops.load_library(str(library))
    if not hasattr(torch.ops.meow_k1, 'pointmap_to_normals'):
        raise RuntimeError(f'{library} does not register meow_k1::pointmap_to_normals')


def pointmap_to_normals(points, mask, *, route="queue", tile=128, blocks=32):
    """Call the registered inference-only NPU K1 operator explicitly."""
    if route not in ("queue", "manual"):
        raise ValueError("route must be queue or manual")
    if tile not in (64, 128, 256) or not 1 <= blocks <= 64:
        raise ValueError("tile must be 64/128/256; blocks must be 1..64")
    if points.device.type != 'npu' or mask.device != points.device:
        raise ValueError('K1 requires points and mask on the same NPU')
    if points.dtype != torch.float32 or mask.dtype != torch.bool:
        raise TypeError('K1 requires FP32 points and bool mask')
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[:-1] != mask.shape:
        raise ValueError('Expected points [B,H,W,3], mask [B,H,W]')
    if points.shape[1] < 1 or points.shape[2] < 1:
        raise ValueError('K1 requires H,W > 0')
    if not points.is_contiguous() or not mask.is_contiguous():
        raise ValueError('K1 requires contiguous inputs; convert explicitly')
    if torch.is_grad_enabled() and points.requires_grad:
        raise RuntimeError('K1 has no backward implementation; use inference_mode')
    if not hasattr(torch.ops.meow_k1, 'pointmap_to_normals'):
        raise RuntimeError('K1 native operator is not loaded; call load_k1_library first')
    return torch.ops.meow_k1.pointmap_to_normals(points, mask, route == "manual", tile, blocks)
