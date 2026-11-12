"""Opaque dispatcher boundaries preserving NPU eager numerical semantics.

These are capture-compatible PyTorch operators, not new device kernels. Keep
bicubic opaque because PyTorch's AOT decomposition differs from the installed
NPU eager implementation for fractional scale factors used by DINOv2.
"""
import torch
import torch.nn.functional as F


@torch.library.custom_op('meow_graph::bicubic', mutates_args=())
def bicubic(x: torch.Tensor, output_h: int, output_w: int,
            scale_h: float, scale_w: float, antialias: bool) -> torch.Tensor:
    kwargs = ({'scale_factor': (scale_h, scale_w)} if scale_h > 0
              else {'size': (output_h, output_w)})
    return F.interpolate(x, mode='bicubic', antialias=antialias, **kwargs).contiguous()


@bicubic.register_fake
def _bicubic_fake(x, output_h, output_w, scale_h, scale_w, antialias):
    return x.new_empty((x.shape[0], x.shape[1], output_h, output_w))


@torch.library.custom_op('meow_graph::bilinear', mutates_args=())
def bilinear(x: torch.Tensor, output_h: int, output_w: int,
             align_corners: bool, scale_h: float = -1.0,
             scale_w: float = -1.0) -> torch.Tensor:
    kwargs = ({'scale_factor': (scale_h, scale_w)} if scale_h > 0
              else {'size': (output_h, output_w)})
    return F.interpolate(x, mode='bilinear', align_corners=align_corners,
                         **kwargs).contiguous()


@bilinear.register_fake
def _bilinear_fake(x, output_h, output_w, align_corners, scale_h=-1.0, scale_w=-1.0):
    return x.new_empty((x.shape[0], x.shape[1], output_h, output_w))
