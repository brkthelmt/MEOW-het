"""PyTorch dispatcher/FakeTensor entry for the TileLang K1 kernel.

The kernel must be JIT-warmed before ACL Graph capture. Outputs own distinct
allocations, with row padding represented identically in eager and Fake mode.
"""
import torch
from .kernel import pointmap_to_normals as _eager


@torch.library.custom_op("meow_tilelang::pointmap_to_normals", mutates_args=(), device_types="npu")
def pointmap_to_normals(points: torch.Tensor, mask: torch.Tensor, blocks: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    return _eager(points, mask, blocks=blocks)


@pointmap_to_normals.register_fake
def _fake(points, mask, blocks=20):
    b, h, w, _ = points.shape
    npad, vpad = ((3*w+7)//8)*8, ((w+31)//32)*32
    n = torch.empty_strided((b,h,w,3), (h*npad,npad,3,1),
                            device=points.device, dtype=torch.float32)
    v = torch.empty_strided((b,h,w), (h*vpad,vpad,1),
                            device=mask.device, dtype=torch.bool)
    return n, v
