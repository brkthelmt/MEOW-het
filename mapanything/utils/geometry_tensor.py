"""Tensor equivalents of the NumPy inference normal/edge filters.

Inputs are batched BHWC points/normals and BHW depth/masks. Keep geometry in
FP32 and preserve the reference's padding, additive epsilon, NaN reductions,
and depth-edge AND normal-edge rule. No host reads or custom kernels.
"""
import math

import torch
import torch.nn.functional as F


def _length3(x):
    # Match the order of NumPy's reduction of three float32 components.
    return torch.sqrt((x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1]) + x[..., 2] * x[..., 2])


def _unit3(x):
    return x / (_length3(x).unsqueeze(-1) + 1e-12)


def _cross3(a, b):
    return torch.stack((a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]), dim=-1)


def _nanmax_pool3(x):
    """3x3 stride-one nanmax, ignoring out-of-image padding like NumPy."""
    present = ~torch.isnan(x)
    value = torch.where(present, x, -torch.inf)
    maximum = F.max_pool2d(value.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    any_present = F.max_pool2d(present.float().unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0
    return torch.where(any_present, maximum, torch.nan)


def points_to_normals_tensor(points, mask):
    """Four valid adjacent triangles per pixel; zero/invalid exterior padding."""
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[:-1] != mask.shape:
        raise ValueError('Expected points [B,H,W,3] and mask [B,H,W]')
    points = points.float()
    padded = F.pad(points, (0, 0, 1, 1, 1, 1))
    valid = F.pad(mask, (1, 1, 1, 1), value=False)
    directions = [padded[:, :-2, 1:-1] - points,
                  padded[:, 1:-1, :-2] - points,
                  padded[:, 2:, 1:-1] - points,
                  padded[:, 1:-1, 2:] - points]
    neighbors = [valid[:, :-2, 1:-1], valid[:, 1:-1, :-2],
                 valid[:, 2:, 1:-1], valid[:, 1:-1, 2:]]
    normals, triangles = [], []
    for i in range(4):
        triangle = mask & neighbors[i] & neighbors[(i + 1) % 4]
        # Multiplication is intentional: NaN * False is NaN in the reference.
        normals.append(_unit3(_cross3(directions[i], directions[(i + 1) % 4])) * triangle.unsqueeze(-1))
        triangles.append(triangle)
    normal = _unit3(((normals[0] + normals[1]) + normals[2]) + normals[3])
    normal_mask = triangles[0] | triangles[1] | triangles[2] | triangles[3]
    return torch.where(normal_mask.unsqueeze(-1), normal, 0), normal_mask


def normals_edge_tensor(normals, mask, tol=5.0):
    """Replicate-border 3x3 angle maximum, then 3x3 nanmax dilation."""
    normals = _unit3(normals.float())
    h, w = normals.shape[1:3]
    padded = F.pad(normals.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='replicate').permute(0, 2, 3, 1)
    padded_mask = F.pad(mask.float().unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1) > 0
    angles = []
    for y in range(3):
        for x in range(3):
            neighbor = padded[:, y:y+h, x:x+w]
            dot = ((normals[..., 0] * neighbor[..., 0] + normals[..., 1] * neighbor[..., 1])
                   + normals[..., 2] * neighbor[..., 2])
            # Do not clamp: the NumPy reference propagates out-of-domain NaNs.
            # The reference calls sliding_window_2d(mask, axis=(-3,-2)) on
            # a 2-D mask: modulo indexing orders its windows W,H, while the
            # normal windows are H,W. Preserve that transposed mask window.
            angles.append(torch.where(padded_mask[:, x:x+h, y:y+w], torch.acos(dot), 0))
    angles = torch.stack(angles)
    nan = torch.isnan(angles)
    maximum = torch.where(nan, -torch.inf, angles).amax(dim=0)
    maximum = torch.where(nan.any(dim=0), torch.nan, maximum)
    return _nanmax_pool3(maximum) > math.radians(tol)


def depth_edge_tensor(depth, mask, rtol=0.03):
    """Reference radial depth range / center depth; no clamp or abs."""
    depth = depth.float()
    maximum = _nanmax_pool3(torch.where(mask, depth, -torch.inf))
    minus_minimum = _nanmax_pool3(torch.where(mask, -depth, -torch.inf))
    return (maximum + minus_minimum) / depth > rtol


def geometry_mask_tensor(points, depth, mask, normal_threshold=5.0, depth_threshold=0.03):
    """Return final validity; all batches remain on device, including empty masks."""
    with torch.autocast(device_type=points.device.type, enabled=False):
        normals, normals_mask = points_to_normals_tensor(points, mask)
        normal_edges = normals_edge_tensor(normals, normals_mask, normal_threshold)
        depth_edges = depth_edge_tensor(depth, mask, depth_threshold)
        return mask & ~(normal_edges & depth_edges)
