"""Explicit fixed-metadata deployment buckets for fullgraph NPU inference.

Image loading/resize and metadata validation happen before this module. No
changes to the default model.infer path are made by importing this module.
"""
import torch


class NpuCoreBucket(torch.nn.Module):
    def __init__(self, model, processed_views):
        super().__init__()
        if model.training:
            raise ValueError('Deployment buckets require model.eval()')
        if model.variable_resolution:
            raise ValueError('Use a fixed common-resolution bucket')
        for name in ('sparse_depth_prob', 'depth_scale_norm_all_prob', 'pose_scale_norm_all_prob'):
            if model.geometric_input_config[name] != 0.0:
                raise ValueError('Configure deterministic inference before creating the bucket')
        for name in ('overall_prob', 'dropout_prob', 'ray_dirs_prob', 'depth_prob', 'cam_prob'):
            if model.geometric_input_config[name] not in (0.0, 1.0):
                raise ValueError('Configure deterministic inference before creating the bucket')
        if model.ar_prob not in (0.0, 1.0):
            raise ValueError('Stochastic aspect-ratio dropout is not a deployment bucket')
        for view in processed_views:
            if any(k in view for k in ('depth_along_ray', 'ray_directions', 'ray_directions_cam',
                                      'camera_pose_quats', 'camera_pose_trans')):
                raise ValueError('This deployment path currently validates RGB plus aspect ratio only')
        # Register the opaque interpolation boundary before Dynamo tracing.
        from mapanything.utils import npu_graph_ops  # noqa: F401
        self.model = model
        self.templates = [{k:v for k,v in view.items() if k != 'img'} for view in processed_views]
        self.image_shapes = tuple(tuple(v['img'].shape) for v in processed_views)
        wrapped, plain = [], []
        offset = 0
        for view in processed_views:
            flags = view.get('pano_wrap', False)
            if torch.is_tensor(flags):
                flags = flags.detach().cpu().flatten().tolist()
            elif isinstance(flags, bool):
                flags = [flags] * view['img'].shape[0]
            else:
                raise ValueError('pano_wrap must be a bool or a per-sample Tensor')
            if len(flags) != view['img'].shape[0]:
                raise ValueError('pano_wrap batch mismatch')
            for flag in flags:
                (wrapped if flag and model.pano_wrap_dpt else plain).append(offset)
                offset += 1
        self.static_pano_indices = (tuple(wrapped),tuple(plain))

    def forward(self, *images):
        if len(images) != len(self.templates):
            raise ValueError('Create a separate bucket for a different view count')
        views = []
        for image, shape, template in zip(images, self.image_shapes, self.templates):
            if tuple(image.shape) != shape:
                raise ValueError('Create a separate bucket for a different image shape')
            views.append(dict(template, img=image))
        with torch.autocast('npu',dtype=torch.bfloat16):
            return self.model(views, memory_efficient_inference=False,
                              static_pano_indices=self.static_pano_indices)


def solve_2x2_tensor(a, b):
    """Partial-pivot LU; no host solve or data-dependent Python branch.

    Singular/nonfinite systems produce nonfinite results (not a host exception).
    Deployment callers must reject invalid intrinsics outside the graph if used
    for projection. This is deliberately not a regularized least-squares fit.
    """
    swap = a[:, 1, 0].abs() > a[:, 0, 0].abs()
    u00 = torch.where(swap, a[:, 1, 0], a[:, 0, 0])
    u01 = torch.where(swap, a[:, 1, 1], a[:, 0, 1])
    l10 = torch.where(swap, a[:, 0, 0], a[:, 1, 0]) / u00
    u11 = torch.where(swap, a[:, 0, 1], a[:, 1, 1]) - l10 * u01
    b0 = torch.where(swap, b[:, 1, 0], b[:, 0, 0])
    b1 = torch.where(swap, b[:, 0, 0], b[:, 1, 0])
    x1 = (b1 - l10 * b0) / u11
    x0 = (b0 - u01 * x1) / u00
    return torch.stack((x0, x1), dim=1)


def recover_intrinsics_tensor(rays):
    """Same sampled normal equations as geometry.py, using device 2x2 LU."""
    from mapanything.utils.geometry import recover_pinhole_intrinsics_from_ray_directions
    b, h, w, _ = rays.shape
    if h * w > 1000000:
        return recover_pinhole_intrinsics_from_ray_directions(rays)
    yi = torch.arange(0, h, max(1, h // 50), device=rays.device)
    xi = torch.arange(0, w, max(1, w // 50), device=rays.device)
    rs = rays[:, yi[:, None], xi[None, :], :].reshape(b, -1, 3)
    yy, xx = torch.meshgrid(yi.float(), xi.float(), indexing='ij')
    solutions = []
    for axis, coords in ((0, xx), (1, yy)):
        ratio = rs[..., axis] / rs[..., 2]
        a = torch.stack((torch.ones_like(ratio), ratio), dim=-1)
        target = coords.reshape(1, -1, 1).expand(b, -1, -1)
        solutions.append(solve_2x2_tensor(torch.bmm(a.transpose(1, 2), a),
                                         torch.bmm(a.transpose(1, 2), target)))
    cx, fx = solutions[0].unbind(-1)
    cy, fy = solutions[1].unbind(-1)
    zero = torch.zeros_like(fx)
    return torch.stack((fx, zero, cx, zero, fy, cy, zero, zero,
                        torch.ones_like(fx)), dim=-1).reshape(b, 3, 3)


class NpuPostprocessBucket(torch.nn.Module):
    """Default geometry outputs on NPU; confidence filtering is not enabled.

    Metadata is fixed at construction; inputs/outputs remain device tensors.
    fused_edges=True selects TileLang K2/K3 in addition to K1; False retains
    the Tensor edge/mask path for A/B testing. JIT and threshold calibration
    warmup must precede compilation/capture.
    """
    def __init__(self, processed_views, tilelang=True, batch_geometry=True, fused_edges=False):
        super().__init__()
        from mapanything.utils.image import IMAGE_NORMALIZATION_DICT
        self.tilelang = tilelang
        self.batch_geometry = batch_geometry
        if fused_edges and not (tilelang and batch_geometry):
            raise ValueError("Fused edges require batched TileLang geometry")
        self.fused_edges = fused_edges
        for i, view in enumerate(processed_views):
            norm = view['data_norm_type'][0]
            if norm == 'identity':
                mean, std = torch.zeros(3), torch.ones(3)
            else:
                spec = IMAGE_NORMALIZATION_DICT[norm]
                mean, std = spec.mean, spec.std
            self.register_buffer(f'mean_{i}', mean.flatten().to(view['img'].device))
            self.register_buffer(f'std_{i}', std.flatten().to(view['img'].device))

    def forward(self, raw_outputs, images):
        from mapanything.utils.geometry import quaternion_to_rotation_matrix
        from mapanything.utils.geometry_tensor import points_to_normals_tensor, normals_edge_tensor, depth_edge_tensor
        if self.tilelang:
            from operators.k1_tilelang.torch_op import pointmap_to_normals
        if self.fused_edges:
            from operators.geometry_tilelang.torch_op import normal_candidates, mask_and_apply
        outputs = []
        with torch.autocast('npu', enabled=False):
            # Independent equal-sized views share one batched geometry call
            # for either the Tensor or fused TileLang edge/mask path.
            masks = None
            filtered = None
            if self.batch_geometry:
                points = torch.cat([r['pts3d'] for r in raw_outputs], dim=0).float()
                base_mask = torch.cat([r['non_ambiguous_mask'] for r in raw_outputs], dim=0)
                depth = torch.cat([r['depth_along_ray'] for r in raw_outputs], dim=0).squeeze(-1)
                if self.tilelang:
                    n, v = pointmap_to_normals(points.contiguous(), base_mask.contiguous())
                else:
                    n, v = points_to_normals_tensor(points, base_mask)
                sizes = [r['pts3d'].shape[0] for r in raw_outputs]
                if self.fused_edges:
                    candidate = normal_candidates(n, v)
                    camera_points = torch.cat([r['pts3d_cam'] for r in raw_outputs], dim=0).float().contiguous()
                    final, p_out, c_out, d_out, z_out = mask_and_apply(
                        candidate, depth.contiguous(), base_mask, points, camera_points)
                    filtered = {key: value.split(sizes, dim=0) for key, value in (
                        ('pts3d', p_out), ('pts3d_cam', c_out),
                        ('depth_along_ray', d_out.unsqueeze(-1)), ('depth_z', z_out.unsqueeze(-1)))}
                else:
                    final = base_mask & ~(normals_edge_tensor(n, v) & depth_edge_tensor(depth, base_mask))
                masks = final.split(sizes, dim=0)
            for i, (raw, image) in enumerate(zip(raw_outputs, images)):
                out = dict(raw)
                out['img_no_norm'] = (image.permute(0, 2, 3, 1).float() * getattr(self, f'std_{i}') + getattr(self, f'mean_{i}')).clamp(0, 1)
                out['depth_z'] = raw['pts3d_cam'][..., 2:3]
                out['intrinsics'] = recover_intrinsics_tensor(raw['ray_directions'].float())
                rot = quaternion_to_rotation_matrix(raw['cam_quats'])
                top = torch.cat((rot, raw['cam_trans'].unsqueeze(-1)), dim=-1)
                bottom = torch.zeros_like(top[:, :1, :])
                bottom[:, :, 3] = 1
                out['camera_poses'] = torch.cat((top, bottom), dim=1)
                if masks is not None:
                    mask = masks[i].unsqueeze(-1)
                else:
                    mask = raw['non_ambiguous_mask']
                    if self.tilelang:
                        normals, valid = pointmap_to_normals(raw['pts3d'].float().contiguous(), mask.contiguous())
                    else:
                        normals, valid = points_to_normals_tensor(raw['pts3d'], mask)
                    normal_edges = normals_edge_tensor(normals, valid)
                    depth_edges = depth_edge_tensor(raw['depth_along_ray'].squeeze(-1), mask)
                    mask = (mask & ~(normal_edges & depth_edges)).unsqueeze(-1)
                for key in ('pts3d', 'pts3d_cam', 'depth_along_ray', 'depth_z'):
                    out[key] = filtered[key][i] if filtered is not None else out[key] * mask
                out['mask'] = mask
                outputs.append(out)
        return outputs


class NpuInferenceBucket(torch.nn.Module):
    """Fixed-resolution/view/routing main model followed by device postprocess."""
    def __init__(self, model, processed_views, tilelang=True, fused_edges=False):
        super().__init__()
        self.core = NpuCoreBucket(model, processed_views)
        self.post = NpuPostprocessBucket(processed_views, tilelang=tilelang, fused_edges=fused_edges)

    def forward(self, *images):
        return self.post(self.core(*images), images)


# Disable optional rewriting until separately qualified on this model. ACL
# capture/replay still works; these options do not request eager fallback.
GRAPH_OPTIONS = {
    'pattern_fusion_pass': False,
    'inplace_pass': False,
    'input_inplace_pass': False,
    'remove_cat_ops': False,
}


class NpuGraphRunner(torch.nn.Module):
    """Build/warm an isolated inference model, then replay fixed-metadata inputs.

    Construction changes the supplied model to deterministic inference mode.
    Do not share it with training or calls that change geometric configuration.
    Outputs may use graph-owned storage; clone before retaining across replays.
    """
    @torch.inference_mode()
    def __init__(self, model, processed_views, fused_edges=False):
        super().__init__()
        model.eval()
        model._configure_geometric_input_config(True, True, True, True, True)
        bucket = NpuInferenceBucket(model, processed_views, fused_edges=fused_edges)
        images = tuple(v['img'] for v in processed_views)
        bucket(*images)  # Compile TileLang before capture.
        self.compiled = torch.compile(bucket, backend='npugraph_ex',
                                      fullgraph=True, dynamic=False,
                                      options=GRAPH_OPTIONS.copy())
        for _ in range(4):
            self.compiled(*images)
        torch.npu.synchronize()

    @torch.inference_mode()
    def forward(self, *images):
        return self.compiled(*images)
