"""Hardware smoke checks; full model uses random weights unless --checkpoint is set."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu

from mapanything.utils.device import get_device, get_amp_dtype


def distributed():
    from mapanything.utils.train_tools import init_distributed_mode, all_reduce_mean, SmoothedValue
    args = SimpleNamespace(nodist=False, dist_url='env://')
    init_distributed_mode(args)
    net = torch.nn.parallel.DistributedDataParallel(
        torch.nn.Linear(8, 4).to('npu'), device_ids=[args.gpu])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    loss = net(torch.ones(2, 8, device='npu')).square().mean()
    loss.backward()
    opt.step()
    assert abs(all_reduce_mean(float(args.rank)) - (args.world_size-1)/2) < 1e-6
    stats = SmoothedValue()
    stats.update(args.rank + 1)
    stats.synchronize_between_processes()
    assert stats.count == args.world_size
    assert stats.total == args.world_size * (args.world_size+1)/2
    print('PASS HCCL DDP optimizer + metric reductions', flush=True)
    torch.distributed.destroy_process_group()


def operators():
    torch.manual_seed(42)
    x = torch.randn(2, 37, 128, requires_grad=True)
    xn = x.detach().to('npu').requires_grad_()
    a, b = x.chunk(2, dim=-1)
    ref = F.silu(a) * b
    out = torch_npu.npu_swiglu(xn, dim=-1)
    torch.testing.assert_close(out.cpu(), ref, atol=1e-5, rtol=1e-4)
    out.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(xn.grad.cpu(), x.grad, atol=1e-5, rtol=1e-4)
    print('PASS fused SwiGLU forward/backward', flush=True)
    for length in (37, 1370):
        inputs = [torch.randn(1, 2, length, 64) * 0.2 for _ in range(3)]
        cpu = [x.clone().requires_grad_() for x in inputs]
        npu = [x.to('npu', dtype=torch.bfloat16).requires_grad_() for x in inputs]
        expected = F.scaled_dot_product_attention(*cpu)
        actual = F.scaled_dot_product_attention(*npu)
        torch.testing.assert_close(actual.float().cpu(), expected, atol=0.004, rtol=0.03)
        expected.square().mean().backward()
        actual.float().square().mean().backward()
        for a, b in zip(npu, cpu):
            torch.testing.assert_close(a.grad.float().cpu(), b.grad, atol=1e-6, rtol=0.08)
        print('PASS SDPA forward/backward', length, flush=True)
    from mapanything.train.losses import compute_normal_loss
    from mapanything.utils.spherical import solid_angle_weights, vmf_nll
    p = torch.randn(1, 16, 20, 3, device='npu', requires_grad=True)
    gt = torch.randn_like(p)
    mask = torch.ones(1, 16, 20, dtype=torch.bool, device='npu')
    loss = compute_normal_loss(p, gt, mask).mean()
    loss = loss + vmf_nll(F.normalize(p, dim=-1)[..., 0], torch.tensor(20., device='npu')).mean()
    weights = solid_angle_weights(F.normalize(gt, dim=-1), mask)
    assert torch.isfinite(weights).all()
    loss.backward()
    assert torch.isfinite(p.grad).all()
    from mapanything.utils.train_tools import NativeScalerWithGradNormCount
    opt = torch.optim.AdamW([p], lr=1e-4)
    scaler = NativeScalerWithGradNormCount(enabled=False, device='npu')
    opt.zero_grad()
    scaler(p.square().mean(), opt, parameters=[p])
    print('PASS geometry/loss backward and AdamW/scaler', flush=True)


def full_model(args):
    from mapanything.models import _init_hydra_config, init_model
    overrides = [
        'model=mapanything', 'machine=default', 'model/task=images_only',
        '++model.encoder.torch_hub_pretrained=false', '++model.task.ar_prob=1.0',
        f'model.model_config.variable_resolution={str(args.mode == "ragged").lower()}',
        f'++model.model_config.pano_wrap_dpt={str(args.mode == "wrap").lower()}',
        f'model.encoder.gradient_checkpointing={str(args.backward).lower()}',
        f'model.info_sharing.module_args.gradient_checkpointing={str(args.backward).lower()}',
        f'model.pred_head.gradient_checkpointing={str(args.backward).lower()}',
    ]
    if args.stage3_loss:
        overrides.append('loss=overall_loss_solid_angle_vmf')
    cfg = _init_hydra_config('configs/train.yaml', overrides=overrides)
    model = init_model(cfg.model.model_str, cfg.model.model_config)
    if args.checkpoint:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from meow_model import load_ckpt_inplace
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        missing = load_ckpt_inplace(model, state.get('model', state))
        if missing:
            raise RuntimeError(f'Checkpoint is missing {missing} aspect-ratio tensors')
        del state
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    model.to('npu').train(args.backward)
    views = []
    for i in range(args.views):
        h = args.res
        w = args.res + (14 * i if args.mode == 'ragged' else 0)
        views.append(dict(img=torch.randn(args.batch, 3, h, w, device='npu'),
                          true_shape=torch.tensor([[h, w]] * args.batch, device='npu'),
                          data_norm_type=['dinov2'] * args.batch,
                          aspect_ratio=torch.full((args.batch,), w/h, device='npu'),
                          pano_wrap=torch.tensor([j % 2 == 0 for j in range(args.batch)], device='npu')))
        if args.stage3_loss:
            yy, xx = torch.meshgrid(torch.linspace(-0.5, 0.5, h, device='npu'),
                                    torch.linspace(-0.5, 0.5, w, device='npu'), indexing='ij')
            rays = F.normalize(torch.stack([xx, yy, torch.ones_like(xx)], -1), dim=-1)
            rays = rays.unsqueeze(0).expand(args.batch, -1, -1, -1).contiguous()
            pose = torch.eye(4, device='npu').repeat(args.batch, 1, 1)
            pose[:, 0, 3] = i * 0.1
            pts_cam = rays * 2
            valid = torch.ones(args.batch, h, w, dtype=torch.bool, device='npu')
            views[-1].update(camera_pose=pose, pts3d=pts_cam + pose[:, None, None, :3, 3],
                pts3d_cam=pts_cam, ray_directions_cam=rays,
                depth_along_ray=torch.full((args.batch, h, w, 1), 2., device='npu'),
                camera_pose_quats=torch.tensor([[0., 0., 0., 1.]], device='npu').repeat(args.batch, 1),
                camera_pose_trans=pose[:, :3, 3], valid_mask=valid,
                non_ambiguous_mask=valid.clone(),
                is_metric_scale=torch.ones(args.batch, dtype=torch.bool, device='npu'),
                is_synthetic=torch.tensor([j % 2 == 0 for j in range(args.batch)], device='npu'))
    torch.npu.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.set_grad_enabled(args.backward), torch.autocast('npu', dtype=torch.bfloat16):
        output = model(views)
    for v, out in zip(views, output):
        assert out['pts3d'].shape == (args.batch, *v['img'].shape[-2:], 3)
        for name, value in out.items():
            if torch.is_tensor(value) and value.is_floating_point():
                assert torch.isfinite(value).all(), name
    if args.backward:
        if args.stage3_loss:
            import mapanything.train.losses as losses
            criterion = eval(cfg.loss.train_criterion, vars(losses)).to('npu')
            loss, details = criterion(views, output)
            assert torch.isfinite(loss)
            print('Stage3 criterion:', float(loss.detach()), 'detail terms:', len(details), flush=True)
        else:
            loss = sum(out['pts3d'].float().square().mean() for out in output)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        print('PASS full backward; gradient tensors:', len(grads), flush=True)
        if args.optimizer_step:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.95))
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            print('PASS full-model AdamW step', flush=True)
    torch.npu.synchronize()
    print(json.dumps(dict(mode=args.mode, backward=args.backward, resolution=args.res,
                          views=args.views, batch=args.batch, elapsed_s=time.perf_counter()-start,
                          peak_allocated_gb=torch.npu.max_memory_allocated()/1e9)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--mode', choices=['plain', 'wrap', 'ragged'], default='plain')
    parser.add_argument('--res', type=int, default=56)
    parser.add_argument('--views', type=int, default=2)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--checkpoint')
    parser.add_argument('--stage3-loss', action='store_true')
    parser.add_argument('--optimizer-step', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    if (args.stage3_loss or args.optimizer_step) and not (args.full and args.backward):
        parser.error('--stage3-loss/--optimizer-step require --full --backward')
    assert args.res >= 56 and args.res % 14 == 0
    if args.distributed:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(42)
    print(torch.__version__, torch_npu.__version__, torch.npu.get_device_name(), flush=True)
    assert get_device().type == 'npu'
    assert get_amp_dtype(get_device()) == torch.bfloat16
    if args.distributed:
        distributed()
    elif args.full:
        full_model(args)
    else:
        operators()
