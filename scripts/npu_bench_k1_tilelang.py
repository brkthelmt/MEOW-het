"""Same-process accuracy and latency comparison of Tensor/Ascend C/TileLang K1."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import torch_npu
from mapanything.utils.geometry_tensor import points_to_normals_tensor
from operators.k1_tilelang.kernel import get_kernel, pointmap_to_normals
from npu_bench_k1 import measure


def diff(n, v, ref, mask):
    finite = torch.isfinite(n) & torch.isfinite(ref)
    delta = (n[finite] - ref[finite]).abs()
    return {
        'max_abs': delta.max().item() if delta.numel() else 0.,
        'mean_abs': delta.mean().item() if delta.numel() else 0.,
        'mask_mismatch': int((v != mask).sum()),
        'nan_mismatch': int((torch.isnan(n) != torch.isnan(ref)).sum()),
        'posinf_mismatch': int((torch.isposinf(n) != torch.isposinf(ref)).sum()),
        'neginf_mismatch': int((torch.isneginf(n) != torch.isneginf(ref)).sum()),
        'exact': torch.allclose(n, ref, atol=0, rtol=0, equal_nan=True) and torch.equal(v, mask),
        'allclose_2e4_1e4': torch.allclose(n, ref, atol=2e-4, rtol=1e-4, equal_nan=True),
    }


def accuracy(p, m, fn):
    n, v = fn(p, m)
    n, v = n.cpu(), v.cpu()
    cpu, cm = points_to_normals_tensor(p.cpu(), m.cpu())
    tn, tm = points_to_normals_tensor(p, m)
    nc = diff(n, v, tn.cpu(), tm.cpu())
    cc = diff(n, v, cpu, cm)
    passed = nc['allclose_2e4_1e4'] and not any(nc[k] for k in (
        'mask_mismatch', 'nan_mismatch', 'posinf_mismatch', 'neginf_mismatch'))
    return {'npu': nc, 'cpu': cc, 'passed': passed}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', type=int, nargs='+', default=[20, 40])
    ap.add_argument('--out', type=Path, default=Path('work/npu-k1/tilelang/results.json'))
    ap.add_argument('--smoke-only', action='store_true')
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--group', type=int, default=10)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(str(Path('work/npu-k1/build-gcc13/libmeow_k1.so').resolve()))
    result = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'tilelang_commit': subprocess.check_output(
                  ['git', '-C', 'work/npu-k1/tilelang/source', 'rev-parse', 'HEAD'], text=True).strip(),
              'smoke': [], 'cases': [], 'compilation': []}
    def save():
        args.out.write_text(json.dumps(result, indent=2))
    def compile_shape(b, h, w, blocks):
        start = time.perf_counter()
        fn = get_kernel(b, h, w, blocks)
        result['compilation'].append({'shape': [b, h, w], 'blocks': blocks,
                                      'seconds_including_cache_lookup': time.perf_counter() - start})
        if hasattr(fn, 'get_kernel_source'):
            (args.out.parent / f'generated_b{b}_h{h}_w{w}_c{blocks}.cpp').write_text(fn.get_kernel_source())
        save()
    gen = torch.Generator().manual_seed(42)
    for b, h, w in [(1,1,1), (1,1,65), (1,65,1), (2,2,3), (2,17,33), (2,65,518), (1,3,576)]:
        compile_shape(b,h,w,args.blocks[0])
        p = torch.randn((b,h,w,3), generator=gen)
        m = torch.rand((b,h,w), generator=gen) > .15
        for special in [False, True]:
            if special:
                p.flatten()[0] = float('nan')
                p.flatten()[-1] = float('inf')
                m[0] = False
            pn, mn = p.npu(), m.npu()
            acc = accuracy(pn, mn, lambda p,m: pointmap_to_normals(p,m,blocks=args.blocks[0]))
            row = {'shape': list(p.shape), 'special': special, **acc}
            result['smoke'].append(row); save(); print(row, flush=True)
            if not acc['passed']:
                raise AssertionError(row)
    # Exercise non-finite points under a valid mask, plus degenerate geometry.
    yy, xx = torch.meshgrid(torch.arange(17), torch.arange(33), indexing="ij")
    plane = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).float()[None].repeat(2,1,1,1)
    for name in ('plane', 'degenerate', 'valid_nan', 'valid_inf'):
        p = plane.clone()
        if name == 'degenerate':
            p.zero_()
        elif name == 'valid_nan':
            p[0,8,16,0] = float('nan')
        elif name == 'valid_inf':
            p[0,8,16,0] = float('inf')
        m = torch.ones((2,17,33), dtype=torch.bool)
        acc = accuracy(p.npu(),m.npu(),lambda p,m: pointmap_to_normals(p,m,blocks=args.blocks[0]))
        row = {'fixture': name, 'shape': list(p.shape), **acc}
        result['smoke'].append(row); save(); print(row,flush=True)
        if not acc['passed']:
            raise AssertionError(row)
    # Explicit non-default stream and producer/consumer dependencies.
    p = torch.randn((2,17,33,3), generator=gen).npu()
    m = (torch.rand((2,17,33), generator=gen) > .1).npu()
    current, other = torch.npu.current_stream(), torch.npu.Stream()
    other.wait_stream(current)
    with torch.npu.stream(other):
        n,v = pointmap_to_normals(p,m,blocks=args.blocks[0])
    current.wait_stream(other)
    tn,tv = points_to_normals_tensor(p,m)
    stream_diff = diff(n.cpu(), v.cpu(), tn.cpu(), tv.cpu())
    result['stream'] = stream_diff; save()
    assert stream_diff['allclose_2e4_1e4'] and stream_diff['mask_mismatch'] == 0
    if args.smoke_only:
        return
    for track in ['blk_mixed', 'blk_erp', 'blk_pinhole', 'blk_fisheye']:
        raw = torch.load(f'work/npu-postprocess/raw/{track}_0.pt', map_location='cpu', weights_only=False)['raw']
        for count in [1,4]:
            p = torch.cat([v['pts3d'].float() for v in raw[:count]]).contiguous().npu()
            m = torch.cat([v['non_ambiguous_mask'] for v in raw[:count]]).contiguous().npu()
            cases = [('tensor', points_to_normals_tensor),
                     ('queue_t256_b40', lambda p,m: torch.ops.meow_k1.pointmap_to_normals(p,m,False,256,40)),
                     ('manual_t256_b40', lambda p,m: torch.ops.meow_k1.pointmap_to_normals(p,m,True,256,40))]
            for blocks in args.blocks:
                compile_shape(*p.shape[:3], blocks)
                cases.append((f'tilelang_rows_b{blocks}', lambda p,m,blocks=blocks: pointmap_to_normals(p,m,blocks=blocks)))
            for name, fn in cases:
                acc = accuracy(p,m,fn)
                row = {'track': track, 'shape': list(p.shape), 'route': name, 'accuracy': acc}
                result['cases'].append(row); save()
                if not acc['passed']:
                    raise AssertionError(row)
                row['latency'] = measure(p,m,fn,args.repeats,args.group)
                save(); print(track, count, name, acc, row['latency']['wall_mean_ms'], flush=True)
            if args.trace:
                prof = torch_npu.profiler
                with prof.profile(activities=[prof.ProfilerActivity.CPU,prof.ProfilerActivity.NPU],
                    schedule=prof.schedule(wait=0,warmup=1,active=2,repeat=1),
                    on_trace_ready=prof.tensorboard_trace_handler(str(args.out.parent/'trace'/f'{track}_b{count}')),
                    record_shapes=True,experimental_config=prof._ExperimentalConfig(
                        profiler_level=prof.ProfilerLevel.Level1)) as pr:
                    for _ in range(3):
                        for name,fn in cases:
                            with torch.autograd.profiler.record_function('K1::'+name):
                                n,v = fn(p,m)
                        torch.npu.synchronize(); pr.step()
    save()

if __name__ == '__main__':
    main()
