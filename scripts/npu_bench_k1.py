"""Validate and benchmark queue/manual K1 against the existing Tensor path."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch_npu
from mapanything.utils.geometry_tensor import points_to_normals_tensor


def compare(p,m,fn):
    ref,rm=points_to_normals_tensor(p.cpu(),m.cpu())
    n,v=fn(p,m);n=n.cpu();v=v.cpu()
    finite=torch.isfinite(ref)&torch.isfinite(n)
    err=(n[finite]-ref[finite]).abs().max().item() if finite.any() else 0.
    row={'max_abs':err,'mask_mismatch':int((v!=rm).sum()),
         'nan_mismatch':int((torch.isnan(n)!=torch.isnan(ref)).sum()),
         'inf_mismatch':int((torch.isinf(n)!=torch.isinf(ref)).sum())}
    assert row['mask_mismatch']==0 and row['nan_mismatch']==0 and row['inf_mismatch']==0,row
    tn,tv=points_to_normals_tensor(p,m)
    tn,tv=tn.cpu(),tv.cpu()
    torch.testing.assert_close(n,tn,atol=0,rtol=0,equal_nan=True)
    assert torch.equal(v,tv)
    row['npu_tensor_exact']=True
    row['cpu_atol_2e4_gate_pass']=err < 2e-4
    return row


def measure(p,m,fn,repeats,group):
    for _ in range(5): n,v=fn(p,m)
    torch.npu.synchronize()
    wall,span=[],[]
    for _ in range(repeats):
        a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        t=time.perf_counter();a.record()
        for _ in range(group): n,v=fn(p,m)
        b.record();torch.npu.synchronize()
        wall.append((time.perf_counter()-t)*1000/group)
        span.append(a.elapsed_time(b)/group)
    return {'wall_ms':wall,'event_span_ms':span,'wall_mean_ms':float(np.mean(wall)),
            'event_mean_ms':float(np.mean(span)), 'wall_p50_ms':float(np.median(wall))}


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--library',type=Path,default=Path('work/npu-k1/build-gcc13/libmeow_k1.so'))
    ap.add_argument('--raw',type=Path,default=Path('work/npu-postprocess/raw'))
    ap.add_argument('--out',type=Path,default=Path('work/npu-k1/results.json'))
    ap.add_argument('--tiles',type=int,nargs='+',default=[128])
    ap.add_argument('--blocks',type=int,nargs='+',default=[32])
    ap.add_argument('--repeats',type=int,default=10)
    ap.add_argument('--group',type=int,default=10)
    ap.add_argument('--trace',action='store_true')
    ap.add_argument('--smoke-only',action='store_true')
    args=ap.parse_args(); torch.ops.load_library(str(args.library.resolve()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    result={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'smoke':[],'cases':[],
            'note':'Native outputs have padded row strides. Event spans include launch gaps; kernel duration requires profiler.'}
    def save(): args.out.write_text(json.dumps(result,indent=2))
    gen=torch.Generator().manual_seed(42)
    for b,h,w in [(1,1,1),(1,1,65),(1,65,1),(2,2,3),(2,17,33),(2,65,518)]:
        p=torch.randn((b,h,w,3),generator=gen);m=torch.rand((b,h,w),generator=gen)>.15
        for special in [False,True]:
            if special:
                p.flatten()[0]=float('nan');p.flatten()[-1]=float('inf');m[0]=False
            pn,mn=p.to('npu'),m.to('npu')
            for manual in [False,True]:
                fn=lambda p,m:torch.ops.meow_k1.pointmap_to_normals(p,m,manual,128,32)
                row={'shape':list(p.shape),'special':special,'manual':manual,**compare(pn,mn,fn)}
                result['smoke'].append(row);print(row,flush=True);save()
    # Exercise task-queue ordering and allocations on a non-default stream.
    p=torch.randn((2,17,33,3),generator=gen).to('npu')
    m=(torch.rand((2,17,33),generator=gen)>.1).to('npu')
    ref,rm=points_to_normals_tensor(p.cpu(),m.cpu())
    current=torch.npu.current_stream(); other=torch.npu.Stream()
    other.wait_stream(current)
    with torch.npu.stream(other):
        outs=[torch.ops.meow_k1.pointmap_to_normals(p,m,r,128,32) for r in [False,True]]
    current.wait_stream(other)
    for n,v in outs:
        torch.testing.assert_close(n.cpu(),ref,atol=2e-4,rtol=0,equal_nan=True)
        assert torch.equal(v.cpu(),rm)
    torch.testing.assert_close(outs[0][0].cpu(),outs[1][0].cpu(),atol=0,rtol=0,equal_nan=True)
    result['nondefault_stream_and_route_agreement']='passed';save()
    if args.smoke_only: return
    for track in ['blk_mixed','blk_erp','blk_pinhole','blk_fisheye']:
        raw=torch.load(args.raw/f'{track}_0.pt',map_location='cpu',weights_only=False)['raw']
        for count in [1,4]:
            p=torch.cat([v['pts3d'].float() for v in raw[:count]]).contiguous().to('npu')
            m=torch.cat([v['non_ambiguous_mask'] for v in raw[:count]]).contiguous().to('npu')
            cases=[('tensor',points_to_normals_tensor)]
            for tile in args.tiles:
                for blocks in args.blocks:
                    for manual in [False,True]:
                        def fn(p,m,manual=manual,tile=tile,blocks=blocks):
                            return torch.ops.meow_k1.pointmap_to_normals(p,m,manual,tile,blocks)
                        cases.append((f'{"manual" if manual else "queue"}_t{tile}_b{blocks}',fn))
            for name,fn in cases:
                row={'track':track,'shape':list(p.shape),'route':name,'accuracy':compare(p,m,fn),
                     'latency':measure(p,m,fn,args.repeats,args.group)}
                result['cases'].append(row);save()
                print(track,count,name,row['accuracy'],'ms',row['latency']['wall_mean_ms'],flush=True)
            if args.trace:
                prof=torch_npu.profiler
                with prof.profile(activities=[prof.ProfilerActivity.CPU,prof.ProfilerActivity.NPU],
                    schedule=prof.schedule(wait=0,warmup=1,active=2,repeat=1),
                    on_trace_ready=prof.tensorboard_trace_handler(str(args.out.parent/'trace'/f'{track}_b{count}')),
                    record_shapes=True,experimental_config=prof._ExperimentalConfig(
                        profiler_level=prof.ProfilerLevel.Level1)) as pr:
                    for _ in range(3):
                        for name,fn in cases:
                            with torch.autograd.profiler.record_function('K1::'+name): n,v=fn(p,m)
                        torch.npu.synchronize();pr.step()
    save()

if __name__=='__main__': main()
