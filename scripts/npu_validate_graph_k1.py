"""Validate fullgraph npugraph_ex K1 with changed data, shapes and output strides."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch_npu
from operators.k1_tilelang.torch_op import pointmap_to_normals
from mapanything.utils.geometry_tensor import points_to_normals_tensor
from npu_bench_k1 import measure


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('work/npu-graph/k1.json'))
    ap.add_argument('--small',action='store_true')
    args=ap.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    result={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'backend':'npugraph_ex',
            'fullgraph':True,'dynamic':False,'cases':[]}
    def save(): args.out.write_text(json.dumps(result,indent=2))
    save()
    shapes=[(2,17,33)] if args.small else [(b,h,518) for b in (1,4) for h in (252,392,518)]
    for b,h,w in shapes:
        p=torch.randn((b,h,w,3),device='npu');m=torch.rand((b,h,w),device='npu')>.15
        pointmap_to_normals(p,m) # JIT outside Dynamo/capture
        compiled=torch.compile(pointmap_to_normals,backend='npugraph_ex',fullgraph=True,dynamic=False)
        row={'shape':[b,h,w,3],'checks':[]};result['cases'].append(row);save()
        start=time.perf_counter()
        for _ in range(4): compiled(p,m)
        torch.npu.synchronize();row['compile_capture_warmup_s']=time.perf_counter()-start
        old=None
        for iteration in range(4):
            # Same shape, both new allocations and changed values; no static-input reuse assumption.
            p=torch.randn((b,h,w,3),device='npu')
            m=torch.rand((b,h,w),device='npu')>(.1+.15*iteration)
            reference=points_to_normals_tensor(p,m)
            eager=pointmap_to_normals(p,m)
            actual=compiled(p,m)
            for a,r,e in zip(actual,reference,eager):
                torch.testing.assert_close(a.cpu(),r.cpu(),atol=0,rtol=0,equal_nan=True)
                assert a.stride()==e.stride(),(a.stride(),e.stride())
            changed=old is None or not torch.equal(old,actual[1].cpu())
            assert changed,'Replay reused stale output'
            old=actual[1].cpu().clone()
            row['checks'].append({'iteration':iteration,'exact':True,'changed_output':changed,
                                  'strides':[list(x.stride()) for x in actual]});save()
        row['eager']=measure(p,m,pointmap_to_normals,10,10)
        row['graph']=measure(p,m,compiled,10,10)
        save();print(row,flush=True)
    print('K1 GRAPH PASSED',flush=True)

if __name__=='__main__': main()
