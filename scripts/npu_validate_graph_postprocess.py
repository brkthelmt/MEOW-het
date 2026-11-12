"""Validate device postprocess and fullgraph replay against real cached outputs."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch_npu
from mapanything.utils.npu_graph import NpuPostprocessBucket, solve_2x2_tensor
from mapanything.utils.inference import postprocess_model_outputs_for_inference
from npu_validate_graph_model import compare
from npu_profile_model import measure, trace


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,default=Path('work/npu-postprocess/raw'))
    ap.add_argument('--out',type=Path,default=Path('work/npu-graph/postprocess'))
    ap.add_argument('--trace',action='store_true')
    ap.add_argument('--fused-edges',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    results={}
    # Exercise row pivoting independently of the normal-equation inputs.
    a=torch.tensor([[[0.,2.],[3.,4.]],[[4.,2.],[2.,5.]]],device='npu')
    b=torch.tensor([[[2.],[7.]],[[8.],[12.]]],device='npu')
    torch.testing.assert_close(solve_2x2_tensor(a,b).cpu(),torch.linalg.solve(a.cpu(),b.cpu()).squeeze(-1))
    for track in ('blk_mixed','blk_erp','blk_pinhole','blk_fisheye'):
        data=torch.load(args.cache/f'{track}_0.pt',map_location='cpu',weights_only=False)
        raw=[{k:v.to('npu') if torch.is_tensor(v) else v for k,v in r.items()} for r in data['raw']]
        views=[{k:v.to('npu') if torch.is_tensor(v) else v for k,v in r.items()} for r in data['processed']]
        images=tuple(v['img'] for v in views)
        post=NpuPostprocessBucket(views,fused_edges=args.fused_edges)
        eager=post(raw,images)
        os.environ['MEOW_GEOMETRY_POSTPROCESS']='tensor'
        reference=postprocess_model_outputs_for_inference(raw,views)
        row={'vs_existing_tensor':compare(eager,reference)};results[track]=row
        if args.fused_edges:
            old=NpuPostprocessBucket(views,fused_edges=False)
            row['vs_unfused']=compare(eager,old(raw,images))
            assert all(r['exact'] for r in row['vs_unfused'])
            old_graph=torch.compile(old,backend='npugraph_ex',fullgraph=True,dynamic=False)
            for _ in range(4):old_graph(raw,images)
            row['unfused_graph']=measure(lambda:old_graph(raw,images),3,10)
        graph=torch.compile(post,backend='npugraph_ex',fullgraph=True,dynamic=False)
        for _ in range(4): graph(raw,images)
        row['replay']=[]
        for step in range(3):
            changed=[dict(r,pts3d=(r['pts3d']*(1+step*.01)).clone(),
                          non_ambiguous_mask=(r['non_ambiguous_mask'] & (torch.rand_like(r['non_ambiguous_mask'],dtype=torch.float32)>.05))) for r in raw]
            checks=compare(graph(changed,images),post(changed,images))
            assert all(r['exact'] for r in checks)
            row['replay'].append(checks)
        row['eager']=measure(lambda:post(raw,images),3,10)
        row['graph']=measure(lambda:graph(raw,images),3,10)
        if args.trace and track=='blk_mixed': trace(lambda:graph(raw,images),args.out/track,'fullgraph',2,False)
        row['status']='passed'
        (args.out/'results.json').write_text(json.dumps(results,indent=2))
        print(track,row['eager'],row['graph'],flush=True)
        del graph,post,eager,reference,raw,views,images,changed
        if args.fused_edges:del old_graph,old
        torch._dynamo.reset()

if __name__=='__main__':main()
