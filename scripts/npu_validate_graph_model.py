"""Compile real MEOW buckets without allowing Dynamo graph breaks."""
import argparse
import json
import os
import gc
import traceback
from pathlib import Path
import torch
import torch_npu
from meow_model import build_model_with_ar, load_ckpt_inplace
from npu_profile_model import prepare, measure, trace
from mapanything.utils.inference import preprocess_input_views_for_inference, validate_input_views_for_inference
from mapanything.utils.npu_graph import NpuCoreBucket, NpuInferenceBucket, NpuPostprocessBucket, GRAPH_OPTIONS, NpuGraphRunner


def compare(actual, reference):
    assert len(actual)==len(reference)
    rows=[]
    for i,(a,r) in enumerate(zip(actual,reference)):
        assert a.keys()==r.keys()
        for key in a:
            if torch.is_tensor(a[key]):
                av,rv=a[key].cpu(),r[key].cpu()
                if av.dtype==torch.bool:
                    mismatch=int((av!=rv).sum());err=0.
                else:
                    finite=torch.isfinite(av)&torch.isfinite(rv)
                    err=float((av[finite]-rv[finite]).abs().max()) if finite.any() else 0.
                    mismatch=int((torch.isnan(av)!=torch.isnan(rv)).sum())+int((torch.isinf(av)!=torch.isinf(rv)).sum())
                exact=torch.allclose(av,rv,atol=0,rtol=0,equal_nan=True)
                rows.append({'view':i,'key':key,'max_abs':err,'nonfinite_or_bool_mismatch':mismatch,'exact':exact})
                try:
                    torch.testing.assert_close(av,rv,atol=2e-4,rtol=2e-4,equal_nan=True)
                except AssertionError as exc:
                    raise AssertionError(f'view={i} key={key} max_abs={err}: {exc}') from exc
    return rows


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='checkpoints/MEOW-sB25.pth')
    ap.add_argument('--data-root',type=Path,default=Path('work/laser-benchmark/meow_laser_benchmark/v1'))
    ap.add_argument('--tracks',nargs='+',default=['blk_mixed'])
    ap.add_argument('--out',type=Path,default=Path('work/npu-graph/model'))
    ap.add_argument('--trace',action='store_true')
    ap.add_argument('--fused-edges',action='store_true')
    ap.add_argument('--compare-unfused',action='store_true')
    ap.add_argument('--stage',choices=['core','full'],default='core')
    ap.add_argument('--runner',action='store_true',help='Validate the production runner including its warmup')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    result={'stage':args.stage,'backend':'npugraph_ex','fullgraph':True,
            'runner':args.runner,'fused_edges':args.fused_edges,'graph_options':GRAPH_OPTIONS,
            'torch':torch.__version__,'torch_npu':torch_npu.__version__,
            'device':torch.npu.get_device_name(0),
            'visible_devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
            'checkpoint':args.checkpoint,'tracks':{}}
    def save(): (args.out/'results.json').write_text(json.dumps(result,indent=2))
    model=build_model_with_ar('npu','wrap')
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert load_ckpt_inplace(model,ckpt.get('model',ckpt))==0
    del ckpt
    model._configure_geometric_input_config(True,True,True,True,True)
    for track in args.tracks:
        row={};result['tracks'][track]=row;save()
        try:
            views,meta=prepare(args.data_root/track,0)
            processed=preprocess_input_views_for_inference(validate_input_views_for_inference(views))
            row['metadata']=meta
            images=tuple(v['img'] for v in processed)
            with torch.autocast('npu',dtype=torch.bfloat16):
                original=model(processed,memory_efficient_inference=False)
            bucket=NpuCoreBucket(model,processed)
            row['static_routing_vs_original']=compare(bucket(*images),original)
            if args.stage == 'full':
                bucket=NpuInferenceBucket(model,processed,fused_edges=args.fused_edges)
                reference=NpuPostprocessBucket(processed)(original,images)
                row['full_vs_staged']=compare(bucket(*images),reference)
                assert all(r['exact'] for r in row['full_vs_staged'])
                del reference
            del original
            if args.compare_unfused:
                assert args.fused_edges and args.stage == 'full'
                unfused = NpuInferenceBucket(model,processed,fused_edges=False)
                old_graph = torch.compile(unfused,backend='npugraph_ex',fullgraph=True,
                                          dynamic=False,options=GRAPH_OPTIONS.copy())
                for _ in range(4):old_graph(*images)
                row['unfused_graph']=measure(lambda:old_graph(*images),3,10)
                del old_graph,unfused
                torch._dynamo.reset();gc.collect();torch.npu.empty_cache()
                save()
            if args.runner:
                assert args.stage == 'full'
                compiled=NpuGraphRunner(model,processed,fused_edges=args.fused_edges)
            else:
                compiled=torch.compile(bucket,backend='npugraph_ex',fullgraph=True,dynamic=False,
                    options=GRAPH_OPTIONS.copy())
            for _ in range(4): compiled(*images)
            torch.npu.synchronize()
            row['replay_checks']=[]
            for step in range(3):
                changed=tuple((x+step*.001).clone() for x in images)
                expected=bucket(*changed)
                actual=compiled(*changed)
                checks=compare(actual,expected)
                assert all(r['exact'] for r in checks)
                row['replay_checks'].append(checks);save()
                del actual,expected
            row['eager']=measure(lambda:bucket(*images),3,5)
            row['graph']=measure(lambda:compiled(*images),3,5)
            if args.trace:
                trace(lambda:compiled(*images),args.out/track,'fullgraph',2,False)
            row['status']='passed';save();print(track,row,flush=True)
            del compiled,bucket,images,processed,views
            torch._dynamo.reset()
        except Exception:
            row['status']='failed';row['error']=traceback.format_exc();save();raise

if __name__=='__main__': main()
