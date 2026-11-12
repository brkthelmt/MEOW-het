"""Device kernel durations for K2/K3 at six model shapes (separate from wall time)."""
import csv
import json
from pathlib import Path
import torch
import torch_npu
from operators.k1_tilelang.kernel import pointmap_to_normals
from operators.geometry_tilelang.kernels import normal_candidates,mask_and_apply
from npu_profile_model import trace

@torch.inference_mode()
def main():
    root=Path('work/npu-edges/kernels');root.mkdir(parents=True,exist_ok=True)
    results=[]
    for b in (1,4):
        for h in (252,392,518):
            w=518
            p=torch.randn(b,h,w,3,device='npu');c=torch.randn_like(p)
            m=torch.rand(b,h,w,device='npu')>.2;d=torch.rand(b,h,w,device='npu')+1
            n,v=pointmap_to_normals(p,m)
            def fn():return mask_and_apply(normal_candidates(n,v),d,m,p,c)
            for _ in range(3):fn()
            out=root/f'b{b}_h{h}_w{w}'
            before=set(out.rglob('kernel_details.csv'))
            trace(fn,out,'K23',2,False)
            files=set(out.rglob('kernel_details.csv'))-before
            times={'k2':[],'k3':[]}
            for path in files:
                for row in csv.DictReader(path.open()):
                    for key in times:
                        if f'meow_{key}_tilelang' in row['Name']:
                            times[key].append(float(row['Duration(us)']))
            assert all(len(v)==2 for v in times.values()),times
            results.append({'shape':[b,h,w],'kernel_us':times,
                            'kernel_mean_us':{k:sum(v)/len(v) for k,v in times.items()}})
            (root/'results.json').write_text(json.dumps(results,indent=2))
            print(results[-1],flush=True)

if __name__=='__main__':main()
