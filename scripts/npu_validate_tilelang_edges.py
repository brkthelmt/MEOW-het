"""Strict geometry fusion checks against existing NPU Tensor operations."""
import argparse
import json
import math
from pathlib import Path
import torch
import torch_npu
from mapanything.utils.geometry_tensor import points_to_normals_tensor, normals_edge_tensor, depth_edge_tensor
from operators.k1_tilelang.kernel import pointmap_to_normals
from operators.geometry_tilelang.kernels import normal_candidates, mask_and_apply
from npu_profile_model import measure, trace


def check(p,c,d,m):
    n,v=pointmap_to_normals(p,m)
    e=normal_candidates(n,v)
    f,po,co,do,zo=mask_and_apply(e,d,m,p,c)
    ne=normals_edge_tensor(n,v)
    er=torch.nn.functional.max_pool2d(e.float().unsqueeze(1),3,1,1).squeeze(1)>0
    torch.testing.assert_close(er,ne,atol=0,rtol=0)
    ref=m & ~(ne & depth_edge_tensor(d,m))
    refs=(ref,p*ref.unsqueeze(-1),c*ref.unsqueeze(-1),d*ref,c[...,2]*ref)
    for a,r in zip((f,po,co,do,zo),refs):
        torch.testing.assert_close(a.cpu(),r.cpu(),atol=0,rtol=0,equal_nan=True)
    return n,v,e


def adversarial():
    cases=[]
    for h,w in ((1,1),(1,33),(17,1),(17,33)):
        y,x=torch.meshgrid(torch.arange(h,device='npu'),torch.arange(w,device='npu'),indexing='ij')
        p=torch.stack((x.float(),y.float(),torch.ones_like(x,dtype=torch.float32)),dim=-1).unsqueeze(0).repeat(2,1,1,1)
        c=p.clone();d=torch.ones((2,h,w),device='npu');m=torch.ones_like(d,dtype=torch.bool)
        check(p,c,d,m);cases.append([h,w,'plane'])
        m.zero_();check(p,c,d,m);cases.append([h,w,'empty'])
        m.fill_(True)
        p.reshape(-1)[0]=float('nan');c.reshape(-1)[-1]=float('inf')
        d.reshape(-1)[0]=float('nan');d.reshape(-1)[-1]=0
        check(p,c,d,m);cases.append([h,w,'nonfinite'])
    # Exercise asymmetric validity, extreme depth and threshold-near normals.
    b,h,w=2,17,33
    p=torch.randn(b,h,w,3,device='npu');c=torch.randn_like(p)
    values=torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,-1.,1.,1.03,1.0300001],device='npu')
    d=values[torch.arange(b*h*w,device='npu')%len(values)].reshape(b,h,w).contiguous()
    m=(torch.arange(b*h*w,device='npu').reshape(b,h,w)%5!=0)
    check(p,c,d,m);cases.append([h,w,'depth_extremes'])
    for angle in [0.,4.99999,5.,5.00001,90.,180.]:
        n=torch.zeros(b,h,w,3,device='npu');n[...,2]=1
        n[:,:,w//2:,0]=math.sin(math.radians(angle))
        n[:,:,w//2:,2]=math.cos(math.radians(angle))
        for special in (False,True):
            if special:
                n[0,0,0,0]=float('nan');n[1,-1,-1,1]=float('inf');n[:,5,6,:]=0
            e=normal_candidates(n,m)
            actual=torch.nn.functional.max_pool2d(e.float().unsqueeze(1),3,1,1).squeeze(1)>0
            torch.testing.assert_close(actual,normals_edge_tensor(n,m),atol=0,rtol=0)
        cases.append([h,w,'angle',angle])
    # Isolate K3's relative-depth cutoff, with normal edges deliberately true.
    for step in (1.0299999,1.03,1.0300001):
        d=torch.ones(b,h,w,device='npu');d[:,:,w//2:]=step
        m=torch.ones(b,h,w,device='npu',dtype=torch.bool)
        candidate=torch.ones_like(m)
        actual=mask_and_apply(candidate,d,m,p,c)
        f=m & ~depth_edge_tensor(d,m)
        refs=(f,p*f.unsqueeze(-1),c*f.unsqueeze(-1),d*f,c[...,2]*f)
        for a,r in zip(actual,refs):
            torch.testing.assert_close(a.cpu(),r.cpu(),atol=0,rtol=0,equal_nan=True)
        cases.append(['depth_threshold',step])
    current, other = torch.npu.current_stream(), torch.npu.Stream()
    other.wait_stream(current)
    with torch.npu.stream(other):
        p=torch.randn(b,h,w,3,device='npu');c=torch.randn_like(p)
        d=torch.rand(b,h,w,device='npu')+1;m=torch.rand(b,h,w,device='npu')>.1
        n,v=pointmap_to_normals(p,m)
        result=mask_and_apply(normal_candidates(n,v),d,m,p,c)
    current.wait_stream(other)
    n,v=points_to_normals_tensor(p,m)
    f=m & ~(normals_edge_tensor(n,v)&depth_edge_tensor(d,m))
    refs=(f,p*f.unsqueeze(-1),c*f.unsqueeze(-1),d*f,c[...,2]*f)
    for actual,ref in zip(result,refs):
        torch.testing.assert_close(actual.cpu(),ref.cpu(),atol=0,rtol=0,equal_nan=True)
    cases.append('nondefault_stream')
    return cases


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--small',action='store_true');ap.add_argument('--trace',action='store_true')
    ap.add_argument('--out',type=Path,default=Path('work/npu-edges/results.json'))
    args=ap.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    rows=[{'adversarial':adversarial()}]
    print('ADVERSARIAL PASSED',flush=True)
    shapes=[(2,17,33)] if args.small else [(b,h,518) for b in (1,4) for h in (252,392,518)]
    for b,h,w in shapes:
        p=torch.randn(b,h,w,3,device='npu');c=torch.randn_like(p)
        d=torch.rand(b,h,w,device='npu')+1;m=torch.rand(b,h,w,device='npu')>.2
        n,v,e=check(p,c,d,m)
        row={'shape':[b,h,w],'random':'exact'}
        if not args.small:
            row['k2']=measure(lambda:normal_candidates(n,v),3,10)
            row['k3']=measure(lambda:mask_and_apply(e,d,m,p,c),3,10)
        rows.append(row);args.out.write_text(json.dumps(rows,indent=2));print(row,flush=True)
    if args.small:return
    for track in ['blk_mixed','blk_erp','blk_pinhole','blk_fisheye']:
        data=torch.load(f'work/npu-postprocess/raw/{track}_0.pt',map_location='cpu',weights_only=False)['raw']
        p,c,d,m=[torch.cat([r[key] for r in data],dim=0).to('npu').contiguous() for key in ['pts3d','pts3d_cam','depth_along_ray','non_ambiguous_mask']]
        d=d.squeeze(-1).contiguous();n,v,e=check(p,c,d,m)
        def fused():
            n,v=pointmap_to_normals(p,m)
            e=normal_candidates(n,v)
            return mask_and_apply(e,d,m,p,c)
        def reference():
            n,v=points_to_normals_tensor(p,m)
            f=m & ~(normals_edge_tensor(n,v)&depth_edge_tensor(d,m))
            return f,p*f.unsqueeze(-1),c*f.unsqueeze(-1),d*f,c[...,2]*f
        row={'track':track,'exact':True,'fused':measure(fused,3,10),'tensor':measure(reference,3,10)}
        rows.append(row);args.out.write_text(json.dumps(rows,indent=2));print(row,flush=True)
        if args.trace and track=='blk_mixed':trace(fused,args.out.parent/'trace','K123_tilelang',2,False)

if __name__=='__main__':main()
