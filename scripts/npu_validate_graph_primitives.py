"""Locate eager/AOT numerical differences before accepting full-model capture."""
import json
from pathlib import Path
import torch
import torch_npu
import torch.nn.functional as F
from mapanything.utils.npu_graph_ops import bicubic, bilinear

@torch.inference_mode()
def main():
    x=torch.randn(4,1037,1536,device='npu')
    linear=torch.nn.Linear(1536,4608,device='npu').eval()
    norm=torch.nn.LayerNorm(1536,device='npu').eval()
    qkv=tuple(torch.randn(4,24,1037,64,device='npu',dtype=torch.bfloat16) for _ in range(3))
    pos=torch.randn(1,1536,37,37,device='npu')
    rays=torch.randn(4,3,392,518,device='npu')
    conv_input=torch.randn(4,96,28,37,device='npu')
    conv=torch.nn.Conv2d(96,256,3,padding=1,device='npu').eval()
    deconv=torch.nn.ConvTranspose2d(96,96,4,stride=4,device='npu').eval()
    cases=[('conv_fp32',conv,(conv_input,)),('deconv_fp32',deconv,(conv_input,)),('normalize',lambda p:F.normalize(p,dim=1),(rays,)),
           ('bilinear',lambda p:F.interpolate(p,size=(56,74),mode='bilinear',align_corners=True),(pos,)),
           ('bilinear_opaque',lambda p:bilinear(p,56,74,True),(pos,)),('bicubic_opaque',lambda p:bicubic(p,28,37,28.1/37,37.1/37,False),(pos,)),('linear',linear,(x,)),('layernorm',norm,(x,)),
           ('sdpa',F.scaled_dot_product_attention,qkv),
           ('bicubic',lambda p:F.interpolate(p,mode='bicubic',scale_factor=(28.1/37,37.1/37),antialias=False),(pos,))]
    rows=[]
    for name,fn,args in cases:
        def wrapped(*inputs):
            with torch.autocast('npu',dtype=torch.bfloat16,enabled=not name.endswith('_fp32')):return fn(*inputs)
        eager=wrapped(*args).cpu()
        graph=torch.compile(wrapped,backend='npugraph_ex',fullgraph=True,dynamic=False,
                            options={'pattern_fusion_pass':False,'inplace_pass':False,'input_inplace_pass':False,'remove_cat_ops':False})
        for _ in range(4):actual=graph(*args)
        actual=actual.cpu()
        row={'name':name,'dtype':str(actual.dtype),'max_abs':float((actual-eager).abs().max()),
             'exact_fraction':float((actual==eager).float().mean())}
        rows.append(row);print(row,flush=True)
        torch._dynamo.reset()
    Path('work/npu-graph/primitives.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
