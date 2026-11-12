"""Compare encoder/transformer/raw outputs to locate numerical graph gaps."""
import json
from pathlib import Path
import torch
import torch_npu
from meow_model import build_model_with_ar,load_ckpt_inplace
from mapanything.utils.npu_graph import NpuCoreBucket,GRAPH_OPTIONS

class Probe(torch.nn.Module):
    def __init__(self,inner,info=False):
        super().__init__();self.inner=inner;self.info=info
    def forward(self,*args,**kwargs):
        out=self.inner(*args,**kwargs)
        self.last=torch.cat(out[0].features,dim=0) if self.info else out.features
        return out

class Diagnostic(torch.nn.Module):
    def __init__(self,model,views):
        super().__init__()
        model.encoder=Probe(model.encoder)
        model.info_sharing=Probe(model.info_sharing,True)
        self.core=NpuCoreBucket(model,views)
    def forward(self,*images):
        raw=self.core(*images)
        return dict(raw[0],encoder=self.core.model.encoder.last,info=self.core.model.info_sharing.last)

@torch.inference_mode()
def main():
    model=build_model_with_ar('npu','wrap')
    state=torch.load('checkpoints/MEOW-sB25.pth',map_location='cpu',weights_only=False)
    assert load_ckpt_inplace(model,state.get('model',state))==0
    del state
    model._configure_geometric_input_config(True,True,True,True,True)
    data=torch.load('work/npu-postprocess/raw/blk_mixed_0.pt',map_location='cpu',weights_only=False)
    views=[{k:v.to('npu') if torch.is_tensor(v) else v for k,v in r.items()} for r in data['processed']]
    diag=Diagnostic(model,views)
    images=tuple(v['img'] for v in views)
    ref={k:v.cpu().clone() for k,v in diag(*images).items()}
    graph=torch.compile(diag,backend='npugraph_ex',fullgraph=True,dynamic=False,options=GRAPH_OPTIONS.copy())
    for _ in range(4):actual=graph(*images)
    rows={}
    for key,a in actual.items():
        a=a.cpu();b=ref[key]
        err=(a.float()-b.float()).abs()
        rows[key]={'max_abs':float(err.max()),'mean_abs':float(err.mean()),'exact_fraction':float((a==b).float().mean())}
    print(rows,flush=True)
    Path('work/npu-graph/diagnostic.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
