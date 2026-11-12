"""Summarize graph latency and trace evidence, without treating links as copies."""
import argparse
import collections
import json
from pathlib import Path


def summarize_trace(path):
    data=json.loads(path.read_text())
    events=data if isinstance(data,list) else data['traceEvents']
    durations=collections.Counter()
    counts=collections.Counter()
    for e in events:
        if e.get('ph')!='X':continue
        name=e.get('name','')
        counts[name]+=1
        durations[name]+=e.get('dur',0)
    return {'path':str(path),'replay_calls':counts['acl_graph_replay'],
            'model_execute_calls':counts['MODEL_EXECUTE'],
            'tilelang_k1_calls':sum(v for k,v in counts.items() if 'meow_k1_tilelang' in k),
            'tilelang_k2_calls':sum(v for k,v in counts.items() if 'meow_k2_tilelang' in k),
            'tilelang_k3_calls':sum(v for k,v in counts.items() if 'meow_k3_tilelang' in k),
            'host_scalar_or_solve':{k:v for k,v in counts.items() if '_local_scalar_dense' in k or 'linalg_solve' in k},
            'copy_events':{k:v for k,v in counts.items() if 'MEMCPY' in k or 'Memcpy' in k},
            'top_duration_us':durations.most_common(20)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    result={'traces':[summarize_trace(p) for p in args.root.rglob('trace_view.json')]}
    result['runs']={}
    for p in args.root.rglob('results.json'):
        data=json.loads(p.read_text())
        if not isinstance(data,dict):continue
        rows=data.get('tracks',data)
        result['runs'][str(p)]={k:{'status':v.get('status'),
            'eager_ms':v.get('eager',{}).get('wall_mean_ms'),
            'graph_ms':v.get('graph',{}).get('wall_mean_ms')} for k,v in rows.items() if isinstance(v,dict)}
    (args.root/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
