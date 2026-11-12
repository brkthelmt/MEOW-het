"""Summarize exported TileLang AIV counters; ratios are not additive wall-time shares."""
import argparse
import csv
import json
import re
from pathlib import Path

METRICS = ['Duration(us)', 'aiv_time(us)', 'aiv_vec_time(us)', 'aiv_vec_ratio',
           'aiv_scalar_time(us)', 'aiv_scalar_ratio', 'aiv_mte2_time(us)',
           'aiv_mte2_ratio', 'aiv_mte3_time(us)', 'aiv_mte3_ratio', 'aiv_icache_miss_rate']


def summarize(roots):
    result=[]
    for root in roots:
        for path in sorted(root.rglob('kernel_details.csv')):
            groups={}
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    name=row.get('Name','')
                    if re.search(r'meow_k[123]_tilelang',name):
                        groups.setdefault(name,[]).append(row)
            for name,rows in groups.items():
                item={'source_csv':str(path),'kernel':name,'samples':len(rows)}
                for metric in METRICS:
                    values=[]
                    for row in rows:
                        try:values.append(float(row[metric]))
                        except (ValueError,KeyError,TypeError):pass
                    item[metric]=sum(values)/len(values) if len(values)==len(rows) else None
                result.append(item)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--roots',nargs='+',type=Path,default=[Path('work/npu-k1/tilelang/vector/trace'),Path('work/npu-edges')])
    ap.add_argument('--out',type=Path,default=Path('work/npu-tilelang-profiling'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=summarize(args.roots)
    (args.out/'pipelines.json').write_text(json.dumps(rows,indent=2))
    with (args.out/'pipelines.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['source_csv','kernel','samples']+METRICS)
        writer.writeheader();writer.writerows(rows)
    print(f'{len(rows)} kernel/run groups written to {args.out}')

if __name__=='__main__':main()
