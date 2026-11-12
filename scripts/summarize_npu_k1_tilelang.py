"""Attribute TileLang direct-launch kernels by their unique compiled names.

The generic torch-npu profiler summary uses host-to-device flow links. Cython
TileLang launches have no such link here, so use the explicit kernel symbol,
check exactly one task per route per step, and retain the original summary.
"""
import argparse
import csv
import json
from pathlib import Path
from summarize_npu_profile import summarize

ap = argparse.ArgumentParser()
ap.add_argument('root', type=Path)
ap.add_argument('--out', required=True, type=Path)
args = ap.parse_args()
result = {}
for file in sorted(args.root.rglob('kernel_details.csv')):
    if file.parent.name != 'ASCEND_PROFILER_OUTPUT':
        continue
    summary = summarize(file.parent)
    kernels = list(csv.DictReader(file.open()))
    steps = summary['steps']
    explicit = {}
    for blocks in (20,40):
        key = f'K1::tilelang_rows_b{blocks}'
        selected = [r for r in kernels if f'meow_k1_tilelang_b{blocks}_kernel' in r['Name']]
        if selected:
            assert len(selected) == steps, (file, blocks, len(selected), steps)
            explicit[key] = {'kernel_ms': sum(float(r['Duration(us)']) for r in selected)/1000/steps,
                             'kernels_per_call': len(selected)/steps,
                             'attribution': 'unique compiled kernel symbol'}
    summary['tilelang_by_symbol'] = explicit
    result[str(file.parent)] = summary
    print(file.parent, {**summary['stage_kernel_ms'], **explicit})
assert result
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2))
