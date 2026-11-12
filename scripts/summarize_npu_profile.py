"""Summarize exported device kernels; exclude wait events and nested totals."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path


def summarize(directory):
    rows = list(csv.DictReader((directory / 'kernel_details.csv').open()))
    assert rows, f'No device kernels in {directory}'
    step_ids = {r['Step Id'] for r in rows if r['Step Id'] not in ('', 'N/A')}
    assert step_ids, 'Need scheduled profiler steps'
    steps = len(step_ids)
    groups = {}
    for column in ('Type', 'Name', 'Accelerator Core'):
        aggregate = defaultdict(lambda: {'ms_per_forward': 0., 'calls_per_forward': 0.})
        for row in rows:
            v = aggregate[row[column]]
            v['ms_per_forward'] += float(row['Duration(us)']) / 1000 / steps
            v['calls_per_forward'] += 1 / steps
        groups[column] = dict(sorted(aggregate.items(), key=lambda kv: -kv[1]['ms_per_forward']))

    trace = json.loads((directory / 'trace_view.json').read_text())
    events = trace if isinstance(trace, list) else trace['traceEvents']
    starts = {e['id']: e for e in events if e.get('cat') == 'async_npu' and e.get('ph') == 's'}
    ends = {(e['pid'], e['tid'], Decimal(str(e['ts']))): starts[e['id']]
            for e in events if e.get('cat') == 'async_npu' and e.get('ph') == 'f' and e['id'] in starts}
    labels = [e for e in events if e.get('name') in (
        'MEOW::encoder', 'MEOW::geometry_ar_fusion', 'MEOW::multiview_transformer',
        'MEOW::pano_heads', 'MEOW::heads', 'MEOW::geometry_mask_tensor')
        or e.get('name', '').startswith('K1::')]
    hardware_pids = {e['pid'] for e in events if e.get('ph') == 'M' and
                     e.get('name') == 'process_name' and e.get('args', {}).get('name') == 'Ascend Hardware'}
    kernels = {(str(e['args']['Task Id']), str(e['tid']), e['name']): e
               for e in events if e.get('pid') in hardware_pids and e.get('ph') == 'X'
               and 'Task Id' in e.get('args', {})}
    stage_ms = defaultdict(float)
    stage_calls = defaultdict(float)
    stage_types = defaultdict(lambda: defaultdict(lambda: {'ms_per_forward': 0., 'calls_per_forward': 0.}))
    for row in rows:
        event = kernels.get((row['Task ID'], row['Stream ID'], row['Name']))
        label = 'unmatched'
        if event:
            host = ends.get((event['pid'], event['tid'], Decimal(str(event['ts']))))
            if host:
                label = 'assembly_or_outside_label'
                for scope in labels:
                    if scope['tid'] == host['tid'] and Decimal(str(scope['ts'])) <= Decimal(str(host['ts'])) <= Decimal(str(scope['ts'])) + Decimal(str(scope['dur'])):
                        label = scope['name']
                        break
        stage_ms[label] += float(row['Duration(us)']) / 1000 / steps
        stage_calls[label] += 1 / steps
        stage_types[label][row['Type']]['ms_per_forward'] += float(row['Duration(us)']) / 1000 / steps
        stage_types[label][row['Type']]['calls_per_forward'] += 1 / steps

    operators = list(csv.DictReader((directory / 'operator_details.csv').open()))
    host_self_ms = defaultdict(float)
    for row in operators:
        host_self_ms[row['Name']] += float(row['Host Self Duration(us)']) / 1000 / steps
    checks = {}
    for name in ('aten::_local_scalar_dense', 'aten::linalg_solve', 'npu::npu_fusion_attention_v3'):
        selected = [r for r in operators if r['Name'] == name]
        checks[name] = {'calls_per_forward': len(selected) / steps,
                        'host_total_ms': sum(float(r['Host Total Duration(us)']) for r in selected) / 1000 / steps}
    kernel_sum = sum(v['ms_per_forward'] for v in groups['Type'].values())
    assert abs(sum(stage_ms.values()) - kernel_sum) < 1e-6
    return {'source': str(directory), 'steps': steps, 'kernels_per_forward': len(rows) / steps,
            'kernel_sum_ms': kernel_sum, 'groups': groups, 'stage_kernel_ms': dict(stage_ms),
            'stage_kernel_calls': dict(stage_calls),
            'stage_kernel_types': {stage: dict(sorted(values.items(), key=lambda kv: -kv[1]['ms_per_forward']))
                                   for stage, values in stage_types.items()},
            'host_self_ms': dict(sorted(host_self_ms.items(), key=lambda kv: -kv[1])[:30]),
            'operator_checks': checks,
            'note': 'Kernel durations exclude wait events. Sums are not wall time; stages use host-to-device flow attribution, not host scope duration.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = {}
    for root in args.roots:
        for path in sorted(root.rglob('kernel_details.csv')):
            if path.parent.name == 'ASCEND_PROFILER_OUTPUT':
                result[str(path.parent)] = summarize(path.parent)
    assert result, 'No kernel_details.csv exports found'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for value in result.values():
        print(value['source'], 'kernels/forward', value['kernels_per_forward'],
              'kernel sum ms', round(value['kernel_sum_ms'], 3))


if __name__ == '__main__':
    main()
