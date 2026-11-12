"""Profile the model's noncausal BNSD SDPA shape on Ascend."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from torch_npu.profiler import ProfilerActivity, profile

parser = argparse.ArgumentParser()
parser.add_argument('--length', type=int, default=10961)
parser.add_argument('--out', default='work/npu-validation/attention-trace.json')
parser.add_argument('--device-trace', action='store_true',
                    help='Include CANN device events (requires compatible profiler parser).')
args = parser.parse_args()
q, k, v = [torch.randn(1, 24, args.length, 64, device='npu', dtype=torch.bfloat16)
           for _ in range(3)]
for _ in range(3):
    torch.nn.functional.scaled_dot_product_attention(q, k, v)
torch.npu.synchronize()
activities = [ProfilerActivity.CPU]
if args.device_trace:
    activities.append(ProfilerActivity.NPU)
with profile(activities=activities, record_shapes=True) as prof:
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.npu.synchronize()
assert torch.isfinite(output).all()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
# TorchNPU's exporter appends to an existing file; each invocation is one trace.
Path(args.out).unlink(missing_ok=True)
prof.export_chrome_trace(args.out)
trace = json.loads(Path(args.out).read_text())
events = trace if isinstance(trace, list) else trace.get('traceEvents', [])
names = sorted({event.get('name', '') for event in events
                if any(s in event.get('name', '').lower() for s in ('attention', 'flash'))})
assert names, 'No attention events were captured'
print(json.dumps({'length': args.length, 'device_trace': args.device_trace,
                  'attention_events': names}, indent=2))
