"""Profile real MEOW inputs; keep latency measurement separate from tracing."""
import argparse
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch_npu

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / 'realset'))
from common import img_paths, load_tuples
from erp_detect import detect_full_erp
from meow_model import build_model_with_ar, load_ckpt_inplace
from mapanything.utils.image import load_images
from mapanything.utils.inference import (
    preprocess_input_views_for_inference, validate_input_views_for_inference,
)


def prepare(track, index):
    spec = load_tuples(track / 'tuples.json')
    spec['frames_root'] = str(track)
    tup = spec['tuples'][index]
    paths = img_paths(spec, tup)
    views = load_images(paths, verbose=False, resize_mode='fixed_squeeze')
    routing = []
    for view, path in zip(views, paths):
        with Image.open(path) as im:
            w, h = im.size
            pano, _ = detect_full_erp(np.asarray(im.convert('RGB')))
        view['aspect_ratio'] = torch.tensor([2.0 if pano else w / h])
        view['pano_wrap'] = bool(pano)
        routing.append(bool(pano))
        for key, value in list(view.items()):
            if torch.is_tensor(value):
                view[key] = value.to('npu')
    torch.npu.synchronize()
    return views, {'tuple_id': tup['id'], 'pano_flags': routing,
                   'image_shapes': [list(v['img'].shape) for v in views]}


@contextmanager
def stage_labels(model):
    # Instrument this profiling process only; do not modify model source/weights.
    targets = [
        (model, '_encode_n_views', 'encoder'),
        (model, '_encode_and_fuse_optional_geometric_inputs', 'geometry_ar_fusion'),
        (model.info_sharing, 'forward', 'multiview_transformer'),
        (model, '_dense_head_pano_wrapped', 'pano_heads'),
        (model, 'downstream_head', 'heads'),
        (model.dense_head, 'forward', 'dpt_dense'),
        (model.pose_head, 'forward', 'pose_head'),
        (model.scale_head, 'forward', 'scale_head'),
    ]
    originals = []
    for obj, attr, label in targets:
        fn = getattr(obj, attr)
        had_local = attr in obj.__dict__
        original_local = obj.__dict__.get(attr)
        originals.append((obj, attr, had_local, original_local))

        def wrap(fn, label):
            @functools.wraps(fn)
            def call(*args, **kwargs):
                with torch.autograd.profiler.record_function('MEOW::' + label):
                    return fn(*args, **kwargs)
            return call

        setattr(obj, attr, wrap(fn, label))
    try:
        yield
    finally:
        for obj, attr, had_local, original_local in reversed(originals):
            if had_local:
                setattr(obj, attr, original_local)
            else:
                delattr(obj, attr)


def measure(fn, warmup, iters):
    for _ in range(warmup):
        output = fn()
        del output
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    wall, stream = [], []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        output = fn()
        end.record()
        torch.npu.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        stream.append(start.elapsed_time(end))
        del output
    return {'wall_ms': wall, 'stream_span_ms': stream,
            'wall_mean_ms': float(np.mean(wall)), 'wall_p50_ms': float(np.median(wall)),
            'wall_p90_ms': float(np.percentile(wall, 90)),
            'stream_span_mean_ms': float(np.mean(stream)),
            'peak_allocated_gb': torch.npu.max_memory_allocated() / 1e9}


def trace(fn, out, label, steps, host_only):
    p = torch_npu.profiler
    existing_exports = set(out.rglob('kernel_details.csv'))
    acts = [p.ProfilerActivity.CPU]
    if not host_only:
        acts.append(p.ProfilerActivity.NPU)
    with p.profile(
        activities=acts,
        schedule=p.schedule(wait=0, warmup=1, active=steps, repeat=1),
        on_trace_ready=p.tensorboard_trace_handler(str(out)),
        record_shapes=True,
        experimental_config=p._ExperimentalConfig(
            profiler_level=p.ProfilerLevel.Level1,
            aic_metrics=p.AiCMetrics.PipeUtilization,
            export_type=p.ExportType.Text, data_simplification=False),
    ) as prof:
        for _ in range(steps + 1):
            with torch.autograd.profiler.record_function('MEOW::' + label):
                output = fn()
                torch.npu.synchronize()
            del output
            prof.step()
    if not host_only:
        import csv
        exports = set(out.rglob('kernel_details.csv')) - existing_exports
        assert exports, 'Profiler returned without exporting device kernels; inspect its logs'
        for path in exports:
            with path.open() as stream:
                assert next(csv.DictReader(stream), None), f'Empty kernel export: {path}'
    print(f'TRACE {label} -> {out}', flush=True)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--tracks', nargs='+', default=['blk_mixed'])
    ap.add_argument('--tuple-index', type=int, default=0)
    ap.add_argument('--out', type=Path, default=Path('work/npu-profile/model'))
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--profile-steps', type=int, default=2)
    ap.add_argument('--host-only', action='store_true')
    ap.add_argument('--skip-trace', action='store_true')
    ap.add_argument('--trace-infer', action='store_true')
    ap.add_argument('--postprocess-backend', choices=['numpy', 'tensor'],
                    default=os.environ.get('MEOW_GEOMETRY_POSTPROCESS', 'numpy'))
    ap.add_argument('--python-profile', action='store_true',
                    help='Also collect cProfile for one warmed infer call, outside timing.')
    args = ap.parse_args()
    os.environ['MEOW_GEOMETRY_POSTPROCESS'] = args.postprocess_backend
    assert args.warmup >= 1 and args.iters >= 1 and args.profile_steps >= 1
    args.out.mkdir(parents=True, exist_ok=True)
    model = build_model_with_ar('npu', 'wrap')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert load_ckpt_inplace(model, ckpt.get('model', ckpt)) == 0
    del ckpt
    print('CHECKPOINT loaded strictly', flush=True)
    result = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'device': torch.npu.get_device_name(0),
              'visible_devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
              'checkpoint': args.checkpoint, 'warmup': args.warmup, 'iters': args.iters,
              'precision': 'BF16 autocast, original FP32 regions',
              'postprocess_backend': args.postprocess_backend, 'tracks': {}}
    for track in args.tracks:
        views, metadata = prepare(args.data_root / track, args.tuple_index)
        processed = preprocess_input_views_for_inference(validate_input_views_for_inference(views))
        # Match infer's deterministic configuration. RGB-only views contain no GT geometry.
        def core():
            model._configure_geometric_input_config(True, True, True, True, True)
            try:
                with torch.autocast('npu', dtype=torch.bfloat16):
                    return model(processed, memory_efficient_inference=False)
            finally:
                model._restore_original_geometric_input_config()

        def infer():
            return model.infer(views, memory_efficient_inference=False,
                               use_amp=True, amp_dtype='bf16', apply_mask=True, mask_edges=True)

        # Core is deliberately measured first, so fallback warnings cannot be
        # suppressed by an earlier infer call in the same process.
        print(f'BEGIN CORE {track} {metadata}', flush=True)
        metadata['core'] = measure(core, args.warmup, args.iters)
        print(f'CORE {track} {metadata["core"]}', flush=True)
        print(f'BEGIN INFER {track}', flush=True)
        metadata['infer'] = measure(infer, args.warmup, args.iters)
        print(f'INFER {track} {metadata["infer"]}', flush=True)
        result['tracks'][track] = metadata
        (args.out / 'latency.json').write_text(json.dumps(result, indent=2))
        if args.python_profile:
            import cProfile
            import pstats
            capture = cProfile.Profile()
            capture.enable()
            output = infer()
            torch.npu.synchronize()
            capture.disable()
            del output
            capture.dump_stats(str(args.out / f'{track}-python.prof'))
            with (args.out / f'{track}-python.txt').open('w') as stream:
                pstats.Stats(capture, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(80)
        if not args.skip_trace:
            with stage_labels(model):
                trace(core, args.out / track / 'core', 'core_forward', args.profile_steps, args.host_only)
                if args.trace_infer:
                    trace(infer, args.out / track / 'infer', 'infer', args.profile_steps, args.host_only)
        del processed, views
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
