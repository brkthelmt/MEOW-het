"""Compare Tensor geometry filters against the original NumPy implementation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapanything.utils.geometry import points_to_normals, normals_edge, depth_edge
from mapanything.utils.geometry_tensor import points_to_normals_tensor, normals_edge_tensor, depth_edge_tensor


def compare_geometry(points, depth, mask, label):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ref_n, ref_nm, ref_ne, ref_de = [], [], [], []
        for p, d, m in zip(points, depth, mask):
            n, nm = points_to_normals(p, mask=m)
            ref_n.append(n); ref_nm.append(nm)
            ref_ne.append(normals_edge(n, tol=5.0, mask=nm))
            ref_de.append(depth_edge(d, rtol=0.03, mask=m))
    ref_n, ref_nm, ref_ne, ref_de = map(np.stack, (ref_n, ref_nm, ref_ne, ref_de))
    output = {'label': label, 'pixels': int(mask.size)}
    for device in ('cpu', 'npu'):
        p, d, m = [torch.from_numpy(x).to(device) for x in (points, depth, mask)]
        n, nm = points_to_normals_tensor(p, m)
        ne = normals_edge_tensor(n, nm)
        de = depth_edge_tensor(d, m)
        n, nm, ne, de = [x.cpu().numpy() for x in (n, nm, ne, de)]
        finite = np.isfinite(n) & np.isfinite(ref_n)
        errors = {
            'normal_max_abs': float(np.abs(n[finite] - ref_n[finite]).max(initial=0)),
            'normal_nonfinite_mismatch': int((np.isfinite(n) != np.isfinite(ref_n)).sum()),
            'normal_mask_mismatch': int((nm != ref_nm).sum()),
            'normal_edge_mismatch': int((ne != ref_ne).sum()),
            'depth_edge_mismatch': int((de != ref_de).sum()),
            'final_mask_mismatch': int(((mask & ~(ne & de)) != (mask & ~(ref_ne & ref_de))).sum()),
        }
        output[device] = errors
    return output


def synthetic():
    rng = np.random.default_rng(42)
    cases = []
    for h, w in ((1, 1), (2, 3), (17, 23)):
        y, x = np.mgrid[:h, :w].astype(np.float32)
        p = np.stack((x / 20, y / 20, np.ones_like(x)), -1)[None]
        d = np.linalg.norm(p, axis=-1)
        m = np.ones(d.shape, bool)
        cases.append((p, d, m, f'plane_{h}x{w}'))
        cases.append((p, d, np.zeros_like(m), f'empty_{h}x{w}'))
    p = rng.normal(size=(2, 19, 27, 3)).astype(np.float32)
    p[..., 2] += 3
    m = rng.random(p.shape[:-1]) > .15
    d = np.linalg.norm(p, axis=-1)
    cases.append((p.copy(), d.copy(), m.copy(), 'random_holes'))
    p[:, 3, 4] = np.nan; p[:, 6, 8] = np.inf
    d[:, 2, 5] = 0; d[:, 8, 4] = np.nan; d[:, 9, 5] = np.inf
    cases.append((p, d, m, 'nonfinite'))
    return [compare_geometry(*case) for case in cases]


def summarize_validation(result):
    cases = result['synthetic'] + [v for row in result['real'] for v in row['views']]
    summary = {
        'cpu_reference_exact': all(all(value == 0 for value in c['cpu'].values()) for c in cases),
        'npu_normal_max_abs': max(c['npu']['normal_max_abs'] for c in cases),
        'npu_normal_finite_pattern_exact': all(c['npu']['normal_nonfinite_mismatch'] == 0 for c in cases),
        'npu_normal_validity_exact': all(c['npu']['normal_mask_mismatch'] == 0 for c in cases),
        'npu_depth_edges_exact': all(c['npu']['depth_edge_mismatch'] == 0 for c in cases),
        'npu_final_mask_exact': all(c['npu']['final_mask_mismatch'] == 0 for c in cases)
            and all(row['final_mask_mismatch'] == 0 for row in result['real']),
    }
    summary['status'] = 'exact_masks' if summary['npu_final_mask_exact'] else 'mask_precision_gap'
    return summary


def check_mask_options(raw, processed, postprocess):
    checked = []
    for options in ({'apply_mask': False}, {'mask_edges': False},
                    {'mask_edges': False, 'apply_confidence_mask': True}):
        outputs = []
        for backend in ('numpy', 'tensor'):
            os.environ['MEOW_GEOMETRY_POSTPROCESS'] = backend
            outputs.append(postprocess(raw, processed, **options))
        for ref, actual in zip(*outputs):
            assert ref.keys() == actual.keys()
            for key in ref:
                if torch.is_tensor(ref[key]):
                    torch.testing.assert_close(ref[key].cpu(), actual[key].cpu(),
                                               atol=0, rtol=0, equal_nan=True)
        checked.append(options)
    return checked


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint')
    ap.add_argument('--data-root', type=Path)
    ap.add_argument('--tracks', nargs='+', default=['blk_mixed', 'blk_erp', 'blk_pinhole', 'blk_fisheye'])
    ap.add_argument('--limit', type=int, default=1)
    ap.add_argument('--cache-raw', type=Path, help='Cache raw outputs for identical-input numerical debugging')
    ap.add_argument('--strict-mask', action='store_true',
                    help='Fail if any NPU final mask differs; otherwise report the gap explicitly')
    ap.add_argument('--out', type=Path, default=Path('work/npu-postprocess/validation.json'))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {'synthetic': synthetic(), 'real': []}
    for case in result['synthetic']:
        print(json.dumps(case), flush=True)
    if args.checkpoint:
        from npu_profile_model import prepare
        from meow_model import build_model_with_ar, load_ckpt_inplace
        from mapanything.utils.inference import (validate_input_views_for_inference,
            preprocess_input_views_for_inference, postprocess_model_outputs_for_inference)
        model = None
        if args.cache_raw:
            args.cache_raw.mkdir(parents=True, exist_ok=True)
        for track in args.tracks:
            spec = json.loads((args.data_root / track / 'tuples.json').read_text())
            count = min(args.limit, len(spec['tuples'])) if args.limit else len(spec['tuples'])
            for index in range(count):
                cached = args.cache_raw / f'{track}_{index}.pt' if args.cache_raw else None
                if cached and cached.exists():
                    bundle = torch.load(cached, map_location='cpu', weights_only=False)
                    meta = bundle['metadata']
                    raw, processed = [[{k:v.to('npu') if torch.is_tensor(v) else v for k,v in view.items()}
                                       for view in bundle[name]] for name in ('raw', 'processed')]
                    views = None
                    del bundle
                else:
                    if model is None:
                        model = build_model_with_ar('npu', 'wrap')
                        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                        assert load_ckpt_inplace(model, ckpt.get('model', ckpt)) == 0
                        del ckpt
                    views, meta = prepare(args.data_root / track, index)
                    processed = preprocess_input_views_for_inference(validate_input_views_for_inference(views))
                    model._configure_geometric_input_config(True, True, True, True, True)
                    with torch.autocast('npu', dtype=torch.bfloat16):
                        raw = model(processed, memory_efficient_inference=False)
                    model._restore_original_geometric_input_config()
                    if cached:
                        torch.save({'metadata':meta, **{name:[{k:v.cpu() if torch.is_tensor(v) else v for k,v in view.items()}
                                    for view in seq] for name,seq in [('raw',raw),('processed',processed)]}}, cached)
                torch.npu.synchronize()
                comparisons = []
                for i, v in enumerate(raw):
                    comparisons.append(compare_geometry(
                        v['pts3d'].float().cpu().numpy(), v['depth_along_ray'].squeeze(-1).float().cpu().numpy(),
                        v['non_ambiguous_mask'].cpu().numpy(), f'{meta["tuple_id"]}/view{i}'))
                outputs, timings = {}, {}
                for backend in ('numpy', 'tensor'):
                    os.environ['MEOW_GEOMETRY_POSTPROCESS'] = backend
                    torch.npu.synchronize(); t0 = time.perf_counter()
                    outputs[backend] = postprocess_model_outputs_for_inference(raw, processed)
                    torch.npu.synchronize(); timings[backend] = (time.perf_counter() - t0) * 1000
                wrong, pixels = 0, 0
                for a, b in zip(outputs['numpy'], outputs['tensor']):
                    wrong += int((a['mask'] != b['mask']).sum().cpu())
                    pixels += a['mask'].numel()
                    # Mask changes are counted separately; compare values on the common valid set.
                    both = (a['mask'] & b['mask']).squeeze(-1)
                    for key in ('pts3d', 'pts3d_cam', 'depth_along_ray', 'depth_z'):
                        torch.testing.assert_close(a[key][both].cpu(), b[key][both].cpu(), atol=0, rtol=0, equal_nan=True)
                row = {'tuple_id': meta['tuple_id'], 'pixels': pixels, 'final_mask_mismatch': wrong,
                       'cold_postprocess_ms': timings, 'views': comparisons,
                       'exact_option_checks': check_mask_options(raw, processed, postprocess_model_outputs_for_inference)}
                result['real'].append(row)
                args.out.write_text(json.dumps(result, indent=2))
                print(json.dumps({k:v for k,v in row.items() if k != 'views'}), flush=True)
                del raw, outputs, views, processed
    result['summary'] = summarize_validation(result)
    args.out.write_text(json.dumps(result, indent=2))
    print('Saved', args.out, flush=True)
    print(json.dumps(result['summary']), flush=True)
    assert result['summary']['cpu_reference_exact'], 'CPU Tensor differs from NumPy reference'
    if args.strict_mask:
        assert result['summary']['npu_final_mask_exact'], 'NPU mask precision gap; see JSON report'


if __name__ == '__main__':
    main()
