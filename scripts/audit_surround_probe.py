"""Read-only artifact and per-direction audit of a completed E13 experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from omegaconf import OmegaConf


def read(path: Path) -> dict:
    """Load a recorded JSON object."""
    return json.loads(path.read_text())


def main() -> None:
    """Check saved evidence and report directions without inventing quality scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.probe_dir
    state, meta = read(root / 'run_status.json'), read(root / 'meta.json')
    cfg = OmegaConf.to_container(OmegaConf.load(root / 'config.yaml'), resolve=True)
    if not state['completed'] or any(s['exit_code'] != 0 for s in state['stages']):
        raise ValueError('completed stages required')
    for source, expected in meta['input_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError(f'input changed: {source}')
    if meta['uses_oracle'] or not meta['generative'] or not meta['generator_weights_frozen']:
        raise ValueError('unexpected input/model provenance')
    audit = {'experiment_commit': meta['dashrecon_commit'],
             'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'round_code_commits': meta['round_code_commits'],
             'kind': 'saved artifact integrity and generated self-consistency; no GT scoring',
             'input_hashes_verified': True, 'rounds': [], 'directions': []}
    scales = []
    input_rows = []
    for r in [0, 1]:
        views = root / 'views' / f'r{r}'
        validation = root / 'validation' / f'r{r}'
        camera_meta, cache, depth, fill = [read(views / p) for p in ['cams.json', 'cache.json', 'depth.json', 'fill.json']]
        report = read(validation / 'validation.json')
        source_cfg = OmegaConf.load(Path(camera_meta['log_dir']) / 'config.yaml').data
        candidates = sorted({sc['frame'] for c in camera_meta['cams'] for sc in c['src']})
        train_sources = [f for f in candidates if f not in cache['skipped_heldout_source_frames']]
        image_dir = Path(fill['image_dir'])
        mask_dir = Path(source_cfg.pixel_source.mask_dir)
        for frame in train_sources:
            paths = [image_dir / f'{frame:03d}_0.jpg', views / 'src' / f'{frame:03d}.npy',
                     *[mask_dir / f'mask_{kind}' / f'{frame:06d}.png' for kind in ['dynamic', 'sky']]]
            input_rows.append({'round': r, 'frame': frame, 'hashes': {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
        for p in [camera_meta, cache, depth, fill, report]:
            if p['dashrecon_commit'] != meta['round_code_commits'][f'r{r}']:
                raise ValueError('undocumented reconstruction commit')
        cams = camera_meta['cams']
        if len(cams) != 121 or len(cache['frames']) != 121 or len(report['frames']) != 121:
            raise ValueError('native 121-frame records required')
        if fill['params']['num_steps'] != 35 or fill['params']['seed'] != 1:
            raise ValueError('unexpected generation budget')
        sizes = {}
        for sub in ['rgb', 'mask', 'filled']:
            paths = sorted((views / sub).glob('*.png'))
            if [p.name for p in paths] != [f'{k:03d}.png' for k in range(121)]:
                raise ValueError(f'missing/extra {sub} images')
            for p in paths:
                with Image.open(p) as im:
                    if im.size != (1280, 704):
                        raise ValueError(f'wrong image resolution: {p}')
            sizes[sub] = len(paths)
        for sub in ['depth', 'count', 'opacity', 'filled_depth']:
            paths = sorted((views / sub).glob('*.npy'))
            if [p.name for p in paths] != [f'{k:03d}.npy' for k in range(121)]:
                raise ValueError(f'missing/extra {sub} arrays')
            for p in paths:
                array = np.load(p)
                if array.shape != (704, 1280) or not np.isfinite(array).all() or (array < 0).any():
                    raise ValueError(f'invalid saved array: {p}')
            sizes[sub] = len(paths)
        for k, row in enumerate(report['frames']):
            conf = np.load(validation / 'confidence' / f'{k:03d}.npy')
            hole = np.asarray(Image.open(views / 'mask' / f'{k:03d}.png')) > 127
            if not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any() or (conf[~hole] > 0).any():
                raise ValueError(f'invalid confidence in {r}/{k}')
            if int((conf > 0).sum()) != row['accepted_pixels']:
                raise ValueError('confidence/report mismatch')
        scales.append(depth['alignment']['scene_scale'])
        selected = [report['frames'][k] for k in cfg['training_view_indices']]
        selected_holes = sum(row['hole_pixels'] for row in selected)
        if selected_holes == 0:
            raise ValueError('surrounding training pool has no holes to evaluate')
        selected_accepted = sum(row['accepted_pixels'] for row in selected)
        audit['rounds'].append({'round': r, 'artifacts': sizes, 'confidence_arrays': 121,
            'static_real_candidate_coverage_before_seed_override': float(np.mean([f['real_coverage'] for f in cache['frames']])),
            'first_conditioning_frame_full_front': True,
            'actual_memory_cache_mean': cache['memory_coverage'],
            'training_targets_this_round': len(selected),
            'accepted_hole_pixels_in_training_targets': selected_accepted,
            'accepted_fraction_in_training_targets': selected_accepted / selected_holes,
            'accepted_pixels_outside_training_targets': sum(row['accepted_pixels'] for row in report['frames']) - selected_accepted,
            'views_with_memory': sum(f['memory_pixels'] > 0 for f in cache['frames']),
            'skipped_heldout_source_frames': cache['skipped_heldout_source_frames'],
            'scene_scale': scales[-1], 'scale_reused': depth['alignment']['scale_reused']})
        for k, (cam, evidence, cached) in enumerate(zip(cams, report['frames'], cache['frames'])):
            if cam['segment'] != f"anchor_{cam['frame']}" or abs(cam['yaw']) not in [0, 45, 90, 135, 180]:
                continue
            audit['directions'].append({'round': r, 'frame': cam['frame'], 'yaw': cam['yaw'], 'index': k,
                'input_hole_fraction': camera_meta['hole_frac'][k], 'real_cache_fraction': cached['real_coverage'],
                'memory_cache_fraction': cached['memory_coverage'],
                **{key: evidence[key] for key in ['hole_pixels', 'accepted_pixels', 'accepted_fraction',
                   'unknown_pixels', 'memory_supported_pixels', 'memory_conflict_pixels']}})
    if scales[0] != scales[1] or not audit['rounds'][1]['scale_reused']:
        raise ValueError('one scene scale must persist across rounds')
    if audit['rounds'][0]['actual_memory_cache_mean'] != 0 or audit['rounds'][1]['actual_memory_cache_mean'] <= 0:
        raise ValueError('second-round validated memory must actually reach generator')
    common = root / 'renders' / 'common'
    cameras = json.loads((common / 'cameras.json').read_text())
    if len(cameras) != 54:
        raise ValueError('54 common cameras required')
    for model in ['E5c', 'E13_right_back_gen', 'E13_surround_gen']:
        for k, cam in enumerate(cameras):
            path = common / model / f'{k:03d}.png'
            with Image.open(path) as im:
                im.verify()
            array = np.load(common / model / f'{k:03d}_depth.npy')
            if not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f'invalid common depth: {model}/{k}')
    audit['common_rgb_and_depth_pairs'] = 162
    audit['source_hash_records'] = len(input_rows)
    (root / 'source_input_hashes.json').write_text(json.dumps(input_rows, indent=2))
    (root / 'artifact_audit.json').write_text(json.dumps(audit, indent=2))
    lines = ['# E13 per-direction evidence', '',
        'All coverage is input cache coverage, not post-fit quality. Acceptance is generated self-consistency inside input holes.', '',
        '| Round | Frame | Yaw | Input holes | Real cache | Memory cache | Accepted within holes |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in audit['directions']:
        values = [100 * row[k] for k in ['input_hole_fraction', 'real_cache_fraction', 'memory_cache_fraction', 'accepted_fraction']]
        lines.append(f"| {row['round']} | {row['frame']} | {row['yaw']:+g} | " + ' | '.join(f'{v:.2f}%' for v in values) + ' |')
    (root / 'DIRECTION_METRICS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: audit[k] for k in ['input_hashes_verified', 'rounds', 'common_rgb_and_depth_pairs']}, indent=2))


if __name__ == '__main__':
    main()
