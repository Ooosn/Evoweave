"""Matched-camera scientific figures from saved predictions, without inference."""
import argparse
import html
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

LABELS = ['OPT300k', 'OPT330k', 'native250k', 'native300k', 'GT']
TITLES = {'OPT300k': 'OPT 300k', 'OPT330k': 'OPT 330k', 'native250k': 'Native 250k',
          'native300k': 'Native 300k', 'GT': 'Ground Truth'}
COLORS = {'OPT300k': '#657ba0', 'OPT330k': '#285c9d', 'native250k': '#d65b4c',
          'native300k': '#a82f2f', 'GT': '#21866b'}
AXES = [0, 2, 1]


def read(path):
    return json.loads(path.read_text())


def skeleton(example, frame, label):
    if label == 'GT':
        return np.asarray(example['truth'][frame]), example['parents'].tolist(), None
    row = example['reference'] if frame == 0 else next(r for r in example['rows']
        if r['method'] == 'predicted_head' and r['target_slot'] == frame)
    f1 = row['original_edge_f1'] if frame == 0 else row['edge_f1']
    if not row['success']:
        return None, None, f1
    return np.asarray(row['predicted_joints']), row['predicted_parents'], f1


def extent(points, skeletons):
    values = [points] + [xyz for xyz, _, _ in skeletons if xyz is not None]
    all_xyz = np.concatenate(values)[:, AXES]
    lo, hi = all_xyz.min(0), all_xyz.max(0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) * .55, 1e-3)
    return center, radius


def draw(ax, points, pose, label, bounds):
    center, radius = bounds
    points = points[:, AXES]
    ax.scatter(*points.T, s=1., c='#99a3ad', alpha=.32, depthshade=False,
        linewidths=0, rasterized=True)
    xyz, parents, f1 = pose
    if xyz is not None:
        xyz = xyz[:, AXES]
        for child, parent in enumerate(parents):
            if parent is not None and int(parent) >= 0:
                edge = xyz[[int(parent), child]]
                ax.plot(*edge.T, color=COLORS[label], lw=1.45, alpha=.97)
        ax.scatter(*xyz.T, s=5.5, color=COLORS[label], depthshade=False, linewidths=0)
    else:
        ax.text2D(.5, .12, 'No valid skeleton', ha='center', transform=ax.transAxes,
            fontsize=9, color='#b3261e')
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - radius, c + radius)
    ax.view_init(elev=18, azim=-60)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if f1 is not None:
        count = 0 if xyz is None else len(xyz)
        ax.text2D(.5, .015, f'F1 {100*f1:.1f}% | Joints {count}', ha='center', transform=ax.transAxes,
            fontsize=8, color='#343434')
    elif xyz is not None:
        ax.text2D(.5, .015, f'Joints {len(xyz)}', ha='center', transform=ax.transAxes,
            fontsize=8, color='#343434')


def figure_case(models, asset, case, path):
    frames = [0, 2, 4, 7]
    fig = plt.figure(figsize=(15, 10.7), facecolor='white')
    for row, frame in enumerate(frames):
        sample = models['native300k'][asset]
        points = np.asarray(sample.get('visual_points', sample['points'])[frame])
        step = max(1, len(points) // 2200)
        points = points[::step]
        poses = [skeleton(models[label][asset] if label != 'GT' else sample, frame, label) for label in LABELS]
        bounds = extent(points, poses)
        for col, (label, pose) in enumerate(zip(LABELS, poses)):
            ax = fig.add_subplot(4, 5, row * 5 + col + 1, projection='3d', proj_type='ortho')
            draw(ax, points, pose, label, bounds)
            if row == 0:
                ax.set_title(TITLES[label], fontsize=11, color=COLORS[label], pad=2)
            if col == 0:
                ax.text2D(-.04, .5, 'Reference 0' if frame == 0 else f'Target {frame}',
                    transform=ax.transAxes, fontsize=10, rotation=90, va='center')
    fig.suptitle(f'Case {case}  |  {asset}', fontsize=14, y=.983)
    fig.text(.5, .01, 'Identical input tensors, camera and scale per row. No alignment or fitting. '
        'Reference: 15-beam generation; targets: predicted-reference head.', ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.025, right=.995, top=.93, bottom=.04, wspace=.015, hspace=.04)
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


def overview(models, assets, cases, frame, path):
    labels = ['OPT330k', 'native250k', 'native300k', 'GT']
    fig = plt.figure(figsize=(13, 11.3), facecolor='white')
    for row, case in enumerate(cases):
        asset = assets[case]
        sample = models['native300k'][asset]
        points = np.asarray(sample.get('visual_points', sample['points'])[frame])[::2]
        poses = [skeleton(models[label][asset] if label != 'GT' else sample, frame, label) for label in labels]
        bounds = extent(points, poses)
        for col, (label, pose) in enumerate(zip(labels, poses)):
            ax = fig.add_subplot(len(cases), 4, row * 4 + col + 1, projection='3d', proj_type='ortho')
            draw(ax, points, pose, label, bounds)
            if row == 0:
                ax.set_title(TITLES[label], fontsize=12, color=COLORS[label], pad=2)
            if col == 0:
                ax.text2D(-.04, .5, f'Case {case}', transform=ax.transAxes,
                    fontsize=11, rotation=90, va='center')
    fig.suptitle('Fixed Hard Cases: ' + ('Reference Pose' if frame == 0 else 'Target Pose 7'), fontsize=15, y=.985)
    fig.text(.5, .008, 'Same camera and bounds within each row; no alignment, fitting, or output repair.',
        ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.03, right=.995, top=.93, bottom=.035, wspace=.02, hspace=.04)
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)


def pct(value, digits=2):
    return 'N/A' if value is None else f'{100*value:.{digits}f}%'


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence, figures = args.output / 'evidence', args.output / 'figures'
    evidence.mkdir(exist_ok=False)
    figures.mkdir(exist_ok=False)
    with tarfile.open(args.archive, 'r:gz') as handle:
        handle.extractall(evidence, filter='data')
    config, review = read(evidence / 'config.json'), read(evidence / 'comparison_review.json')
    if not review['passed'] or review['optimizer_updates'] != 0:
        raise ValueError('only complete reviewed predictions may be rendered')
    models = {}
    declared_cases = dict(zip(config['evaluation']['asset_ids'], config['evaluation']['cases']))
    for label in LABELS[:-1]:
        records = []
        for path in (evidence / 'models' / label).glob('examples_rank_*.pt'):
            records.extend(torch.load(path, map_location='cpu', weights_only=True))
        models[label] = {r['asset_id']: r for r in records}
        if set(models[label]) != set(config['evaluation']['asset_ids']) or len(records) != 12:
            raise ValueError('visual coverage differs')
        if any(r['panel_index'] != declared_cases[r['asset_id']] for r in records):
            raise ValueError('original case indices differ')
    assets = dict(zip(config['evaluation']['cases'], config['evaluation']['asset_ids']))
    for case, asset in assets.items():
        figure_case(models, asset, case, figures / f'case_{case:03d}.png')
    for frame in (0, 7):
        overview(models, assets, [30, 70, 105, 108], frame, figures / f'hardcases_pose{frame}.png')
    matrix = read(evidence / 'development_matrix.json')
    matched = matrix['strictly_paired_scopes']['native_candidates_vs_OPT300k']
    dev_rows = [[TITLES[k], pct(v['stage1_edge_F1']), pct(v['GT_MPJPE'], 4), pct(v['GT_edge_F1'])]
        for k, v in matched['metrics'].items()]
    common = review['all4_identical_common_success_geometry']
    official_rows = []
    for label, value in review['models'].items():
        head = value['summaries']['predicted_head']
        official_rows.append([TITLES[label], pct(value['reference_edge_F1']), pct(head['edge_f1']),
            pct(common['normalized_J2J'][label], 4), pct(value['summaries']['GT_head']['normalized_mpjpe_known_id'], 4)])
    asset108 = assets[108]
    old_all = review['models']['OPT330k']['summaries']['predicted_head']['normalized_j2j_successful']
    old108 = review['per_case']['OPT330k'][asset108]['predicted_head']['normalized_j2j_successful']
    sensitivity = {}
    for label in ('native250k', 'native300k'):
        new_all = review['models'][label]['summaries']['predicted_head']['normalized_j2j_successful']
        new108 = review['per_case'][label][asset108]['predicted_head']['normalized_j2j_successful']
        sensitivity[label] = {'case108_fraction_of_all12_J2J_improvement':
            ((old108 - new108) / 12) / (old_all - new_all),
            'other11_OPT330k_J2J': (old_all * 12 - old108) / 11,
            'other11_native_J2J': (new_all * 12 - new108) / 11}
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Evoweave Native Qwen Comparison</title><style>body{font:15px/1.6 system-ui,sans-serif;color:#222;margin:0;background:#fff}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:right;padding:9px;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}img{width:100%;height:auto}p{max-width:1100px}a{color:#285c9d}.scope{color:#555}</style><main>',
        '<h1>Native Qwen: Fixed Checkpoint Comparison</h1>',
        '<p>Native 250k and 300k use one checkpoint for both stages. OPT 300k/330k results are retained baselines, not regenerated. No training or model promotion was performed.</p>',
        f'<h2>Strict Development Matrix ({matched["assets"]} Assets)</h2>',
        table(['Model', 'Greedy Skeleton F1', 'GT-Reference MPJPE', 'GT-Reference Edge F1'], dev_rows),
        '<p class="scope">All eight raw frames, normalization, surface tensors and original GT match. MPJPE is normalized by mesh bounding-box diagonal. These GT-reference and greedy scores are not end-to-end 15-beam scores. Including the older 330k adaptation panel leaves only 19 strictly matched assets.</p>',
        '<h2>Formal Fixed 12 Cases</h2>',
        table(['Model', 'Reference F1 (12)', 'Predicted-Reference Target F1 (84)', 'Matched Target J2J', 'GT-Reference MPJPE'], official_rows),
        f'<p class="scope">F1 includes every failed generation as zero. J2J uses the identical {len(common["asset_ids"])} common-success assets / {common["views"]} target views across all four models, without alignment. GT-reference controls are separate. Fixed descriptive cases are not an untouched population test. The old case105 exact-token replay discrepancy remains unresolved.</p>',
        '<h2>Reading the Tradeoff</h2><p>The native 250k candidate improves case30 substantially, but neither native checkpoint uniformly beats the working 330k. Case70 still misses major branches; case105 has better edge matching but worse joint positions. Case108 overgenerates 102/131 joints at native250k/300k versus only 5 GT joints, with edge F1 zero. A lower nearest-joint distance is not sufficient evidence of a correct skeleton.</p>',
        f'<p>Post-hoc decomposition, without changing the primary 12-case denominator: case108 contributes {100*sensitivity["native250k"]["case108_fraction_of_all12_J2J_improvement"]:.1f}% of native250k\'s aggregate J2J reduction versus OPT330k. Across the other eleven cases, mean J2J is {pct(sensitivity["native250k"]["other11_OPT330k_J2J"],4)} versus {pct(sensitivity["native250k"]["other11_native_J2J"],4)}. This diagnostic explains the aggregate; it is not a replacement benchmark or a new checkpoint selection rule.</p>',
        '<h2>Reference Skeletons</h2><img src="figures/hardcases_pose0.png" alt="Reference skeleton comparisons for cases 30,70,105,108">',
        '<h2>Target Pose 7</h2><img src="figures/hardcases_pose7.png" alt="Target pose comparisons for cases 30,70,105,108">',
        '<h2>All 12 Cases</h2>']
    case_rows = []
    for case, asset in assets.items():
        case_rows.append([case, *[pct(review['per_case'][label][asset]['predicted_head']['edge_f1']) for label in LABELS[:-1]]])
    parts.append(table(['Case', *[TITLES[k] + ' Target F1' for k in LABELS[:-1]]], case_rows))
    for case, asset in assets.items():
        parts.append(f'<h2>Case {case}</h2><p>{html.escape(asset)}</p><a href="figures/case_{case:03d}.png"><img src="figures/case_{case:03d}.png" alt="Case {case}, four matched poses and all model variants"></a>')
    parts.append('</main></html>')
    report = args.output / 'comparison_report.html'
    report.write_text('\n'.join(parts), encoding='utf-8')
    manifest = {'complete': True, 'figures': len(list(figures.glob('*.png'))),
        'cases': list(assets), 'display_frames': [0, 2, 4, 7],
        'camera': {'elevation': 18, 'azimuth': -60, 'projection': 'orthographic'},
        'same_camera_and_bounds_across_models_per_pose': True, 'alignment_or_fitting': False,
        'report': str(report.resolve()), 'source_review': str((evidence / 'comparison_review.json').resolve()),
        'posthoc_geometry_contribution_not_a_new_primary_denominator': sensitivity}
    (args.output / 'visual_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
