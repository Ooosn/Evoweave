#!/usr/bin/env python3
"""Render a small, deterministic set of saved XYZ-fill evaluation examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def skeleton(ax, joints, parents, color):
    points = np.asarray(joints, dtype=np.float32)
    if points.ndim != 2 or points.shape != (len(parents), 3) or not np.isfinite(points).all():
        raise ValueError('saved skeleton is not a finite unpadded joint/parent pair')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=7, c=color, depthshade=False)
    for child, parent in enumerate(parents):
        if parent is None or parent == -1:
            continue
        if not 0 <= parent < len(points) or child == parent:
            raise ValueError('saved skeleton contains an invalid parent')
        bone = points[[parent, child]]
        ax.plot(bone[:, 0], bone[:, 1], bone[:, 2], color=color, linewidth=1.4)


def load_rows(evaluation, group, fit_step=None):
    evaluation = Path(evaluation)
    if fit_step is not None:
        if fit_step < 0:
            raise ValueError('fit step must be nonnegative')
        paths = sorted(evaluation.glob(f'fit_eval_{fit_step:04d}_rank_*.json'))
        if not paths:
            raise ValueError('no saved fixed-fit evidence for the requested step')
        reports = [json.loads(path.read_text()) for path in paths]
        return [row for report in reports for row in report['rows']], 'fixed_fit'
    reports = [json.loads(path.read_text()) for path in sorted(evaluation.glob('rank_*.json'))]
    if not reports or not all(report.get('complete') for report in reports):
        raise ValueError('visualization requires complete saved evaluation ranks')
    return [row for report in reports for row in report['groups'][group]['rows']], group


def render(evaluation, destination, *, group='valid', assets=2, fit_step=None):
    rows, group = load_rows(evaluation, group, fit_step)
    counts = {row['asset_id']: len(row['original_target_joints']) for row in rows}
    keys = sorted(counts)
    if not keys:
        raise ValueError('evaluation group is empty')
    # Include a non-52-joint case when available instead of showing only the common template.
    order = [key for key in keys if counts[key] != 52] + [key for key in keys if counts[key] == 52]
    if assets >= 2 and any(counts[key] == 52 for key in keys):
        first = order[0]
        common = next(key for key in keys if counts[key] == 52)
        order = list(dict.fromkeys([first, common] + order))
    chosen = order[:assets]
    selected = []
    for key in chosen:
        matching = sorted([row for row in rows if row['asset_id'] == key], key=lambda row: row['frame_index'])
        selected.extend(matching if len(matching) <= 2 else [matching[0], matching[-1]])
    figure = plt.figure(figsize=(12, 3.4 * len(selected)), constrained_layout=True)
    colors = ('#15803d', '#2563eb', '#ea580c')
    for index, row in enumerate(selected):
        anchors = np.asarray(row['input_anchor_points'], dtype=np.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 3 or not np.isfinite(anchors).all():
            raise ValueError('saved input mesh anchors are invalid')
        views = (
            ('GT', row['original_target_joints'], row['original_target_parents']),
            (f'GT structure + XYZ fill | L1 {row["gt_structure_xyz_l1"]:.3f}',
             row['gt_structure_predicted_joints'], row['gt_structure_predicted_parents']),
            (f'Generated structure + XYZ fill | F1 {row["original_edge_f1"]:.3f}',
             row['predicted_joints'], row['predicted_parents']),
        )
        for column, (title, joints, parents) in enumerate(views):
            ax = figure.add_subplot(len(selected), 3, index * 3 + column + 1, projection='3d')
            ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], s=1.5, c='#737373', alpha=.2, depthshade=False)
            if joints is not None:
                skeleton(ax, joints, parents, colors[column])
            else:
                ax.text2D(.05, .5, row['status'], transform=ax.transAxes, color='#b91c1c', fontsize=10)
            ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), zlim=(-1.05, 1.05))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-60)
            ax.set_axis_off()
            ax.set_title(title, fontsize=9, pad=1)
            if column == 0:
                ax.text2D(.01, .01, f'{row["asset_id"]}\nraw frame {row["raw_frame_index"]}',
                          transform=ax.transAxes, fontsize=8)
    suffix = '' if fit_step is None else f' | update {fit_step}'
    figure.suptitle(f'{group}: fixed geometry examples{suffix}', fontsize=13)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, facecolor='white')
    plt.close(figure)
    destination.with_suffix('.json').write_text(json.dumps({
        'evaluation': str(evaluation), 'group': group, 'fit_step': fit_step,
        'selection_rule': 'fixed non-52/common-template examples, first and last scored frames',
        'rows': [{key: row[key] for key in ('asset_id', 'frame_index', 'raw_frame_index', 'original_edge_f1',
                                          'gt_structure_xyz_l1', 'status')} for row in selected]}, indent=2) + '\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', default='valid')
    parser.add_argument('--assets', type=int, default=2)
    parser.add_argument('--fit-step', type=int)
    arguments = parser.parse_args()
    if arguments.assets < 1:
        parser.error('--assets must be positive')
    if arguments.fit_step is not None and arguments.fit_step < 0:
        parser.error('--fit-step must be nonnegative')
    print(render(arguments.evaluation, arguments.output, group=arguments.group,
                 assets=arguments.assets, fit_step=arguments.fit_step))
