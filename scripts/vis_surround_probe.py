"""Scientific sampling plots and eight-camera contact sheets for an E13 run."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashrecon.gen.trajectory import surrounding_plan
from dashrecon.provenance import git_commit

LAYOUT = [[-45., 0., 45.], [-90., None, 90.], [-135., 180., 135.]]


def contact(images: dict[float, Image.Image], label: str, frame: int) -> Image.Image:
    """Arrange the eight virtual directions spatially, with the rig at the centre."""
    width, height, caption = 384, 211, 24
    canvas = Image.new('RGB', (width * 3, (height + caption) * 3), '#171b20')
    draw = ImageDraw.Draw(canvas)
    for row, angles in enumerate(LAYOUT):
        for col, yaw in enumerate(angles):
            x, y = col * width, row * (height + caption)
            if yaw is None:
                draw.text((x + 20, y + 50), f'{label}\nframe {frame}\nFRONT-only reconstruction\nAll completion: generated\nYaw 0 = front, 180 = back', fill='white')
            else:
                canvas.paste(images[yaw].resize((width, height), Image.Resampling.LANCZOS), (x, y + caption))
                draw.text((x + 6, y + 6), f'yaw {yaw:+g} deg', fill='white')
    return canvas


def sampling(out: Path, cfg: dict) -> None:
    """Plot the planned query-camera sequence and virtual surrounding rig."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for side, label in [(1, 'Round 0: right/back'), (-1, 'Round 1: left/back')]:
        plan = surrounding_plan(cfg['anchors'], cfg['pan_views'], cfg['transfer_views'], side)
        axes[0].plot([v['yaw'] for v in plan], label=label)
        axes[1].plot([v['pose_frame'] for v in plan], label=label)
    axes[0].set_ylabel('Yaw (degrees)')
    axes[0].set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    axes[0].legend()
    axes[1].set_ylabel('Position along estimated FRONT trajectory (frame coordinate)')
    axes[1].set_xlabel('Query index within native 121-frame clip')
    axes[1].set_yticks(cfg['anchors'])
    for axis in axes:
        axis.grid(alpha=.25)
    fig.savefig(out / 'sampling_plan.png', dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.set_aspect('equal')
    for yaw in [-135, -90, -45, 0, 45, 90, 135, 180]:
        x, y = np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw))
        axis.arrow(0, 0, .75*x, .75*y, width=.015, head_width=.07, color='#3c83bf')
        axis.text(1.05*x, 1.05*y, f'{yaw:+g} deg', ha='center', va='center')
    axis.text(0, .16, 'FRONT', ha='center')
    axis.text(0, -.16, 'Virtual rig\nshared optical centre', ha='center', va='top')
    axis.set_xlim(-1.3, 1.3)
    axis.set_ylim(-1.3, 1.3)
    axis.axis('off')
    fig.savefig(out / 'virtual_rig.png', dpi=160)
    plt.close(fig)


def results(root: Path, out: Path, cfg: dict) -> None:
    """Use saved common-camera RGB and direct generation without re-rendering."""
    common = root / 'renders/common'
    cams = json.loads((common / 'cameras.json').read_text())
    generated = [json.loads((root / 'views' / f'r{r}' / 'cams.json').read_text())['cams'] for r in [0, 1]]
    for frame in cfg['anchors']:
        for model in ['E5c', 'E13_right_back_gen', 'E13_surround_gen']:
            images = {}
            for yaw in [a for row in LAYOUT for a in row if a is not None]:
                selected = [k for k, c in enumerate(cams) if c['frame'] == frame and c['yaw'] == yaw and c['right'] == 0]
                if len(selected) != 1:
                    raise ValueError(f'exactly one common camera required: {frame}, {yaw}')
                images[yaw] = Image.open(common / model / f'{selected[0]:03d}.png').convert('RGB')
            contact(images, model, frame).save(out / f'{model}_frame{frame}.png')
        images = {}
        for yaw in [a for row in LAYOUT for a in row if a is not None]:
            r = 1 if yaw < 0 else 0
            selected = [k for k, c in enumerate(generated[r]) if c['segment'] == f'anchor_{frame}' and abs(c['yaw']-yaw) < 1e-4]
            if len(selected) != 1:
                raise ValueError(f'exactly one generated camera required: {frame}, {yaw}')
            images[yaw] = Image.open(root / 'views' / f'r{r}' / 'filled' / f'{selected[0]:03d}.png').convert('RGB')
        contact(images, 'Direct generation (two clips)', frame).save(out / f'direct_generation_frame{frame}.png')


def main() -> None:
    """Build selected static visualizations from a frozen experiment snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['sampling', 'results'], required=True)
    args = parser.parse_args()
    commit = git_commit()
    if commit.endswith('-dirty'):
        raise ValueError('commit visualization code before generating artifacts')
    cfg = OmegaConf.to_container(OmegaConf.load(args.probe_dir / 'config.yaml'), resolve=True)
    out = args.probe_dir / 'renders' / 'overview'
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'sampling':
        sampling(out, cfg)
    else:
        results(args.probe_dir, out, cfg)
    (out / f'{args.mode}_meta.json').write_text(json.dumps({'dashrecon_commit':commit,'mode':args.mode,
        'source_experiment':str(args.probe_dir),'generative':True,'uses_oracle':False,
        'note':'contact sheets do not prove panorama seam consistency or validated scene geometry'},indent=2))
    print(out)


if __name__ == '__main__':
    main()
