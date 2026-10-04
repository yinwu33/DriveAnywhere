"""E34 (docs/E34_DATA.md): contact sheet of finished pairs for visual checks: per row one target, columns
input of each available type | real target with the loss mask shaded red.

Example (any venv):
    python scripts/e34_contact_sheet.py --pairs_dir data/e34/pairs/wt646 --cams 1 2 3 4 --times 40 100 160 \
        --width 512 --out data/e34/vis/wt646.jpg
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

TYPES = ("points", "gs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--cams", type=int, nargs="+", required=True)
    parser.add_argument("--times", type=int, nargs="+", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    types = [t for t in TYPES if os.path.isdir(os.path.join(args.pairs_dir, t))]
    rows = []
    for t in args.times:
        for c in args.cams:
            name = f"{t:03d}_{c}"
            tgt = os.path.join(args.pairs_dir, "target", f"{name}.jpg")
            assert os.path.exists(tgt), f"no target {tgt}"
            target = np.asarray(Image.open(tgt).convert("RGB")).astype(np.float32)
            mask = np.asarray(Image.open(os.path.join(args.pairs_dir, "mask", f"{name}.png"))) > 127
            shaded = target.copy()
            shaded[mask] = 0.5 * shaded[mask] + 0.5 * np.array([255.0, 0.0, 0.0])
            cells = [Image.open(os.path.join(args.pairs_dir, typ, f"{name}.jpg")).convert("RGB") for typ in types]
            cells.append(Image.fromarray(shaded.astype(np.uint8)))
            h = int(round(cells[0].size[1] * args.width / cells[0].size[0]))
            row = Image.new("RGB", (args.width * len(cells), h))
            for i, cell in enumerate(cells):
                row.paste(cell.resize((args.width, h), Image.BILINEAR), (i * args.width, 0))
            ImageDraw.Draw(row).text((4, 4), f"t={t} cam={c}  " + " | ".join(types + ["target"]), fill=(255, 255, 0))
            rows.append(row)
    sheet = Image.new("RGB", (rows[0].size[0], sum(r.size[1] for r in rows)))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.size[1]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out, quality=88)
    print(f"[e34_contact_sheet] {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
