#!/usr/bin/env python3
"""Generated 4-frame strips -> one 8-direction sprite sheet.

    python3 tools/build_strip_sprites.py --character mouse --animation walk
    python3 tools/build_strip_sprites.py --character cat   --animation trapped

Same output contract as the video pipeline — 32 frames, one 4x8 sheet, one metadata file
— but a different kind of source. Video has to be searched for its four phases, which is
most of what build_cat_sprites.py does. A strip is four already-chosen phases in one
image, so there is no phase to find and only a much smaller problem left: cutting the
strip apart in the right places.

Drawing all four phases in ONE image is the point, not a convenience. Four separately
generated images would be four slightly different characters; inside a single image the
generator holds proportions, colour and style together, which is the same reason the
cat's walk came out of one continuous video rather than five stills.

Only the five left-facing and axis-aligned directions are generated. RIGHT, DOWN_RIGHT
and UP_RIGHT are exact horizontal mirrors, as everywhere else here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

C8 = np.ones((3, 3), bool)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from catlib import sheet as SH                       # noqa: E402
from catlib.frames import geometry, place, band_profile, best_shift  # noqa: E402
from catlib.matte import cutout                      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'

# Who each character is on disk and in the sheet's file names.
CHARACTERS = {
    'cat': dict(folder='cat', hero='tom'),
    'mouse': dict(folder='mouse', hero='jerry'),
}

DEFAULTS = {
    'frameSize': 256, 'fps': 8, 'charHeight': 0.80, 'groundY': 0.90, 'anchorX': 0.5,
    'source': {d: d + '.png' for d in SH.SOURCES},
    'scale': {}, 'idle': {}, 'split': {},
    # 'torso' keeps the body still and lets the feet move, which is what a walk wants.
    # 'ground' keeps whatever is on the floor still — for the trapped strips that is the
    # trap, and it is the only fixed thing in the shot.
    'pivot': 'torso',
    # Multiplies the character's size on screen without touching how the frames are laid
    # out. Kept apart from `scale` on purpose: `scale` changes the art inside the frame
    # and the metadata then measures the result, so `scale` alone can never change the
    # rendered size — it cancels itself out.
    'renderScale': 1.0,
}


def log(*a):
    print(*a, flush=True)


def split_strip(alpha: np.ndarray, want: int = 4, override=None) -> list[tuple[int, int]]:
    """Column ranges of the four drawn frames.

    Cut on the empty columns between the mice rather than on even quarters: the generator
    does not space them evenly, and a quarter cut that clips a tail or a lifted foot
    turns into a missing limb three steps later, where it looks like a matte bug.
    """
    if override:
        edges = [0] + list(override) + [alpha.shape[1]]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


    col = (alpha > 0.5).sum(axis=0)
    occupied = col > max(2, int(0.002 * alpha.shape[0]))
    lab, n = ndi.label(occupied)
    runs = [(int(np.where(lab == i)[0].min()), int(np.where(lab == i)[0].max()))
            for i in range(1, n + 1)]
    # join fragments that are really one mouse (a tail tip detached by a thin gap)
    merged, gap = [], max(4, int(0.012 * alpha.shape[1]))
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    merged.sort(key=lambda r: col[r[0]:r[1] + 1].sum(), reverse=True)
    keep = sorted(merged[:want])

    # Two neighbouring poses sometimes touch — a thrown-out arm or a whipping tail closes
    # the gap — and then one run holds two frames. Split the widest run at the thinnest
    # column inside it rather than at its midpoint, so the cut lands in the pinch between
    # the two characters instead of through one of them.
    while len(keep) < want:
        i = max(range(len(keep)), key=lambda k: keep[k][1] - keep[k][0])
        a, b = keep[i]
        lo, hi = a + int(0.25 * (b - a)), a + int(0.75 * (b - a))
        if hi <= lo:
            raise SystemExit('cannot split the strip into %d frames — set "split" in the config'
                             % want)
        cut = lo + int(np.argmin(col[lo:hi + 1]))
        keep[i:i + 1] = [(a, cut), (cut + 1, b)]
        keep.sort()
    if len(keep) != want:
        raise SystemExit('strip split found %d frames, not %d — set "split" in the config'
                         % (len(keep), want))
    # widen each range to the midpoint of the gap so nothing is shaved off
    out = []
    for i, (a, b) in enumerate(keep):
        lo = 0 if i == 0 else (keep[i - 1][1] + a) // 2
        hi = alpha.shape[1] if i == want - 1 else (b + keep[i + 1][0]) // 2
        out.append((lo, hi))
    return out


def _only_this_pose(alpha: np.ndarray, keep: float = 0.15) -> np.ndarray:
    """Drop anything in the cut-out that belongs to the neighbouring pose.

    The cut falls in the gap between two drawings, but the gap is not always empty — a
    thrown-out arm or a whipping tail from the pose next door can reach across it, and
    it then arrives as a fragment floating beside the character. The figure itself is one
    connected piece, trap included, because the jaw is closed on its foot; a fragment big
    enough to be a real limb rather than a leak has to be a sixth of the body.
    """
    m = alpha > 0.5
    lab, n = ndi.label(m, structure=C8)
    if n <= 1:
        return alpha
    sizes = ndi.sum(m, lab, np.arange(1, n + 1))
    biggest = sizes.max()
    wanted = np.isin(lab, [i for i in range(1, n + 1) if sizes[i - 1] >= keep * biggest])
    return np.where(wanted, alpha, 0.0)


def build(cfg: dict, src: pathlib.Path, out: pathlib.Path, hero: str, animation: str,
          config_name: str) -> dict:
    size = int(cfg['frameSize'])
    canvases, report = {}, {}

    cut = {}
    for d in SH.SOURCES:
        path = src / cfg['source'][d]
        if not path.exists():
            raise SystemExit('missing %s — save the %s %s strip there first'
                             % (path, d, animation))
        rgb, alpha = cutout(np.asarray(Image.open(path).convert('RGB')), keep_all=True)
        spans = split_strip(alpha, override=cfg['split'].get(SH.CAMEL[d]))
        frames = []
        for a, b in spans:
            sub_a, sub_rgb = alpha[:, a:b], rgb[:, a:b]
            if (sub_a > 0.5).sum() < 500:
                raise SystemExit('empty frame in %s' % path)
            sub_a = _only_this_pose(sub_a)
            frames.append((sub_rgb, sub_a, geometry(sub_a)))
        cut[d] = frames
        log('%-10s %s  ->  4 frames at x %s' % (d, path.name, [s[0] for s in spans]))

    # One character size across all eight directions. There is no camera drift to undo
    # here — a strip is one drawing — so each direction is simply scaled by the median
    # height of its own four frames, which leaves the walk's own bounce alone.
    med = {d: float(np.median([g['h'] for _, _, g in cut[d]])) for d in SH.SOURCES}
    target = float(np.median(list(med.values())))
    log('median heights %s  -> common %.0f px' % ({k: int(v) for k, v in med.items()}, target))

    for d in SH.SOURCES:
        key = SH.CAMEL[d]
        base = cfg['charHeight'] * size / med[d] * cfg['scale'].get(key, 1.0)
        # With the ground pivot, line the frames up on the thing that is genuinely
        # identical between them — the trap — rather than on the centroid of everything
        # touching the floor. On the cat that distinction matters: his free paw rests on
        # the ground beside the trap in some poses and not others, and the centroid
        # follows it, sliding the trap 15px across the cycle.
        # A walk anchors on the torso and lets the feet move; taking the feet here
        # instead makes the body swing with every step, which is what a walk cycle is.
        pivots = [g['torso_cx'] for _, _, g in cut[d]]
        if cfg.get('pivot') == 'ground':
            pivots = [g['foot_cx'] for _, _, g in cut[d]]
            profs = [band_profile(al, g) for _, al, g in cut[d]]
            # Each frame's own rough anchor, corrected by however far its floor profile
            # sits from the first frame's. Correcting frame 0's anchor instead would
            # throw every frame by the difference between the two anchors.
            pivots = [p + best_shift(profs[0], pr) for p, pr in zip(pivots, profs)]

        frames, notes = [], []
        for (rgb, alpha, geo), pivot in zip(cut[d], pivots):
            frames.append(place(rgb, alpha, geo, base, size, cfg['anchorX'], cfg['groundY'],
                                pivot_x=pivot))
            notes.append(dict(heightPx=geo['h'], groundPx=geo['ground'],
                              torsoCx=round(geo['torso_cx'], 1),
                              pivotX=round(pivot, 1), scale=round(base, 4)))
        canvases[d] = frames
        report[key] = notes

    SH.mirror_all(canvases)
    for dst, src in SH.MIRROR.items():
        report[SH.CAMEL[dst]] = {'mirrorOf': SH.CAMEL[src]}

    stem = '%s-%s' % (hero, animation)
    meta = SH.write(canvases, out, stem + '.png', stem + '.json', size,
                    cfg['fps'], {'x': cfg['anchorX'], 'y': cfg['groundY']}, cfg['idle'],
                    {'strips': cfg['source'], 'config': config_name},
                    frames_dir=animation, render_scale=cfg.get('renderScale', 1.0))
    (BUILD / 'qa').mkdir(parents=True, exist_ok=True)
    (BUILD / 'qa' / (stem + '-build-report.json')).write_text(json.dumps(report, indent=2) + '\n')
    log('sheet %dx%d -> %s' % (size * 4, size * len(SH.ROWS), out / meta['image']))
    return dict(canvases=canvases, meta=meta, size=size, stem=stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--character', required=True, choices=sorted(CHARACTERS))
    ap.add_argument('--animation', required=True)
    a = ap.parse_args()

    who = CHARACTERS[a.character]
    src = ROOT / 'source-animation' / who['folder'] / a.animation
    out = ROOT / 'app' / 'assets' / who['folder']
    config = ROOT / ('%s-%s.config.json' % (a.character, a.animation))

    cfg = dict(DEFAULTS)
    if config.exists():
        cfg.update(json.loads(config.read_text()))
    else:
        config.write_text(json.dumps(cfg, indent=2) + '\n')
        log('wrote %s' % config)

    res = build(cfg, src, out, who['hero'], a.animation, config.name)
    SH.qa(res['canvases'], BUILD / 'qa', res['size'], cfg['fps'], prefix=res['stem'] + '-')
    log('QA written to %s' % (BUILD / 'qa'))


if __name__ == '__main__':
    main()
