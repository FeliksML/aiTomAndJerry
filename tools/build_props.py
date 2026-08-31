#!/usr/bin/env python3
"""The two painted props — the trap and the mouse hole — into one small atlas.

    python3 tools/build_props.py

The characters get eight directions because they walk; a prop sits on its cell and is
looked at from the one camera the whole arena is drawn from, so it needs one row and no
mirrors. Everything else is deliberately the same as the character pipeline: the same
white-background matte, the same ground-line placement, and metadata carrying the
character height so paint.js sizes by the object rather than by the padding around it.

    source-animation/props/trap.png   four states in one strip: set, glint, snap, shut
    source-animation/props/hole.png   one drawing
    ->  app/assets/props/props.png    a 5-frame atlas
        app/assets/props/props.json   frame index, anchor, and height per prop
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_strip_sprites import split_strip                      # noqa: E402
from catlib.frames import geometry, place, band_profile, best_shift  # noqa: E402
from catlib.matte import cutout                                  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / 'source-animation' / 'props'
OUT = ROOT / 'app' / 'assets' / 'props'
BUILD = ROOT / 'build' / 'qa'

SIZE = 256
GROUND_Y = 0.86
ANCHOR_X = 0.5

# The trap's four states, in the order the strip draws them. `set` and `glint` are the
# same trap with a highlight moved, so a caller can shimmer an armed trap without
# animating it; `snap` and `shut` are the two frames of it closing.
TRAP_STATES = ['trapSet', 'trapGlint', 'trapSnap', 'trapShut']


def log(*a):
    print(*a, flush=True)


def strip_frames(path: pathlib.Path, want: int) -> list:
    rgb, alpha = cutout(np.asarray(Image.open(path).convert('RGB')), keep_all=True)
    spans = split_strip(alpha, want=want) if want > 1 else [(0, alpha.shape[1])]
    out = []
    for a, b in spans:
        sa, sr = alpha[:, a:b], rgb[:, a:b]
        if (sa > 0.5).sum() < 500:
            raise SystemExit('empty frame in %s' % path)
        out.append((sr, sa, geometry(sa)))
    log('%-10s %s -> %d frames at x %s' % (path.stem, path.name, len(out), [s[0] for s in spans]))
    return out


def canvases(frames: list, char_height: float, align: bool) -> list:
    """Scale every frame so the median silhouette is `char_height` of the canvas, and
    drop each on its own canvas by the ground line. The trap's four states are lined up
    on the wooden board — the one part of it that does not move — for the same reason the
    trapped strips are lined up on the trap and not on the character standing in it."""
    med = float(np.median([g['h'] for _, _, g in frames]))
    base = char_height * SIZE / med
    if align:
        pivots = [g['foot_cx'] for _, _, g in frames]
        profs = [band_profile(al, g) for _, al, g in frames]
        pivots = [p + best_shift(profs[0], pr) for p, pr in zip(pivots, profs)]
    else:
        # A single drawing has nothing to line up against, and its floor centroid is not
        # its middle — the hole's lit tunnel floor runs off to one side, which pulls the
        # foot anchor far enough over that the arch clips the edge of the canvas.
        pivots = [(g['bbox'][0] + g['bbox'][2]) / 2.0 for _, _, g in frames]
    return [place(rgb, al, geo, base, SIZE, ANCHOR_X, GROUND_Y, pivot_x=p)
            for (rgb, al, geo), p in zip(frames, pivots)], med


def main():
    trap = strip_frames(SRC / 'trap.png', 4)
    hole = strip_frames(SRC / 'hole.png', 1)

    # Both props are drawn to fill their own canvas; what they measure against each other
    # on the map is decided by paint.js, which sizes each by cells like a character.
    # Heights chosen so the widest frame of each prop still clears the canvas: the trap
    # is 1.63 times wider than it is tall with the jaws open, and a third taller again in
    # mid-snap, so 0.52 is what fits both without cropping a jaw off.
    trap_c, trap_med = canvases(trap, 0.52, align=True)
    hole_c, hole_med = canvases(hole, 0.74, align=False)

    frames = trap_c + hole_c
    names = TRAP_STATES + ['hole']
    atlas = np.zeros((SIZE, SIZE * len(frames), 4), np.uint8)
    for i, f in enumerate(frames):
        atlas[:, i * SIZE:(i + 1) * SIZE] = f.astype(np.uint8)

    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas).save(OUT / 'props.png')
    meta = {
        'image': 'props.png',
        'frameWidth': SIZE, 'frameHeight': SIZE,
        'columns': len(frames), 'rows': 1,
        'anchor': {'x': ANCHOR_X, 'y': GROUND_Y},
        'frame': {n: i for i, n in enumerate(names)},
        # Silhouette height as a share of the canvas, per prop — the same number the
        # character sheets carry, and used the same way: size = cells * CS / charHeight.
        'charHeight': {'trap': 0.52, 'hole': 0.74},
        'source': {'trap': 'trap.png', 'hole': 'hole.png'},
    }
    (OUT / 'props.json').write_text(json.dumps(meta, indent=2) + '\n')
    log('atlas %dx%d -> %s' % (atlas.shape[1], atlas.shape[0], OUT / 'props.png'))

    # The same QA the sheets get: every prop over the backgrounds a matte can fail on.
    BUILD.mkdir(parents=True, exist_ok=True)
    for tag, bg in [('slate', (24, 34, 52)), ('green', (0, 255, 0)), ('white', (255, 255, 255))]:
        a = atlas[:, :, 3:4].astype(np.float32) / 255.0
        b = np.zeros_like(atlas[:, :, :3], np.float32)
        b[:] = bg
        comp = (atlas[:, :, :3].astype(np.float32) * a + b * (1 - a)).astype(np.uint8)
        Image.fromarray(comp).save(BUILD / ('props-%s.png' % tag))
    log('QA written to %s' % BUILD)


if __name__ == '__main__':
    main()
