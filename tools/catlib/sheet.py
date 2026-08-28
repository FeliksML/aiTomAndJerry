"""Packing eight directions into one sheet, and the QA renders that prove it.

Shared by both characters. The sheet contract is fixed and the runtime depends on it:
four columns are the walk phases F1..F4, and the eight rows run DOWN, DOWN_LEFT, LEFT,
UP_LEFT, UP, UP_RIGHT, RIGHT, DOWN_RIGHT — mirrors directly opposite their sources.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from PIL import Image

ROWS = ['down', 'down-left', 'left', 'up-left', 'up', 'up-right', 'right', 'down-right']
MIRROR = {'right': 'left', 'down-right': 'down-left', 'up-right': 'up-left'}
SOURCES = ['down', 'up', 'left', 'down-left', 'up-left']
CAMEL = {'down': 'down', 'up': 'up', 'left': 'left',
         'down-left': 'downLeft', 'up-left': 'upLeft',
         'right': 'right', 'down-right': 'downRight', 'up-right': 'upRight'}

BGS = [('black', (0, 0, 0)), ('slate', (24, 34, 52)),
       ('green', (0, 255, 0)), ('white', (255, 255, 255))]

# What the four columns mean, per animation. The walk loops all four; `trapped` front-loads
# the one-shot impact so a caller can play 0-1 once and then loop 2-3 for as long as the
# hold lasts, which is what the environment's frozen counter actually describes.
PHASES = {
    'walk': ['contactA', 'passingA', 'contactB', 'passingB'],
    'trapped': ['snap', 'recoil', 'struggleA', 'struggleB'],
}

# How many leading frames play once rather than looping. A walk loops all four. Being
# caught does not: the snap happens once and then the struggle continues, so frames 0-1
# are a one-shot and 2-3 are the loop. The environment holds a trapped agent for
# CFG.freezeSteps = 5 steps, which lands exactly on snap, recoil, struggle, struggle,
# struggle.
ONE_SHOT = {'walk': 0, 'trapped': 2}


def mirror_all(canvases: dict) -> dict:
    """The three right-facing directions, as exact horizontal flips. Generating them
    instead would give a subtly different character on one side of the screen."""
    for dst, src in MIRROR.items():
        canvases[dst] = [f[:, ::-1, :].copy() for f in canvases[src]]
    return canvases


def write(canvases: dict, out: pathlib.Path, image: str, meta_name: str,
          size: int, fps: int, anchor: dict, idle: dict, source: dict,
          frames_dir: str = 'walk', render_scale: float = 1.0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for d in ROWS:
        sub = out / frames_dir / d
        sub.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(canvases[d]):
            Image.fromarray(f.astype(np.uint8)).save(sub / ('%d.png' % i))

    sheet = np.zeros((size * len(ROWS), size * 4, 4), np.uint8)
    heights, clipped = [], []
    for r, d in enumerate(ROWS):
        for c, f in enumerate(canvases[d]):
            sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = f.astype(np.uint8)
            m = f[:, :, 3] > 127
            ys, xs = np.where(m)
            if ys.size:
                heights.append(int(ys.max() - ys.min() + 1))
                if ys.min() <= 0 or ys.max() >= size - 1 or xs.min() <= 0 or xs.max() >= size - 1:
                    clipped.append('%s F%d' % (d, c))
    Image.fromarray(sheet).save(out / image)
    if clipped:
        print('  WARNING: %d frames touch the frame edge: %s'
              % (len(clipped), ', '.join(clipped[:8])), flush=True)

    meta = {
        'image': image,
        'frameWidth': size, 'frameHeight': size,
        'columns': 4, 'rows': len(ROWS), 'framesPerDirection': 4,
        'directions': {CAMEL[d]: i for i, d in enumerate(ROWS)},
        'directionOrder': ROWS,
        'defaultFPS': fps,
        'anchor': anchor,
        # How tall one character is, as a fraction of the frame — the renderer divides by
        # this, so a sheet padded more generously still draws the same character.
        #
        # It is the *measured* silhouette divided by `renderScale`, and the division is
        # the whole point. Measured alone, every sheet would draw its silhouette at the
        # same height — which is wrong the moment a pose set has both arms thrown up,
        # because that inflates the silhouette without making the character any bigger.
        # Measured, the trapped cat came out 24% oversized and the trapped mouse 14%
        # undersized against their own walk sheets. renderScale is the correction, set
        # from head width, which is the one measure a crouch does not change.
        'charHeight': round(float(np.median(heights)) / size / render_scale, 4)
                      if heights else 1.0,
        'renderScale': render_scale,
        'idleFrame': {CAMEL[d]: int(idle.get(CAMEL[d], idle.get(CAMEL[MIRROR.get(d, d)], 0)))
                      for d in ROWS},
        'mirrored': {CAMEL[k]: CAMEL[v] for k, v in MIRROR.items()},
        'phases': PHASES.get(frames_dir, PHASES['walk']),
        'oneShotFrames': ONE_SHOT.get(frames_dir, 0),
        'framesDir': frames_dir,
        'source': source,
    }
    (out / meta_name).write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def composite(f: np.ndarray, bg) -> np.ndarray:
    a = f[:, :, 3:4].astype(np.float32) / 255.0
    b = np.zeros_like(f[:, :, :3], np.float32)
    b[:] = bg
    return (f[:, :, :3].astype(np.float32) * a + b * (1 - a)).astype(np.uint8)


def qa(canvases: dict, qadir: pathlib.Path, size: int, fps: int, prefix: str = ''):
    """Every frame over every background a matte can fail on, the animation as a GIF,
    and the stance strips that show whether the gait reads wide-close-wide-close."""
    qadir.mkdir(parents=True, exist_ok=True)
    for d in ROWS:
        strip = np.vstack([np.hstack([composite(f, bg) for f in canvases[d]])
                           for _, bg in BGS])
        Image.fromarray(strip).save(qadir / ('%sframes-%s.png' % (prefix, d)))
        gif = [Image.fromarray(composite(f, (24, 34, 52))) for f in canvases[d]]
        gif[0].save(qadir / ('%swalk-%s.gif' % (prefix, d)), save_all=True,
                    append_images=gif[1:], duration=int(1000 / fps), loop=0, disposal=2)

    cell = 128
    for name, bg in (('slate', (24, 34, 52)), ('green', (0, 255, 0))):
        board = np.zeros((cell * len(ROWS), cell * 4, 3), np.uint8)
        for r, d in enumerate(ROWS):
            for c, f in enumerate(canvases[d]):
                board[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = np.asarray(
                    Image.fromarray(composite(f, bg)).resize((cell, cell), Image.LANCZOS))
        Image.fromarray(board).save(qadir / ('%ssheet-%s.png' % (prefix, name)))

    strips = []
    for d in ROWS:
        row = []
        for f in canvases[d]:
            m = f[:, :, 3] > 127
            ys = np.where(m.any(axis=1))[0]
            band = m[max(0, ys.max() - int(0.09 * (ys.max() - ys.min()))):ys.max() + 1, :]
            img = np.asarray(Image.fromarray((band * 255).astype(np.uint8)).resize((size, 28)))
            row.append(np.pad(img, ((0, 0), (0, 4)), constant_values=90))
        strips.append(np.hstack(row))
    Image.fromarray(np.vstack([np.pad(s, ((0, 6), (0, 0)), constant_values=90) for s in strips])
                    ).save(qadir / ('%sgait-check.png' % prefix))
