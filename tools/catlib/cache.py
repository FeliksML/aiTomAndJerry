"""Matte every extracted frame once and keep the alpha around for analysis."""
from __future__ import annotations

import pathlib
import numpy as np
from PIL import Image

from .matte import cutout

DIRS5 = ['down', 'up', 'left', 'down-left', 'up-left']


def frame_paths(root: pathlib.Path, d: str) -> list[pathlib.Path]:
    return sorted((root / 'build' / 'frames-raw' / d).glob('*.png'))


def alpha_cache(root: pathlib.Path, d: str, log=print) -> np.ndarray:
    out = root / 'build' / 'alpha' / (d + '.npy')
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = frame_paths(root, d)
    if out.exists():
        a = np.load(out)
        if len(a) == len(paths):
            return a
    stack = []
    for i, p in enumerate(paths):
        rgb = np.asarray(Image.open(p).convert('RGB'))
        _, a = cutout(rgb)
        stack.append((a * 255).astype(np.uint8))
        if log and i % 25 == 0:
            log('  %s %3d/%d' % (d, i, len(paths)))
    arr = np.stack(stack)
    np.save(out, arr)
    return arr
