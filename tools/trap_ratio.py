#!/usr/bin/env python3
"""How wide the trap is, as a share of the character's height.

The number the trapped strips have to hit. paint.js draws an unsprung trap 0.84 cells
wide and says a hazard must read at 22px, so at the game's 44px cell the trap inside a
sprite has to clear 0.5 of the cat's height and 0.72 of the mouse's — anything less and
the trap that caught you is smaller than the ones you walked past.
"""
import os, sys, importlib.util
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catlib.matte import cutout                      # noqa: E402

spec = importlib.util.spec_from_file_location(
    'b', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_strip_sprites.py'))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)

CELLS = {'tom': 1.38, 'jerry': 0.95}
CS = 44


def ratio(path):
    rgb, a = cutout(np.asarray(Image.open(path).convert('RGB')), keep_all=True)
    out = []
    for x0, x1 in B.split_strip(a):
        sa, sr = a[:, x0:x1], rgb[:, x0:x1]
        m = sa > 0.5
        ys = np.where(m.any(axis=1))[0]
        h = ys.max() - ys.min() + 1
        lo = ys.max() - int(0.12 * h)
        band_a, band = sa[lo:ys.max() + 1], sr[lo:ys.max() + 1]
        R, G, Bc = band[:, :, 0], band[:, :, 1], band[:, :, 2]
        wood = (band_a > 0.5) & (R > Bc + 30) & (R > 70) & (R < 210) & (G < R - 10)
        if wood.sum() > 40:
            c = np.where(wood)[1]
            out.append((c.max() - c.min() + 1) / h)
    return float(np.median(out)) if out else float('nan')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        hero = 'jerry' if 'jerry' in os.path.basename(p) else 'tom'
        r = ratio(p)
        px = r * CELLS[hero] * CS
        need = 22.0 / (CELLS[hero] * CS)
        print('%-26s %-5s ratio %.2f  -> %4.1f px on screen   %s  (need >= %.2f, vector trap 37px)'
              % (os.path.basename(p), hero, r, px,
                 'PASS' if px >= 22 else 'TOO SMALL', need))
