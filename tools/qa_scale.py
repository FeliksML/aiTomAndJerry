#!/usr/bin/env python3
"""Draw every sheet the way the game draws it, so the sizes can be compared by eye.

The renderer places a frame by its ground anchor and sizes it so the character is
`cells` map-cells tall — `size = cells * CS / charHeight`. Reproducing exactly that here
is the point: a composite built any other way would prove nothing about the game.
"""
import json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
CS = 78                      # map cell size, chosen so the sheet fits on screen
# What each sheet is sized by, in map cells. Walking and being trapped both draw the
# CHARACTER at its own height; the catch does too — Tom is still Tom with a mouse in his
# fist. The escape is the exception and is sized by the ARCH, because the arch is what the
# sheet's silhouette actually is, and it has to end up the same size as the standing hole
# the map draws on that cell.
CELLS = {('tom', 'walk'): 1.38, ('tom', 'trapped'): 1.38, ('tom', 'catch'): 1.38,
         ('jerry', 'walk'): 0.95, ('jerry', 'trapped'): 0.95, ('jerry', 'escape'): 1.05}
SHEETS = [('tom', 'walk', 'cat'), ('tom', 'trapped', 'cat'), ('tom', 'catch', 'cat'),
          ('jerry', 'walk', 'mouse'), ('jerry', 'trapped', 'mouse'),
          ('jerry', 'escape', 'mouse')]
BG = (24, 34, 52)


def frame(hero, anim, folder, row, col):
    d = ROOT / 'app' / 'assets' / folder
    m = json.loads((d / ('%s-%s.json' % (hero, anim))).read_text())
    S = m['frameWidth']
    sh = np.asarray(Image.open(d / m['image']))
    f = sh[row * S:(row + 1) * S, col * S:(col + 1) * S]
    size = CELLS[(hero, anim)] * CS / m['charHeight']   # exactly what paint.js computes
    k = size / S
    im = Image.fromarray(f).resize((max(1, int(S * k)),) * 2, Image.LANCZOS)
    return im, m['anchor'], m


def main():
    cols, cell_w, cell_h = 5, 230, 300
    out = Image.new('RGB', (cell_w * cols + 120, cell_h * len(SHEETS)), BG)
    dr = ImageDraw.Draw(out)
    ground = cell_h - 46
    for r, (hero, anim, folder) in enumerate(SHEETS):
        picks = [(0, 0)] if anim == 'walk' else []
        picks += [(0, c) for c in range(4)] if anim == 'trapped' else [(0, c) for c in range(4)]
        picks = picks[:cols]
        for c, (row, col) in enumerate(picks):
            im, anchor, m = frame(hero, anim, folder, row, col)
            x = 120 + c * cell_w + cell_w // 2 - int(im.width * anchor['x'])
            y = r * cell_h + ground - int(im.height * anchor['y'])
            out.paste(im, (x, y), im)
        dr.line([(120, r * cell_h + ground), (out.width, r * cell_h + ground)],
                fill=(90, 110, 140), width=1)
        dr.text((8, r * cell_h + cell_h // 2), '%s\n%s' % (hero, anim), fill=(230, 238, 250))
    out.save(ROOT / 'build' / 'qa' / 'scale-check.png')
    print('wrote build/qa/scale-check.png', out.size)


if __name__ == '__main__':
    main()
