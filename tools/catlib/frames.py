"""Per-frame geometry, quality scoring, and the placement of a frame on the sprite canvas."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

C8 = np.ones((3, 3), bool)

TORSO_TOP, TORSO_BOT = 0.34, 0.66   # share of body height used as the pelvis/torso anchor
GROUND_MASS = 0.004                 # share of silhouette mass that defines the ground line
HEAD_BAND = 0.26                    # share of body height treated as the head, for morph checks
FOOT_BAND = 0.07                    # share of body height that is "whatever is on the ground"


def geometry(alpha: np.ndarray) -> dict:
    """bbox, the torso anchor, and the ground line, all in source pixels."""
    m = alpha > 0.5
    ys, xs = np.where(m)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h = y1 - y0 + 1

    rows = m.sum(axis=1).astype(np.float64)
    cum = np.cumsum(rows[::-1])[::-1]           # mass at or below each row
    ground = int(np.max(np.where(cum >= GROUND_MASS * m.sum())[0]))

    band = m[y0 + int(TORSO_TOP * h): y0 + int(TORSO_BOT * h) + 1, :]
    torso_cx = float(np.where(band)[1].mean()) if band.any() else (x0 + x1) / 2.0

    # Whatever is standing on the ground. For a walk that is the feet, which move, so it
    # is no use as an anchor. For the trapped animation it is the trap — a rigid object
    # the character is pinned to, and the only thing in the frame that genuinely does not
    # move. Anchoring on the torso there would swing the trap 40px either way and read as
    # the character dragging the trap around rather than being held by it.
    foot = m[max(0, ground - int(FOOT_BAND * h)):ground + 1, :]
    foot_cx = float(np.where(foot)[1].mean()) if foot.any() else torso_cx

    return dict(bbox=(x0, y0, x1, y1), h=h, w=x1 - x0 + 1, area=int(m.sum()),
                ground=ground, torso_cx=torso_cx, foot_cx=foot_cx)


def head_signature(alpha: np.ndarray, geo: dict) -> np.ndarray:
    """A small picture of the head, used to notice the generator redrawing the face."""
    x0, y0, x1, y1 = geo['bbox']
    top = (alpha[y0:y0 + max(2, int(HEAD_BAND * geo['h'])), x0:x1 + 1] > 0.5).astype(np.float32)
    zy = max(1, top.shape[0] // 16); zx = max(1, top.shape[1] // 16)
    return top[:zy * 16, :zx * 16].reshape(16, zy, 16, zx).mean((1, 3))


def quality(alphas: np.ndarray, rgbs, geos: list[dict]) -> np.ndarray:
    """0..1 per frame. Low means the generator wobbled: a lost paw, a smeared face,
    a sudden change of build, or motion blur. Used to nudge a chosen phase onto a
    neighbouring frame that shows the same pose more cleanly."""
    n = len(geos)
    area = np.array([g['area'] for g in geos], float)
    h = np.array([g['h'] for g in geos], float)

    med_area = ndi.median_filter(area, size=9, mode='nearest')
    med_h = ndi.median_filter(h, size=9, mode='nearest')
    d_area = np.abs(area / med_area - 1.0)
    d_h = np.abs(h / med_h - 1.0)

    heads = np.stack([head_signature(alphas[i], geos[i]) for i in range(n)])
    head_med = np.median(heads, axis=0)
    d_head = np.abs(heads - head_med).mean(axis=(1, 2))

    sharp = np.empty(n)
    blobs = np.empty(n)
    for i in range(n):
        x0, y0, x1, y1 = geos[i]['bbox']
        m = alphas[i][y0:y1 + 1, x0:x1 + 1] > 0.5
        g = rgbs[i][y0:y1 + 1, x0:x1 + 1].mean(axis=2)
        sharp[i] = float(ndi.laplace(g)[m].var())
        lab, k = ndi.label(m, C8)
        sizes = ndi.sum(m, lab, np.arange(1, k + 1)) if k else np.array([1.0])
        blobs[i] = 1.0 - sizes.max() / max(1.0, sizes.sum())   # share of stray fragments

    sharp_n = np.clip(sharp / max(1e-9, float(np.median(sharp))), 0, 2) / 2.0
    q = (1.0
         - 6.0 * np.clip(d_area, 0, 0.10)
         - 6.0 * np.clip(d_h, 0, 0.10)
         - 3.0 * np.clip(d_head / max(1e-9, float(np.median(d_head)) * 3), 0, 1) * 0.25
         - 8.0 * np.clip(blobs, 0, 0.05)
         + 0.25 * (sharp_n - 0.5))
    return np.clip(q, 0.0, 1.0)


def height_trend(heights: np.ndarray, stride: int) -> np.ndarray:
    """The slow part of the silhouette height: how the camera is framing the cat, with
    the walking pose averaged out.

    The window is one stride wide on purpose. Over a whole stride every pose occurs once,
    so the median across it cancels pose entirely and what is left is framing. A straight
    line fitted to the whole clip is not enough — one clip's framing tightens in a curve,
    and correcting it linearly left the character 10% larger halfway through its own
    walk cycle."""
    w = max(5, int(stride) | 1)
    med = ndi.median_filter(heights.astype(float), size=w, mode='nearest')
    return ndi.uniform_filter1d(med, max(3, w // 2 | 1), mode='nearest')


BAND_W = 260         # px either side of the anchor that the alignment profile covers
MAX_SHIFT = 90       # px; more than this and the match is not the same object


def band_profile(alpha: np.ndarray, geo: dict) -> np.ndarray:
    """How much of the character is touching the floor, column by column, centred on the
    rough anchor. Used to line frames up on the one rigid thing in the shot."""
    m = alpha > 0.5
    g, h = geo['ground'], geo['h']
    band = m[max(0, g - int(FOOT_BAND * h)):g + 1, :].sum(axis=0).astype(np.float32)
    c = int(round(geo['foot_cx']))
    out = np.zeros(2 * BAND_W + 1, np.float32)
    lo, hi = max(0, c - BAND_W), min(len(band), c + BAND_W + 1)
    out[lo - (c - BAND_W):hi - (c - BAND_W)] = band[lo:hi]
    return out


def best_shift(ref: np.ndarray, other: np.ndarray) -> int:
    """The offset that best lines `other` up with `ref`.

    The trap is the same rigid object in every frame of a trapped strip, so the offset
    that maximises overlap is the offset that holds the trap still. A centroid cannot do
    this job: it is pulled by whatever else happens to be resting on the floor.
    """
    best, score = 0, -1.0
    for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
        a = other if s == 0 else np.roll(other, s)
        if s > 0:
            a[:s] = 0
        elif s < 0:
            a[s:] = 0
        v = float(np.minimum(ref, a).sum())
        if v > score:
            best, score = s, v
    return -best


def place(rgb: np.ndarray, alpha: np.ndarray, geo: dict, scale: float,
          size: int, anchor_x: float, ground_y: float,
          pivot_x: float | None = None) -> np.ndarray:
    """Scale a matted frame and drop it on the sprite canvas so that the torso sits on
    the vertical axis and the ground line sits on the anchor row. Anchoring the ground
    rather than the feet is what keeps the body's own bounce: at least one foot is down
    through a walk, so the ground line is steady while the body rides over it."""
    from PIL import Image

    h, w = alpha.shape
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    src = np.dstack([np.clip(rgb, 0, 255), np.clip(alpha * 255.0, 0, 255)]).astype(np.uint8)
    # Premultiply before resampling so half-transparent edge pixels cannot drag the
    # paper colour back into the sprite.
    pm = src.astype(np.float32)
    pm[:, :, :3] *= pm[:, :, 3:4] / 255.0
    small = np.asarray(Image.fromarray(pm.astype(np.uint8)).resize((nw, nh), Image.LANCZOS)).astype(np.float32)
    a = np.clip(small[:, :, 3:4], 0, 255)
    col = np.where(a > 0.5, np.clip(small[:, :, :3] * 255.0 / np.maximum(a, 1e-3), 0, 255), 0.0)
    small = np.dstack([col, a])

    dst = np.zeros((size, size, 4), np.float32)
    px = geo['torso_cx'] if pivot_x is None else pivot_x
    ox = int(round(anchor_x * size - px * scale))
    oy = int(round(ground_y * size - geo['ground'] * scale))

    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cw = min(nw - sx0, size - dx0)
    ch = min(nh - sy0, size - dy0)
    if cw > 0 and ch > 0:
        dst[dy0:dy0 + ch, dx0:dx0 + cw] = small[sy0:sy0 + ch, sx0:sx0 + cw]
    return dst
