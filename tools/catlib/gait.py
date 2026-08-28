"""Finding the four walking phases in an AI-generated clip.

The naive thing — take frames at 0%, 25%, 50%, 75% of the clip — does not work here.
Nothing guarantees a generated clip starts on a contact pose, contains a whole number of
strides, or holds a steady cadence, and none of these five clips does any of the three.
Two of them stand still for the first half-second before setting off, and the left-facing
one dollies in throughout.

So the cycle is measured:

  picture  Each drawn pose is reduced to a small ground-aligned, scale-normalised
           picture of the legs. Ground-aligned because the character drifts up the frame
           as the camera moves; scale-normalised because it also grows; legs only because
           on the side views the tail sweeps through the lower body and, left in, it
           swamps every measurement made here.

  stride   The lag at which that sequence repeats. The cost curve cannot simply be
           minimised — it rises with lag, so the shortest lag always wins — so a genuine
           local dip is required. Measured over the moving part of each clip this gives
           9, 16, 14, 17 and 13 drawn poses: 1.1s to 2.1s per stride.

  phase    Contacts are maxima of stance spread — the width of the part of the cat
           actually touching the ground. F1 sits on a contact and the rest follow at
           even quarters of the measured stride, so F3 is the opposite contact by
           construction and the four play at a constant rate. Even quarters *of a
           measured stride anchored on a measured contact* is a different thing from
           slicing the clip at 0/25/50/75%, which assumes the clip is exactly one stride
           long and begins on a contact.

  choice   Every candidate contact is scored on how well its four poses read as
           wide-close-wide-close, on how different the two contacts are from each other,
           and on how evenly F4 flows back into F1; the best cycle wins. Within each
           phase one neighbouring pose either side may be substituted if it shows the
           same pose more cleanly, because the biomechanically ideal frame is sometimes
           the one where the generator lost a paw.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GROUND_BAND = 0.07   # bottom share of the silhouette that counts as "on the ground"
LEG_BAND = 0.50      # bottom share of the body used for the self-similarity picture
GRID = 48            # size of that picture
MIN_STRIDE, MAX_STRIDE = 4, 26    # drawn poses; 0.5s .. 3.2s at the clips' 8 fps
MIN_MOTION = 0.30    # share of the clip's median pose-to-pose change below which the
                     # cat is standing still, which several clips do before setting off
QUALITY_WINDOW = 1   # drawn poses either side of a phase that may be substituted


def features(alpha: np.ndarray, geo: dict) -> dict:
    m = alpha > 0.5
    x0, y0, x1, y1 = geo['bbox']
    h = geo['h']
    ground = geo['ground']

    # Stance spread: how far apart the parts of the cat touching the ground are.
    band = m[max(0, ground - int(GROUND_BAND * h)):ground + 1, :]
    cols = np.where(band.any(axis=0))[0]
    spread = float(cols.max() - cols.min() + 1) / h if cols.size else 0.0

    cx = int(geo['torso_cx'])
    half = max(2, int(0.62 * h))
    top = max(0, ground - int(LEG_BAND * h))
    box = np.zeros((int(LEG_BAND * h) + 2, 2 * half), np.float32)
    src = m[top:ground + 2, max(0, cx - half):cx + half].astype(np.float32)
    box[:src.shape[0], :src.shape[1]] = src
    pic = np.asarray(Image.fromarray((box * 255).astype(np.uint8)).resize((GRID, GRID))) / 255.0

    return dict(ok=m.sum() > 500, spread=spread, pic=pic)


def _moving(pics: np.ndarray) -> np.ndarray:
    n = len(pics)
    d = np.zeros(n)
    flat = pics.reshape(n, -1)
    d[1:] = np.abs(flat[1:] - flat[:-1]).mean(axis=1)
    d[0] = d[1] if n > 1 else 1.0
    med = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    return ndi.uniform_filter1d(d, 3) > MIN_MOTION * med


def stride_period(pics: np.ndarray, moving: np.ndarray) -> int:
    idx = np.where(moving)[0]
    lo, hi = (int(idx.min()), int(idx.max())) if idx.size else (0, len(pics) - 1)
    flat = pics.reshape(len(pics), -1)
    lags = list(range(MIN_STRIDE, min(MAX_STRIDE, (hi - lo) // 2) + 1))
    if len(lags) < 5:
        return max(MIN_STRIDE, (hi - lo) // 2 or MIN_STRIDE)
    cost = np.array([np.abs(flat[lo:hi + 1 - l] - flat[lo + l:hi + 1]).mean() for l in lags])
    best, best_prom = None, -np.inf
    for k in range(1, len(cost) - 1):
        if cost[k] <= cost[k - 1] and cost[k] <= cost[k + 1]:
            prom = max(cost[max(0, k - 4):k].max(initial=cost[k]),
                       cost[k + 1:k + 5].max(initial=cost[k])) - cost[k]
            if prom > best_prom:
                best, best_prom = lags[k], prom
    return int(best if best is not None else lags[int(np.argmin(cost))])


def contacts(spread: np.ndarray, moving: np.ndarray) -> list[int]:
    """Poses where the stance is widest. Detrended first, because a clip that dollies in
    raises the whole curve and would otherwise hide every early contact behind a later
    one; and reported at the centre of a plateau rather than its leading edge, because on
    a three-quarter view the widest stance is held for several poses."""
    det = spread - ndi.uniform_filter1d(spread, 21, mode='nearest')
    span = float(np.percentile(det, 95) - np.percentile(det, 5)) or 1e-9
    floor = np.percentile(det, 55) + 0.20 * span
    out, i, n = [], 0, len(det)
    while i < n:
        if det[i] < floor or not moving[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and det[j + 1] >= floor and moving[j + 1]:
            j += 1
        seg = det[i:j + 1]
        peak = np.where(seg >= seg.max() - 0.15 * span)[0]
        out.append(i + int(round(peak.mean())))
        i = j + 1
    return out


QUALITY_GAIN = 0.12  # a substitution has to be worth this much to be worth the timing


def refine(feats, quality, i: int, period: int) -> int:
    """Swap a phase onto a neighbouring pose only when that pose is markedly cleaner.

    The latitude is one drawn pose, and on the shortest clip a quarter of the stride is
    only two poses — so moving one frame is a 45% timing error, which reads as a limp.
    The substitution therefore has to buy a real improvement, and on strides too short
    for it to be cheap it is not offered at all."""
    if period / 4.0 < 3.0:
        return i
    n = len(feats)
    cands = [j for j in range(max(0, i - QUALITY_WINDOW), min(n - 1, i + QUALITY_WINDOW) + 1)
             if feats[j]['ok']]
    ref = feats[i]['spread']

    def value(j):
        return float(quality[j]) - 4.0 * abs(feats[j]['spread'] - ref)

    best = max(cands, key=value) if cands else i
    return best if value(best) - value(i) > QUALITY_GAIN else i


def choose_cycle(feats: list[dict], quality: np.ndarray, override=None) -> dict:
    n = len(feats)
    pics = np.stack([f['pic'] for f in feats])
    flat = pics.reshape(n, -1)
    spread = ndi.uniform_filter1d(np.array([f['spread'] for f in feats]), 3)
    moving = _moving(pics)
    period = int(override) if override else stride_period(pics, moving)
    quarter, half = period / 4.0, period / 2.0

    def dist(i, j):
        return float(np.abs(flat[i] - flat[j]).mean())

    starts = [c for c in contacts(spread, moving) if feats[c]['ok']]
    if not starts:
        starts = [int(np.argmax(np.where(moving, spread, -1)))]

    best = None
    for c in starts:
        idx = [c, int(round(c + quarter)), int(round(c + half)), int(round(c + 3 * quarter))]
        if idx[3] + QUALITY_WINDOW >= n or len(set(idx)) < 4:
            continue
        if not all(feats[i]['ok'] for i in idx) or not moving[idx].all():
            continue
        rhythm = (spread[idx[0]] + spread[idx[2]]) - (spread[idx[1]] + spread[idx[3]])
        opposite = dist(idx[0], idx[2]) + dist(idx[1], idx[3])
        adj = [dist(idx[0], idx[1]), dist(idx[1], idx[2]), dist(idx[2], idx[3])]
        loop = dist(idx[3], idx[0])
        # Prefer a cycle away from the ends of the clip: generated video is least
        # settled in its first and last half-second.
        mid = np.where(moving)[0]
        centre = 1.0 - abs((idx[0] + idx[3]) / 2.0 - mid.mean()) / max(1.0, mid.ptp() / 2.0)
        score = (3.0 * rhythm + 1.5 * opposite
                 - 3.0 * abs(loop - float(np.mean(adj)))
                 + 0.8 * float(quality[idx].mean())
                 + 0.5 * float(np.clip(centre, -1, 1)))
        if best is None or score > best['score']:
            best = dict(score=score, idx=idx, period=period, rhythm=float(rhythm),
                        opposite=float(opposite), loop=float(loop),
                        step=float(np.mean(adj)), contact=int(c))
    if best is None:                      # degenerate clip: even quarters from the middle
        c = max(0, n // 2 - int(half))
        best = dict(score=0.0, period=period, contact=c,
                    idx=[min(n - 1, int(round(c + k * quarter))) for k in range(4)],
                    rhythm=0.0, opposite=0.0, loop=0.0, step=0.0)
    return best
