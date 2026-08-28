"""Finding the frames that actually differ.

Every one of the five clips is 8 fps of animation living in a 24 fps container: the
picture changes on one frame in three and is held for the other two. Left alone this
poisons everything downstream — self-similarity reports a stride of three frames because
the sequence genuinely repeats every three, quality scoring compares a frame against two
copies of itself, and a chosen "timestamp" may land on a held frame rather than the drawn
one. So the held copies are collapsed first and the whole analysis runs on drawn poses.
"""

from __future__ import annotations

import collections

import numpy as np
from scipy import ndimage as ndi


def hold_factor(alphas: np.ndarray) -> tuple[int, int, np.ndarray]:
    """(frames each pose is held, index of the first drawn frame, per-pair difference)."""
    a = alphas.astype(np.int16)
    diff = np.abs(a[1:] - a[:-1]).mean(axis=(1, 2))
    top = float(np.median(diff[diff > np.median(diff)])) if diff.size else 1.0
    changed = diff > 0.25 * top

    runs, run = [], 1
    for c in changed:
        if c:
            runs.append(run)
            run = 1
        else:
            run += 1
    runs.append(run)
    if not runs:
        return 1, 0, diff
    modal, count = collections.Counter(runs).most_common(1)[0]
    if modal < 2 or count < 0.6 * len(runs):
        return 1, 0, diff

    # Phase from where the change actually lands, summed over the whole clip. Reading it
    # off the first change instead is fragile: two of these clips twitch irregularly in
    # their opening frames as the cat sets off, and a phase one frame out puts the group
    # boundary in the middle of a drawn pose — which silently mixes two poses together
    # and makes the stance signal look like noise.
    score = [float(diff[i - 1:len(diff):modal].sum()) for i in range(1, modal + 1)]
    phase = (int(np.argmax(score)) + 1) % modal
    return int(modal), phase, diff


def keyframes(alphas: np.ndarray, rgbs=None) -> list[int]:
    """One source index per drawn pose — the sharpest copy of each held group, so the
    frame that gets exported is the least re-compressed of the three."""
    k, phase, _ = hold_factor(alphas)
    if k <= 1:
        return list(range(len(alphas)))
    out = []
    starts = ([0] if phase else []) + list(range(phase, len(alphas), k))
    for start in starts:
        group = list(range(start, min(start + (phase if start == 0 and phase else k),
                                      len(alphas))))
        if rgbs is None:
            out.append(group[len(group) // 2])
        else:
            out.append(max(group, key=lambda i: float(ndi.laplace(rgbs[i].mean(axis=2)).var())))
    return out
