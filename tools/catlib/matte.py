"""Background-connected matting for the Tom renders.

The renders are cel art: a closed dark ink line runs all the way round the character,
and every light feature that matters — muzzle, chest, belly, paws, feet, tail tip —
sits *inside* that line. So the background is exactly "the light region reachable from
the image border without crossing ink", which is a connectivity question, not a colour
question. A plain "make near-white transparent" pass would delete the paws: the brightest
genuine character pixel measured across all five renders sits only ~40 units off white.

Four details earn their place:

  * The barrier is the drawn line, not a whiteness threshold. The renders carry a soft
    contact shadow under the feet that is *darker* than the character's own cream, so no
    single "how white is it" cut can separate them. A line can: it is either far darker
    than any shadow, or — where a foot sits in shade and its outline goes pale — it is
    still a step, and the shadow is not. Measured on these renders the page gradient
    tops out at 1.2 and the shadow averages 0.5, while a drawn line reaches 166, so the
    gradient term has two orders of magnitude of headroom. Without it the flood pours
    through the pale outline and hollows the shaded foot out.

  * Enclosed holes. The gap between an arm and the torso is background that never
    touches the image border, and the ink barrier alone would leave it opaque. It is
    told apart from a lit paw by how much *pure* white it holds: measured across the
    renders, real holes run 29-86% pure white and no character feature exceeds a single
    pixel, so the test has an enormous margin. Seeding on "is there pure white here at
    all" instead would hollow out the paws, which is exactly what it did once.

  * The feather band. Alpha inside a thin band around the silhouette is taken from the
    pixel's own distance-from-white, so the edge keeps the render's own anti-aliasing.
    Outside that band alpha is a flat zero — that is what removes the drop shadow and
    the KlingAI watermark instead of leaving them as grey ghosts.

  * Decontamination. Edge pixels are a mix of ink and white paper. Left alone they read
    as a white halo on a dark game background. Un-premultiplying against the measured
    paper colour recovers the ink, so the edge stays dark on black and on green.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

C8 = np.ones((3, 3), bool)

T_INK = 110.0      # distance-from-white above which a pixel is ink / fur / a feature
T_SAT = 32.0       # ...or saturated enough to be one (pink ears, yellow eyes)
T_EDGE = 8.0       # ...or a drawn step. Page noise measures 1.2, a line 166.
EDGE_BLUR = 0.8    # px; suppresses codec noise before the gradient is taken
T_PURE = 20.0      # a pixel this close to white is certainly paper
HOLE_PURE_PX = 40   # a hole holds at least this many pure-white pixels...
HOLE_PURE_FRAC = 0.08   # ...and this share of its area. Character features hold none.
FEATHER = 1.8      # px; width of the content-driven alpha ramp at the silhouette
A_LO, A_HI = 10.0, 95.0   # distance-from-white mapped to alpha 0..1 inside the band
DECON_FLOOR = 0.20        # below this alpha, un-premultiplying only amplifies noise


def page_colour(rgb: np.ndarray, border: int = 12) -> np.ndarray:
    """The paper colour, read off the outer border of the image."""
    r = np.concatenate([rgb[:border].reshape(-1, 3), rgb[-border:].reshape(-1, 3),
                        rgb[:, :border].reshape(-1, 3), rgb[:, -border:].reshape(-1, 3)])
    return np.median(r, axis=0).astype(np.float32)


def whiteness(rgb: np.ndarray, page: np.ndarray | None = None) -> np.ndarray:
    """Distance from the paper, per channel, worst channel wins.

    Against white this is exactly 255 minus the darkest channel, which is what the cat
    renders were measured with — a saturated-but-light pixel such as a pink ear stays
    firmly in the foreground, where a luma measure would lose it. Passing an explicit
    paper colour lets the same logic run against off-white stock without retuning."""
    if page is None:
        return 255.0 - rgb.min(axis=2)
    return np.abs(rgb - page.reshape(1, 1, 3)).max(axis=2)


def _paper(rgb: np.ndarray, d: np.ndarray) -> np.ndarray:  # noqa: C901
    """Everything the ink does not enclose: the page, the drop shadow, and any hole."""
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    lum = ndi.gaussian_filter(rgb.mean(axis=2), EDGE_BLUR)
    grad = np.hypot(ndi.sobel(lum, 0), ndi.sobel(lum, 1)) / 4.0
    open_ = ~((d > T_INK) | (sat > T_SAT) | (grad > T_EDGE))
    lab, n = ndi.label(open_, structure=C8)
    if n == 0:
        return np.zeros(d.shape, bool)

    touching = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    touching.discard(0)
    paper = np.isin(lab, sorted(touching))

    # Holes: enclosed light regions that are mostly pure white. The muzzle, the chest
    # and a lit paw are enclosed too, so area alone is not enough — purity is.
    pure = d < T_PURE
    enclosed = np.unique(lab[~paper & (lab > 0)])
    holes = []
    for i in enclosed:
        m = lab == i
        npx = int(m.sum())
        npure = int(pure[m].sum())
        if npure >= HOLE_PURE_PX and npure >= HOLE_PURE_FRAC * npx:
            holes.append(i)
    if holes:
        paper |= np.isin(lab, holes)
    return paper


def cutout(rgb: np.ndarray, page: np.ndarray | None = None,
           keep_all: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(rgb decontaminated, float32) and (alpha in 0..1, float32).

    `page` overrides the assumed paper colour. `keep_all` keeps every foreground blob
    instead of only the largest — a strip of four walk frames is four blobs, and
    dropping three of them would be unhelpful."""
    rgb = rgb.astype(np.float32)
    d = whiteness(rgb, page)
    paper = _paper(rgb, d)

    solid = ~paper
    lab, n = ndi.label(solid, structure=C8)
    if n > 1:                              # drop the watermark and stray specks
        sizes = ndi.sum(solid, lab, np.arange(1, n + 1))
        if keep_all:
            solid = np.isin(lab, [i for i in range(1, n + 1)
                                  if sizes[i - 1] >= 0.02 * sizes.max()])
        else:
            solid = lab == (int(np.argmax(sizes)) + 1)

    dist = ndi.distance_transform_edt(~solid)
    band = (dist > 0) & (dist <= FEATHER)

    alpha = solid.astype(np.float32)
    alpha[band] = np.clip((d[band] - A_LO) / (A_HI - A_LO), 0.0, 1.0)

    paper_v = float(np.median(rgb[dist > 12.0])) if (dist > 12.0).any() else 250.0
    soft = (alpha > 0.02) & (alpha < 0.995)
    if soft.any():
        a = np.maximum(alpha[soft], DECON_FLOOR)[:, None]
        rgb[soft] = np.clip(paper_v + (rgb[soft] - paper_v) / a, 0, 255)
    return rgb, alpha
