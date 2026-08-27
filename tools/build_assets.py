"""Turn the five source renders in assets-src/ into the eight assets the app wants.

    python tools/build_assets.py

Backdrops are a straight cover-crop to the 1920x1080 canvas. The three school
buildings are cut out of `night.png`, which is the wide shot all three were rendered
in — so the cards keep the exact lighting of the establishing shot.

The cut-out alpha is built from the render itself rather than hand-masked: a closed
silhouette of the lit stonework carries the body, and raw luminance carries the glow
and the light beams, so the halo fades out instead of ending on a hard edge. The
cards sit on a near-black panel, so the building's own dark windows staying slightly
transparent reads as depth, not as holes.
"""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image, ImageFilter

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets-src"
OUTS = [ROOT / "app" / "assets", ROOT / "viz" / "assets"]

CANVAS = (1920, 1080)
CARD = 620

BACKDROPS = {
    "bg-academy": "night.png",
    "bg-ppo": "PPO.png",
    "bg-ga": "GA.png",
    "bg-cmaes": "cma-es.png",
    "bg-final": "arena.png",
}

# Measured off night.png (1672x941). Bottom is the waterline: below it is reflection.
BUILDINGS = {
    "school-ppo": dict(x0=50, x1=414, y0=228, y1=645),
    "school-ga": dict(x0=665, x1=985, y0=308, y1=645),
    "school-cmaes": dict(x0=1238, x1=1566, y0=243, y1=645),
}


def cover(im: Image.Image, size) -> Image.Image:
    """Scale-and-centre-crop to exactly `size`, preserving aspect."""
    tw, th = size
    sw, sh = im.size
    scale = max(tw / sw, th / sh)
    nw, nh = round(sw * scale), round(sh * scale)
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return im.crop((left, top, left + tw, top + th))


def cutout(src: Image.Image, box: dict, size: int) -> Image.Image:
    """Square, bottom-anchored crop of one building with a render-derived alpha."""
    x0, x1, y0, y1 = box["x0"], box["x1"], box["y0"], box["y1"]
    cx = (x0 + x1) // 2
    side = round(max(x1 - x0, y1 - y0) * 1.10)
    # Bottom-anchored: the waterline is the natural lower edge, and every card then
    # sits its building on the same baseline.
    bx0, bx1 = cx - side // 2, cx - side // 2 + side
    by1 = y1 + round(side * 0.03)
    by0 = by1 - side
    crop = src.crop((bx0, by0, bx1, by1)).resize((size, size), Image.LANCZOS)

    a = np.asarray(crop).astype(np.float32)
    lum = a.mean(2)

    # Body: the lit stonework, closed so interior windows and roof shadow stay solid.
    body = Image.fromarray(((lum > 20) * 255).astype(np.uint8), "L")
    r = max(3, size // 44) | 1
    body = body.filter(ImageFilter.MaxFilter(r)).filter(ImageFilter.MinFilter(r))
    body = body.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2.5))
    body_a = np.asarray(body).astype(np.float32) / 255.0

    # Halo: raw luminance, so beams and lamp glow trail off instead of being clipped.
    halo = np.clip((lum - 7.0) / 42.0, 0, 1) ** 0.85

    alpha = np.clip(np.maximum(body_a, halo * 0.92), 0, 1)

    # Soften the four edges so nothing meets the card border on a hard line.
    fade = np.ones((size, size), np.float32)
    m = max(6, size // 26)
    ramp = np.linspace(0, 1, m, dtype=np.float32)
    fade[:m, :] *= ramp[:, None]
    fade[-m:, :] *= ramp[::-1][:, None]
    fade[:, :m] *= ramp[None, :]
    fade[:, -m:] *= ramp[::-1][None, :]
    alpha *= fade

    out = np.dstack([a, (alpha * 255)]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def save(im: Image.Image, name: str) -> None:
    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.png"
        im.save(p, optimize=True)
    kb = (OUTS[0] / f"{name}.png").stat().st_size / 1024
    print(f"  {name:<14} {im.size[0]}x{im.size[1]}  {kb:7.0f} KB")


def main() -> None:
    print("backdrops")
    for name, fn in BACKDROPS.items():
        save(cover(Image.open(SRC / fn).convert("RGB"), CANVAS), name)

    print("school cut-outs (from night.png)")
    night = Image.open(SRC / "night.png").convert("RGB")
    for name, box in BUILDINGS.items():
        save(cutout(night, box, CARD), name)


if __name__ == "__main__":
    main()
