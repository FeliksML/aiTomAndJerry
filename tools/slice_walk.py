"""Нарезка spriteTomaAndJerrywalling.png -> кадры + GIF ходьбы Тома."""
from PIL import Image
import os

SHEET = "spriteTomaAndJerrywalling.png"
OUT = "viz/anim"
COLS, ROWS = 8, 4          # 4 колонки Том + 4 колонки Джерри, 4 ряда направлений
ROW_NAMES = ["up", "down", "side_a", "side_b"]

im = Image.open(SHEET).convert("RGBA")
W, H = im.size
cw, ch = W // COLS, H // ROWS
os.makedirs(OUT, exist_ok=True)

frames = {}
for r, name in enumerate(ROW_NAMES):
    for c in range(4):                      # только Том (левая половина)
        box = (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
        f = im.crop(box)
        f.save(f"{OUT}/tom_{name}_{c}.png")
        frames.setdefault(name, []).append(f)

for name in ("down", "up"):
    fs = [f.convert("P", palette=Image.ADAPTIVE) for f in frames[name]]
    fs[0].save(f"{OUT}/tom_walk_{name}.gif", save_all=True,
               append_images=fs[1:], duration=120, loop=0)

print(f"cell {cw}x{ch}; кадров {sum(len(v) for v in frames.values())}")
