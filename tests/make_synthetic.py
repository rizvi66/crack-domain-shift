"""Create a tiny fake SDNET2018-style dataset so the pipeline can be tested without the real data.

Surfaces get different background textures (so a domain shift exists); cracked images
contain dark random polylines. This is NOT real data and is used only for smoke tests.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BG = {"D": (150, 150, 150), "W": (200, 190, 170), "P": (90, 90, 95)}
DIRS = {"D": ("UD", "CD"), "W": ("UW", "CW"), "P": ("UP", "CP")}


def make(root, per_surface=240, size=64, seed=0):
    rng = np.random.default_rng(seed)
    for s, (u, c) in DIRS.items():
        for label, d in enumerate((u, c)):
            out = Path(root) / s / d
            out.mkdir(parents=True, exist_ok=True)
            n = int(per_surface * (0.3 if label else 0.7))
            for k in range(n):
                photo = 7000 + rng.integers(0, 40)  # several crops share a "photograph"
                noise = rng.normal(0, 12 if s != "P" else 25, (size, size, 3))
                arr = np.clip(np.array(BG[s]) + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
                if label:
                    dr = ImageDraw.Draw(img)
                    pts = [(int(rng.integers(0, size)), int(rng.integers(0, size))) for _ in range(4)]
                    dr.line(pts, fill=(25, 25, 25), width=2)
                img.save(out / f"{photo}-{k}.jpg", quality=95)


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
