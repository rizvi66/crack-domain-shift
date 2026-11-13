"""Data handling for SDNET2018 crack classification across surface types.

SDNET2018 folder layout (one folder per surface, two class folders each):
    D/CD  D/UD   bridge decks     (cracked / uncracked)
    W/CW  W/UW   walls
    P/CP  P/UP   pavements

Each 256x256 image is a crop of a larger photograph. File names look like
"7001-115.jpg": the number before the dash identifies the source photograph.
Crops from one photograph look alike, so splitting at the *photograph* level
(not the crop level) is necessary to avoid train/test leakage.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

SURFACES = ("D", "W", "P")
SURFACE_NAMES = {"D": "deck", "W": "wall", "P": "pavement"}
# (uncracked folder, cracked folder); label 0 = uncracked, 1 = cracked
CLASS_DIRS = {"D": ("UD", "CD"), "W": ("UW", "CW"), "P": ("UP", "CP")}
IMG_EXT = {".jpg", ".jpeg", ".png"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _find_dir(root: Path, name: str, parent: str) -> Path:
    cand = root / parent / name
    if cand.is_dir():
        return cand
    for p in root.rglob(name):  # tolerate extra nesting (e.g. Kaggle layouts)
        if p.is_dir() and p.parent.name == parent:
            return p
    raise FileNotFoundError(f"Could not find folder '{parent}/{name}' under {root}")


def group_id(surface: str, filename: str) -> str:
    stem = Path(filename).stem
    m = re.match(r"^(\d+)[-_]", stem)
    return f"{surface}:{m.group(1) if m else stem}"


def assign_splits(df: pd.DataFrame, seed: int, fractions=(0.7, 0.15, 0.15)) -> pd.Series:
    """Assign train/val/test by photograph group, separately for each surface."""
    assert abs(sum(fractions) - 1.0) < 1e-9
    split = pd.Series(index=df.index, dtype=object)
    rng = np.random.default_rng(seed)
    for s in df["surface"].unique():
        sub = df[df["surface"] == s]
        groups = np.array(sorted(sub["group"].unique()))
        rng.shuffle(groups)
        n = len(groups)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        where = {g: "train" for g in groups[:n_tr]}
        where.update({g: "val" for g in groups[n_tr:n_tr + n_va]})
        where.update({g: "test" for g in groups[n_tr + n_va:]})
        split.loc[sub.index] = sub["group"].map(where)
    return split


def build_manifest(root, seed: int = 42, fractions=(0.7, 0.15, 0.15),
                   max_per_class: int | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns: path, surface, label, group, split.

    max_per_class caps images per (surface, label) in the train split; val and
    test are capped at max_per_class // 4. Meant for quick smoke tests only.
    """
    root = Path(root)
    rows = []
    for s in SURFACES:
        for label, cname in enumerate(CLASS_DIRS[s]):
            d = _find_dir(root, cname, s)
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    rows.append((str(p), s, label, group_id(s, p.name)))
    df = pd.DataFrame(rows, columns=["path", "surface", "label", "group"])
    if df.empty:
        raise RuntimeError(f"No images found under {root}")
    df["split"] = assign_splits(df, seed, fractions)
    if max_per_class:
        parts = []
        shuffled = df.sample(frac=1.0, random_state=seed)
        for split, cap in (("train", max_per_class), ("val", max(1, max_per_class // 4)),
                           ("test", max(1, max_per_class // 4))):
            sub = shuffled[shuffled["split"] == split]
            parts.append(sub.groupby(["surface", "label"]).head(cap))
        df = pd.concat(parts).sort_values("path").reset_index(drop=True)
    return df.reset_index(drop=True)


def check_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any photograph group appears in more than one split."""
    n_splits = df.groupby("group")["split"].nunique()
    bad = n_splits[n_splits > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} photograph groups span several splits")


def describe(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["surface", "split", "label"]).size().unstack("label", fill_value=0)
    t.columns = ["uncracked", "cracked"]
    t["total"] = t.sum(axis=1)
    t["pct_cracked"] = (100 * t["cracked"] / t["total"]).round(1)
    return t


def make_transforms(img_size: int = 224, train: bool = False):
    ops = [T.Resize((img_size, img_size))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(10)]
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


class CrackDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 224, train: bool = False):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tf = make_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), torch.tensor(self.labels[i]), i
