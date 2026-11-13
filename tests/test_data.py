import tempfile

from src.data import build_manifest, check_no_leakage
from tests.make_synthetic import make


def test_split_has_no_photograph_leakage_and_all_splits_present():
    with tempfile.TemporaryDirectory() as d:
        make(d, per_surface=120)
        df = build_manifest(d, seed=1)
        check_no_leakage(df)
        assert set(df["split"]) == {"train", "val", "test"}
        assert set(df["surface"]) == {"D", "W", "P"}
        assert set(df["label"]) == {0, 1}


def test_split_is_reproducible_for_same_seed():
    with tempfile.TemporaryDirectory() as d:
        make(d, per_surface=60)
        a, b = build_manifest(d, seed=3), build_manifest(d, seed=3)
        assert a["split"].tolist() == b["split"].tolist()
