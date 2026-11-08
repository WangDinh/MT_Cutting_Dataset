import numpy as np
import pandas as pd
import pytest

from mtcut import SAMPLE_RATE, discover_sessions, read_cutting_csv, read_labels, segment_table
from mtcut.config import IMI_DIR
from mtcut.labels import cohen_kappa, consensus, to_binary


@pytest.fixture(scope="module")
def sessions():
    return discover_sessions()


def test_discovers_all_sessions(sessions):
    assert sum(s.dataset == "IMI" for s in sessions) == 18
    assert [s.name for s in sessions if s.dataset == "KRPM"] == ["KRPM"]
    assert all(s.sensors == [0, 1, 2] for s in sessions)


@pytest.mark.parametrize("folder", ["Exp0-1", "Exp1-4", "Exp3-2"])  # BOM / no BOM / "Start,End"
def test_cutting_csv_header_variants(folder):
    df = read_cutting_csv(IMI_DIR / folder / "cutting.csv")
    assert list(df.columns) == ["start", "end", "duration"]
    assert (df["duration"] > 0).all()


def test_labels_align_with_cutting_csv(sessions):
    labels = read_labels()
    for s in sessions:
        if s.dataset != "IMI":
            continue
        lab = labels[labels["experiment"] == s.name]
        iv = s.intervals()
        np.testing.assert_allclose(lab[["start", "end"]].to_numpy(), iv[["start", "end"]].to_numpy())


def test_label_parsing():
    labels = read_labels()
    assert set(labels["direction"]) == {"up", "down"}
    assert set(labels["flutes"]) == {2, 5}
    assert labels[["chatter_op1", "chatter_op2"]].isin([0, 1, 2]).all().all()


def test_segment_table_joins_labels(sessions):
    seg = segment_table(sessions)
    assert seg["segment_id"].is_unique
    assert seg.loc[seg["dataset"] == "IMI", "chatter_op1"].notna().all()
    assert seg.loc[seg["dataset"] == "KRPM", "chatter_op1"].isna().all()


def test_segment_slicing(sessions):
    s = sessions[0]
    row = s.intervals().iloc[0]
    x = s.segment(0, 0)
    assert abs(len(x) - row["duration"] * SAMPLE_RATE) <= 1
    assert x.dtype == np.float32 and np.abs(x).max() <= 1.0


def test_consensus_and_kappa():
    a = pd.Series([0, 1, 2, 0, 2])
    b = pd.Series([0, 2, 2, 1, 0])
    assert consensus(a, b, "max").tolist() == [0, 2, 2, 1, 2]
    assert consensus(a, b, "mean").tolist() == [0, 2, 2, 1, 1]
    assert consensus(a, b, "agree").isna().tolist() == [False, True, False, True, True]
    assert to_binary(pd.Series([0, 1, 2]), uncertain_as=None).isna().tolist() == [False, True, False]
    assert cohen_kappa(a, a) == pytest.approx(1.0)
