import numpy as np

from mtcut.detect import frame_dataset, gap_features, loso_detectability
from mtcut.inpaint import LOW_BAND_BINS, fill_mean
from mtcut.masking import N_BINS, N_FRAMES


def _patches(n=6, seed=0):
    rng = np.random.default_rng(seed)
    clean = rng.normal(-60, 6, size=(n, N_BINS, N_FRAMES)).astype(np.float32)  # noisy, like real spectrograms
    mask = np.zeros((n, N_FRAMES), dtype=bool)
    mask[:, 20:34] = True
    sessions = np.array(["a", "a", "b", "b", "c", "c"][:n])
    return clean, mask, sessions


def test_features_cover_every_gap_frame_including_at_the_border():
    clean, mask, _ = _patches()
    assert gap_features(clean[0], mask[0]).shape == (14, 4 + LOW_BAND_BINS)
    edge = np.zeros(N_FRAMES, dtype=bool)
    edge[:5] = True
    assert gap_features(clean[0], edge).shape[0] == 5


def test_frame_dataset_is_balanced_real_vs_filled():
    clean, mask, sessions = _patches()
    _, y, g = frame_dataset(clean, fill_mean(np.where(mask[:, None, :], 0.0, clean), mask), mask, sessions)
    assert y.mean() == 0.5 and set(g) == {"a", "b", "c"}


def test_flat_fill_is_easy_to_detect_and_identical_fill_is_not():
    clean, mask, sessions = _patches()
    flat = fill_mean(np.where(mask[:, None, :], 0.0, clean), mask)
    res = loso_detectability(clean, flat, mask, sessions)
    assert res["auc"] > 0.95 and set(res) == {"auc", "a", "b", "c"}
    assert abs(loso_detectability(clean, clean.copy(), mask, sessions)["auc"] - 0.5) < 1e-6
