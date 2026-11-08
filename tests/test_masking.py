import numpy as np

from mtcut.masking import (
    N_BINS,
    N_FRAMES,
    PATCH_SAMPLES,
    apply_mask,
    denormalize_db,
    normalize_db,
    patch_spectrogram_db,
    random_span_mask,
)


def test_patch_spectrogram_shape_and_range():
    x = (np.random.default_rng(0).standard_normal(PATCH_SAMPLES) * 1000).astype(np.int16)
    db = patch_spectrogram_db(x)
    assert db.shape == (N_BINS, N_FRAMES)
    assert db.min() >= -100 and db.max() <= 0


def test_normalize_roundtrip():
    db = np.linspace(-100, 0, 11)
    assert np.allclose(denormalize_db(normalize_db(db)), db)
    assert normalize_db(np.array([-100.0, 0.0])).tolist() == [-1.0, 1.0]


def test_random_span_mask_is_seeded_and_bounded():
    a = random_span_mask(np.random.default_rng(1))
    b = random_span_mask(np.random.default_rng(1))
    assert a.shape == (N_FRAMES,) and a.any() and not a.all()
    assert (a == b).all()


def test_apply_mask_only_touches_masked_frames():
    spec = np.ones((N_BINS, N_FRAMES), dtype=np.float32)
    mask = np.zeros(N_FRAMES, dtype=bool)
    mask[5:9] = True
    out = apply_mask(spec, mask)
    assert (out[:, mask] == 0).all() and (out[:, ~mask] == 1).all()
    assert (spec == 1).all()
