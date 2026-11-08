import os

# torch and Anaconda's numpy/scipy each bundle an OpenMP runtime; see scripts/eval_inpainting.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mtcut.inpaint import fill_linear, fill_mean, gap_stats, masked_errors  # noqa: E402
from mtcut.masking import N_BINS, N_FRAMES  # noqa: E402


def _ramp_patch():
    """A patch whose level rises linearly in time, so linear interpolation is exact."""
    t = np.linspace(-80, -40, N_FRAMES)
    return np.tile(t, (1, N_BINS, 1)).astype(np.float32)


def _mask(*spans):
    m = np.zeros((1, N_FRAMES), dtype=bool)
    for a, b in spans:
        m[0, a:b] = True
    return m


def test_linear_recovers_a_ramp_and_observed_frames_are_untouched():
    x, m = _ramp_patch(), _mask((10, 20), (40, 45))
    observed = np.where(m[:, None, :], 0.0, x)
    out = fill_linear(observed, m)
    assert np.allclose(out, x, atol=1e-4)
    assert (out[:, :, ~m[0]] == x[:, :, ~m[0]]).all()


def test_mean_fill_uses_only_observed_frames():
    x, m = _ramp_patch(), _mask((0, 32))
    observed = np.where(m[:, None, :], 1e6, x)  # garbage in the gap must not leak into the fill
    out = fill_mean(observed, m)
    assert np.allclose(out[0][:, m[0]], x[0][:, ~m[0]].mean(axis=1, keepdims=True))


def test_edge_gap_is_held_not_extrapolated():
    x, m = _ramp_patch(), _mask((0, 8))
    out = fill_linear(np.where(m[:, None, :], 0.0, x), m)
    assert np.allclose(out[0][:, :8], x[0][:, 8:9], atol=1e-4)


def test_metrics_perfect_and_blur():
    x, m = _ramp_patch(), _mask((10, 20))
    assert masked_errors(x, x, m).mae_db.iloc[0] == 0
    assert masked_errors(x + 2, x, m).mae_db.iloc[0] == pytest.approx(2.0)
    blur = np.where(m[:, None, :], x[:, :, 10:11], x)  # constant in the gap
    assert gap_stats(blur, x, m, bins=slice(0, N_BINS)).std_ratio.iloc[0] == pytest.approx(0.0, abs=1e-3)


def test_diffusion_shapes_and_repaint_keeps_known_region():
    torch = pytest.importorskip("torch")
    from mtcut.diffusion import UNet, Diffusion, inpaint

    model = UNet(base=8).eval()
    diff = Diffusion(T=5, device="cpu")
    x = torch.randn(2, 1, N_BINS, N_FRAMES)
    assert model(x, torch.zeros(2, dtype=torch.long)).shape == x.shape
    assert diff.loss(model, x).ndim == 0

    rng = np.random.default_rng(0)
    db = rng.uniform(-90, -20, size=(2, N_BINS, N_FRAMES)).astype(np.float32)
    mask = np.zeros((2, N_FRAMES), dtype=bool)
    mask[:, 20:30] = True
    out = inpaint(model, diff, np.where(mask[:, None, :], 0.0, db), mask, n_resample=2)
    assert out.shape == db.shape
    assert (out[:, :, ~mask[0]] == db[:, :, ~mask[0]]).all()  # observed frames returned exactly
    assert (out >= -100).all() and (out <= 0).all()
