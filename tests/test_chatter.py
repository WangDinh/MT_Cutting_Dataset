import numpy as np
import pandas as pd

from mtcut.chatter import balanced_acc, binary_labels, bootstrap_auc, make_classifier, patch_features, recalls


def test_patch_features_are_per_bin_mean_and_std():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 128, 64)).astype(np.float32)
    f = patch_features(x)
    assert f.shape == (5, 256)
    assert np.allclose(f[:, :128], x.mean(axis=2)) and np.allclose(f[:, 128:], x.std(axis=2))


def test_binary_labels_drop_the_uncertain_class():
    keep, y = binary_labels(pd.Series([0.0, 1.0, 2.0, np.nan, 2.0]))
    assert keep.tolist() == [True, False, True, False, True]
    assert y.tolist() == [0, 1, 1]


def test_bootstrap_auc_perfect_and_chance():
    y = np.array([0] * 30 + [1] * 30)
    assert bootstrap_auc(y, y.astype(float), n=50) == (1.0, 1.0, 1.0)
    auc, lo, hi = bootstrap_auc(y, np.random.default_rng(1).random(60), n=200)
    assert lo < 0.5 < hi and lo <= auc <= hi
    assert balanced_acc(y, y.astype(float)) == 1.0


def test_recalls_at_a_fixed_threshold():
    y = np.array([0, 0, 0, 0, 1, 1])
    assert recalls(y, np.array([0.1, 0.2, 0.6, 0.4, 0.9, 0.4])) == (0.5, 0.75)


def test_classifier_separates_an_easy_problem():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, (60, 256)), rng.normal(1, 1, (60, 256))])
    y = np.array([0] * 60 + [1] * 60)
    assert (make_classifier().fit(x, y).predict(x) == y).mean() > 0.9
