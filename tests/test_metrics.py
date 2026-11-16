import math

from ensemble.metrics import (
    accuracy,
    brier_score,
    calibration_table,
    compute_metrics,
    log_loss,
    macro_f1,
    per_slice_metrics,
    roc_auc,
)


def test_roc_auc_perfect_and_worst():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert roc_auc([1, 1, 1], [0.9, 0.8, 0.7]) == 0.5  # single class present


def test_accuracy_and_macro_f1():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    assert accuracy(y_true, y_pred) == 0.6
    # Class 0: tp=1, fp=1, fn=1 -> F1 = 0.5. Class 1: tp=2, fp=1, fn=1 -> F1 = 2/3.
    # Macro F1 = (0.5 + 2/3) / 2 = 7/12.
    assert abs(macro_f1(y_true, y_pred) - 7 / 12) < 1e-9


def test_log_loss_and_brier_known_values():
    y_true = [1, 0]
    y_prob = [0.9, 0.1]
    assert abs(log_loss(y_true, y_prob) - (-math.log(0.9))) < 1e-9
    assert abs(brier_score(y_true, y_prob) - 0.01) < 1e-9


def test_compute_metrics_round_trip():
    m = compute_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.roc_auc == 1.0
    assert m.n == 4


def test_per_slice_metrics_split_by_segment():
    y_true = [0, 1, 0, 1]
    y_prob = [0.1, 0.9, 0.2, 0.8]
    segment = [0, 0, 1, 1]
    slices = per_slice_metrics(y_true, y_prob, segment)
    assert [s.slice_name for s in slices] == ["segment-0", "segment-1"]
    assert all(s.accuracy == 1.0 for s in slices)
    assert all(s.n == 2 for s in slices)


def test_calibration_table_and_ece():
    # Perfectly calibrated: predicted 0.2 -> observed 0.2, 0.8 -> 0.8.
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    y_prob = [0.2] * 5 + [0.8] * 5
    bins, ece = calibration_table(y_true, y_prob, n_bins=10)
    assert len(bins) == 10
    assert ece < 0.05
    # Miscalibrated: all predictions 0.9 but only half are positive.
    y_true2 = [1, 0] * 5
    bins2, ece2 = calibration_table(y_true2, [0.9] * 10, n_bins=10)
    assert ece2 > 0.3
    assert bins2[9].n == 10
