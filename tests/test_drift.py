from ensemble.data import generate_dataset
from ensemble.drift import drift_report, ks_statistic, member_agreement, psi


def test_psi_zero_for_identical_distributions():
    values = [float(i) for i in range(100)]
    assert psi(values, values) == 0.0


def test_psi_large_for_shifted_distribution():
    reference = [float(i) for i in range(100)]
    shifted = [v + 50.0 for v in reference]
    assert psi(reference, shifted) > 0.25


def test_ks_zero_for_identical_and_one_for_disjoint():
    assert ks_statistic([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert ks_statistic([1.0, 2.0], [10.0, 20.0]) == 1.0


def test_member_agreement_full_and_none():
    member_probs = [[0.9, 0.1], [0.8, 0.2]]
    assert member_agreement(member_probs, [0.9, 0.1]) == 1.0
    assert member_agreement(member_probs, [0.1, 0.9]) == 0.0


def test_drift_report_ok_on_same_distribution(settings):
    # Halves of a large sample: PSI between two draws from the same
    # distribution stays under the warn threshold (small samples are
    # noisy — PSI between two 200-row halves can exceed 0.1 by chance).
    data = generate_dataset(n=2000, seed=42)
    ref_X = data.X[:1000]
    cur_X = data.X[1000:]
    report = drift_report(
        ref_X, cur_X, data.feature_names, 0.95, 0.94, settings,
    )
    assert report.level == "ok"
    assert report.flagged is False


def test_drift_report_critical_on_shifted_feature(settings):
    data = generate_dataset(n=400, seed=42)
    ref_X = data.X
    cur_X = [[v + 2.0 if j == 0 else v for j, v in enumerate(row)] for row in data.X]
    report = drift_report(
        ref_X, cur_X, data.feature_names, 0.95, 0.94, settings,
    )
    assert report.level == "critical"
    assert report.flagged is True
    assert report.psi_per_feature["f0"] > settings.psi_critical
    assert report.max_psi == report.psi_per_feature["f0"]


def test_drift_report_warn_on_moderate_shift(settings):
    data = generate_dataset(n=400, seed=42)
    ref_X = data.X
    cur_X = [[v + 0.6 if j == 0 else v for j, v in enumerate(row)] for row in data.X]
    report = drift_report(
        ref_X, cur_X, data.feature_names, 0.95, 0.94, settings,
    )
    assert report.level in ("warn", "critical")
    assert report.flagged is True
