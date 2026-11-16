from ensemble.diversity import (
    diversity_report,
    error_correlation,
    pairwise_disagreement,
    q_statistic,
)


def test_pairwise_disagreement_extremes():
    assert pairwise_disagreement([0, 1, 0], [0, 1, 0]) == 0.0
    assert pairwise_disagreement([0, 1, 0], [1, 0, 1]) == 1.0
    assert pairwise_disagreement([0, 1, 0], [0, 0, 0]) == 1 / 3


def test_q_statistic_identical_members_is_one():
    # Identical correctness: Q = 1 (no diversity at all).
    assert q_statistic([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_q_statistic_opposite_correctness_is_negative_one():
    # Perfectly anti-correlated correctness: Q = -1 (maximal diversity).
    assert q_statistic([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0


def test_q_statistic_degenerate_table_is_zero():
    # One member always right: denominator is 0 -> defined as 0.
    assert q_statistic([1, 1, 1], [1, 0, 1]) == 0.0


def test_error_correlation_identical_and_opposite():
    assert error_correlation([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0
    assert error_correlation([0, 1, 0, 1], [1, 0, 1, 0]) == -1.0
    # Constant error vector -> undefined correlation -> 0.
    assert error_correlation([0, 0, 0], [0, 1, 0]) == 0.0


def test_diversity_report_beats_best_single(fitted_members, split):
    member_probs = {
        m.name: m.predict_proba(split.X_test) for m in fitted_members
    }
    ens_probs = [
        sum(member_probs[name][i] for name in member_probs) / len(member_probs)
        for i in range(len(split.X_test))
    ]
    report = diversity_report(member_probs, ens_probs, split.y_test)
    assert 0.0 <= report.mean_pairwise_disagreement <= 1.0
    assert report.mean_pairwise_disagreement > 0.02  # members genuinely differ
    assert report.best_single_name in member_probs
    assert 0.5 < report.best_single_auc <= 1.0
    assert report.ensemble_auc > 0.5
    assert report.beat_single_delta == report.ensemble_auc - report.best_single_auc
    assert len(report.pairwise_disagreement) == 6  # 4 members -> 6 pairs
