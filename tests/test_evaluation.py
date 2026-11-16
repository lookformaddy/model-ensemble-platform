from ensemble.evaluation import best_ensemble, evaluate, train_and_evaluate_all
from ensemble.ensembles import VotingEnsemble


def test_evaluate_report_shape(fitted_members, split):
    ens = VotingEnsemble(fitted_members, mode="soft")
    report = evaluate(ens, "voting-ensemble@1", split.X_test, split.y_test, split.segment_test)
    assert report.ensemble_key == "voting-ensemble@1"
    assert set(report.member_metrics) == {"logistic", "tree", "knn", "naive_bayes"}
    assert report.ensemble_metrics.n == len(split.y_test)
    assert 0.5 < report.ensemble_metrics.roc_auc <= 1.0
    assert len(report.per_slice) == 2
    assert len(report.calibration) == 10
    assert 0.0 <= report.expected_calibration_error <= 1.0
    assert report.estimated_latency_ms > 0
    assert report.estimated_size_kb > 0
    assert report.diversity.best_single_name in report.member_metrics


def test_train_and_evaluate_all_returns_three(reports):
    assert set(reports) == {
        "voting-ensemble@1",
        "blending-ensemble@1",
        "stacking-ensemble@1",
    }
    for report in reports.values():
        assert report.ensemble_metrics.accuracy > 0.6
        assert report.ensemble_metrics.roc_auc > 0.7


def test_best_ensemble_is_highest_auc(reports):
    best = best_ensemble(reports)
    aucs = {k: v.ensemble_metrics.roc_auc for k, v in reports.items()}
    assert aucs[best] == max(aucs.values())


def test_stacking_wins_on_mixed_regime_data(reports):
    # The planted mixture (linear + interaction regimes) is exactly the
    # case where stacking should beat the best single member.
    stacking = reports["stacking-ensemble@1"]
    assert stacking.diversity.beat_single_delta > 0
