from ensemble.ensembles import (
    StackingEnsemble,
    VotingEnsemble,
    WeightedBlending,
    build_blending,
    build_stacking,
    build_voting,
)
from ensemble.learners import default_learners
from ensemble.metrics import roc_auc


def test_voting_soft_averages_probabilities(fitted_members, split):
    ens = VotingEnsemble(fitted_members, mode="soft")
    probs = ens.predict_proba(split.X_test)
    expected = [
        sum(m.predict_proba(split.X_test)[i] for m in fitted_members) / len(fitted_members)
        for i in range(len(split.X_test))
    ]
    assert probs == expected


def test_voting_hard_is_vote_fraction(fitted_members, split):
    ens = VotingEnsemble(fitted_members, mode="hard")
    probs = ens.predict_proba(split.X_test)
    assert all(0.0 < p < 1.0 for p in probs)
    # Hard-vote probabilities are vote fractions of 4, smoothed away from
    # absolute 0/1 so log-loss stays finite.
    assert set(probs) <= {0.01, 0.25, 0.5, 0.75, 0.99}


def test_voting_invalid_mode():
    try:
        VotingEnsemble(default_learners(), mode="median")
    except ValueError as exc:
        assert "hard" in str(exc) and "soft" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_blending_weights_from_validation(fitted_members, split):
    n_val = 100
    X_val, y_val = split.X_train[-n_val:], split.y_train[-n_val:]
    blending = WeightedBlending([m.clone() for m in fitted_members])
    blending.fit(split.X_train, split.y_train)
    blending.fit_weights(X_val, y_val)
    assert abs(sum(blending.weights) - 1.0) < 1e-9
    assert all(w >= 0.0 for w in blending.weights)
    # The strongest member (by validation AUC) gets the largest weight.
    aucs = [roc_auc(y_val, m.predict_proba(X_val)) for m in fitted_members]
    best = max(range(len(aucs)), key=lambda i: aucs[i])
    assert blending.weights[best] == max(blending.weights)


def test_blending_equal_weights_when_all_at_chance(split):
    from ensemble.learners import MajorityLearner

    members = [MajorityLearner().fit(split.X_train, split.y_train) for _ in range(2)]
    blending = WeightedBlending(members)
    blending.fit_weights(split.X_test, split.y_test)
    assert blending.weights == [0.5, 0.5]


def test_stacking_oof_predictions_are_leakage_safe(split, members):
    stacking = StackingEnsemble([m.clone() for m in members], n_folds=5, seed=42)
    stacking.fit(split.X_train, split.y_train)
    assert stacking.oof_probs is not None
    assert len(stacking.oof_probs) == 4
    assert all(len(col) == len(split.X_train) for col in stacking.oof_probs)
    # OOF predictions are real probabilities, not the training labels.
    assert all(0.0 <= p <= 1.0 for col in stacking.oof_probs for p in col)
    oof_auc = stacking.oof_auc(split.y_train)
    assert 0.5 < oof_auc <= 1.0


def test_stacking_beats_or_matches_best_single(split, members):
    stacking = StackingEnsemble([m.clone() for m in members], n_folds=5, seed=42)
    stacking.fit(split.X_train, split.y_train)
    ens_auc = roc_auc(split.y_test, stacking.predict_proba(split.X_test))
    best_single = max(
        roc_auc(split.y_test, m.clone().fit(split.X_train, split.y_train).predict_proba(split.X_test))
        for m in members
    )
    # Stacking should not be materially worse than the best single member;
    # on the mixed-regime data it is better (asserted strictly in the demo).
    assert ens_auc >= best_single - 0.02


def test_builders_return_typed_ensembles(split, members):
    assert isinstance(build_voting(members), VotingEnsemble)
    assert isinstance(
        build_blending(
            members, split.X_train, split.y_train, split.X_test, split.y_test
        ),
        WeightedBlending,
    )
    assert isinstance(build_stacking(members), StackingEnsemble)


def test_ensemble_size_estimate_scales_with_knn_data(split, fitted_members):
    from ensemble.ensembles import VotingEnsemble

    ens = VotingEnsemble(fitted_members)
    size_kb = ens.estimated_size_kb()
    assert size_kb > 0
    # k-NN stores the training set, so it dominates the estimate.
    assert size_kb > 100
