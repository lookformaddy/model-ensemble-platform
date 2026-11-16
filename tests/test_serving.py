import pytest

from ensemble.ensembles import VotingEnsemble
from ensemble.serving import DelayedMember, EnsembleServer


@pytest.fixture()
def server(fitted_members, settings):
    ens = VotingEnsemble(fitted_members, mode="soft")
    return EnsembleServer(ens, "voting-ensemble@1", settings=settings)


def test_predict_one_uses_all_members(server, split):
    result = server.predict_one(split.X_test[0])
    assert result.label in (0, 1)
    assert 0.0 <= result.probability <= 1.0
    assert set(result.members_used) == {"logistic", "tree", "knn", "naive_bayes"}
    assert result.members_dropped == []
    assert result.degraded is False
    assert result.fallback_used is False
    assert result.latency_ms >= 0


def test_slow_member_is_dropped_gracefully(split, settings):
    from ensemble.learners import default_learners

    members = default_learners()
    slow = [
        members[0].clone(),
        members[1].clone(),
        DelayedMember(members[2].clone(), delay_s=0.30),
        members[3].clone(),
    ]
    ens = VotingEnsemble(slow, mode="soft").fit(split.X_train, split.y_train)
    server = EnsembleServer(ens, "demo-serving@1", settings=settings)
    result = server.predict_one(split.X_test[0], per_member_timeout_ms=100.0)
    assert result.degraded is True
    assert result.members_dropped == ["knn-slow"]
    assert set(result.members_used) == {"logistic", "tree", "naive_bayes"}
    assert result.label in (0, 1)
    # Served despite the timeout — no exception, and fast enough.
    assert result.latency_ms < 1000


def test_all_members_timeout_falls_back(split, settings):
    from ensemble.learners import default_learners

    members = [DelayedMember(m.clone(), delay_s=0.30) for m in default_learners()]
    ens = VotingEnsemble(members, mode="soft").fit(split.X_train, split.y_train)
    fallback = VotingEnsemble(
        [m.clone() for m in default_learners()], mode="soft"
    ).fit(split.X_train, split.y_train)
    server = EnsembleServer(
        ens, "slow@1", settings=settings, fallback=fallback, fallback_key="fallback@1"
    )
    result = server.predict_one(split.X_test[0], per_member_timeout_ms=50.0)
    assert result.fallback_used is True
    assert result.degraded is True
    assert result.label in (0, 1)


def test_all_members_timeout_without_fallback_raises(split, settings):
    from ensemble.learners import default_learners

    members = [DelayedMember(m.clone(), delay_s=0.30) for m in default_learners()]
    ens = VotingEnsemble(members, mode="soft").fit(split.X_train, split.y_train)
    server = EnsembleServer(ens, "slow@1", settings=settings)
    with pytest.raises(RuntimeError, match="all ensemble members timed out"):
        server.predict_one(split.X_test[0], per_member_timeout_ms=50.0)


def test_latency_stats_tracked(server, split):
    for row in split.X_test[:5]:
        server.predict_one(row)
    stats = server.latency_stats()
    assert stats["count"] == 5
    assert stats["p50_ms"] > 0
    assert stats["p95_ms"] >= stats["p50_ms"]
    assert stats["max_ms"] >= stats["p95_ms"]


def test_predict_batch(server, split):
    results = server.predict_batch(split.X_test[:3])
    assert len(results) == 3
    assert all(r.label in (0, 1) for r in results)
