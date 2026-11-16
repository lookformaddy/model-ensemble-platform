import pytest

from ensemble.evaluation import member_specs
from ensemble.learners import default_learners
from ensemble.models import EnsembleStatus
from ensemble.registry import PromotionBlocked, multi_seed_stability


def _register(registry, reports, key="stacking-ensemble@1"):
    registry.register(key, "stacking", member_specs(default_learners()), seed=42)
    registry.submit_candidate(key)
    return registry.evaluate_gates(key, reports[key], stability_std=0.01)


def test_gates_pass_for_strong_ensemble(registry, reports):
    gates = _register(registry, reports)
    assert all(g.passed for g in gates), [(g.name, g.actual) for g in gates if not g.passed]
    promoted = registry.promote("stacking-ensemble@1")
    assert promoted.status == EnsembleStatus.ACTIVE
    assert registry.active().key == "stacking-ensemble@1"


def test_promotion_blocked_by_accuracy_floor(registry, reports, settings):
    settings.accuracy_floor = 0.99  # unreachable on this data
    registry2 = registry.__class__(settings)
    gates = _register(registry2, reports)
    failed = [g for g in gates if not g.passed]
    assert any(g.name == "accuracy_floor" for g in failed)
    with pytest.raises(PromotionBlocked):
        registry2.promote("stacking-ensemble@1")
    assert registry2.get("stacking-ensemble@1").status == EnsembleStatus.BLOCKED


def test_promotion_blocked_when_ensemble_loses_to_best_single(registry, reports, settings):
    settings.min_beat_single_delta = 0.5  # impossible delta
    registry2 = registry.__class__(settings)
    gates = _register(registry2, reports)
    assert any(g.name == "beats_best_single" and not g.passed for g in gates)


def test_promote_retires_previous_active(registry, reports):
    _register(registry, reports, "stacking-ensemble@1")
    registry.promote("stacking-ensemble@1")
    _register(registry, reports, "voting-ensemble@1")
    registry.promote("voting-ensemble@1")
    assert registry.active().key == "voting-ensemble@1"
    assert registry.get("stacking-ensemble@1").status == EnsembleStatus.RETIRED


def test_rollback_restores_previous(registry, reports):
    _register(registry, reports, "stacking-ensemble@1")
    registry.promote("stacking-ensemble@1")
    _register(registry, reports, "voting-ensemble@1")
    registry.promote("voting-ensemble@1")
    rolled = registry.rollback("stacking-ensemble@1")
    assert rolled.status == EnsembleStatus.ACTIVE
    assert registry.active().key == "stacking-ensemble@1"
    assert registry.get("voting-ensemble@1").status == EnsembleStatus.RETIRED


def test_rollback_requires_retired(registry, reports):
    _register(registry, reports, "stacking-ensemble@1")
    with pytest.raises(ValueError, match="RETIRED"):
        registry.rollback("stacking-ensemble@1")


def test_duplicate_registration_rejected(registry):
    registry.register("voting-ensemble@1", "voting", member_specs(default_learners()))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("voting-ensemble@1", "voting", member_specs(default_learners()))


def test_multi_seed_stability():
    assert multi_seed_stability([0.8]) == 0.0
    assert multi_seed_stability([]) == 0.0
    assert abs(multi_seed_stability([0.80, 0.82, 0.81]) - 0.00816496580927726) < 1e-9
