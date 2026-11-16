"""Golden-path demo runner.

Runs the whole platform end to end on seeded synthetic data:

1. Train the four base learners and compare them.
2. Build voting / blending / stacking and compare — stacking must beat
   the best single member.
3. Register the winner, run the promotion gates, promote it to ACTIVE.
4. Serve with a deliberately slow member and show graceful degradation.
5. Feed a shifted input window and show drift being flagged.

Returns a dict of headline numbers so ``scripts/demo.py`` (and tests)
can assert on the outcome.
"""
from __future__ import annotations

from .config import Settings
from .data import generate_dataset
from .deployment import DeploymentManager
from .ensembles import VotingEnsemble
from .learners import default_learners
from .platform import EnsemblePlatform
from .registry import PromotionBlocked
from .serving import DelayedMember, EnsembleServer


def _line(char: str = "-", width: int = 72) -> str:
    return char * width


def run_demo(settings: Settings | None = None) -> dict:
    s = settings or Settings()
    print("Model Ensemble Platform — demo")
    print(_line("="))
    platform = EnsemblePlatform(s)
    reports = platform.build_all()

    print("\n[1] Base learners (held-out, n=%d)" % next(iter(reports.values())).ensemble_metrics.n)
    member_metrics = {}
    for key, report in reports.items():
        for name, m in sorted(report.member_metrics.items()):
            member_metrics[name] = m
    for name, m in sorted(member_metrics.items(), key=lambda kv: -kv[1].roc_auc):
        print(f"    {name:12s} acc={m.accuracy:.4f}  auc={m.roc_auc:.4f}  "
              f"f1={m.macro_f1:.4f}  logloss={m.log_loss:.4f}")

    print("\n[2] Ensembles (same held-out set)")
    for key, report in reports.items():
        m = report.ensemble_metrics
        d = report.diversity
        print(f"    {key:22s} acc={m.accuracy:.4f}  auc={m.roc_auc:.4f}  "
              f"f1={m.macro_f1:.4f}  brier={m.brier:.4f}  "
              f"beats-best-single +{d.beat_single_delta:.4f} AUC")
    best_key = max(reports, key=lambda k: reports[k].ensemble_metrics.roc_auc)
    best = reports[best_key]
    best_single = best.diversity.best_single_name
    stacking_beats_single = best.diversity.beat_single_delta > 0
    print(f"\n    Winner: {best_key} — beats best single ({best_single}, "
          f"AUC {best.diversity.best_single_auc:.4f}) by "
          f"+{best.diversity.beat_single_delta:.4f} AUC")
    print(f"    Diversity: mean pairwise disagreement "
          f"{best.diversity.mean_pairwise_disagreement:.4f}, "
          f"mean Q {best.diversity.mean_q_statistic:.4f}, "
          f"mean error corr {best.diversity.mean_error_correlation:.4f}")
    print(f"    Calibration: ECE {best.expected_calibration_error:.4f}")
    for sl in best.per_slice:
        print(f"    Slice {sl.slice_name}: n={sl.n} acc={sl.accuracy:.4f} auc={sl.roc_auc:.4f}")

    print("\n[3] Promotion gates — %s" % best_key)
    promoted = False
    try:
        promoted_key = platform.promote_best(reports)
        promoted = promoted_key == best_key
    except PromotionBlocked as exc:
        print("    BLOCKED:")
        for g in exc.gates:
            mark = "PASS" if g.passed else "FAIL"
            print(f"    [{mark}] {g.name}: {g.actual:.4f} (threshold {g.threshold:.4f}) — {g.detail}")
        return {
            "stacking_beats_single": stacking_beats_single,
            "promoted": False,
            "best_key": best_key,
        }
    version = platform.registry.get(best_key)
    for g in version.gates:
        mark = "PASS" if g.passed else "FAIL"
        print(f"    [{mark}] {g.name}: {g.actual:.4f} (threshold {g.threshold:.4f}) — {g.detail}")
    print(f"    PROMOTED: {best_key} is now ACTIVE")

    print("\n[4] Serving — latency-budgeted, slow member degraded gracefully")
    split = platform._split
    members = default_learners(s.default_seed)
    slow_members = [
        members[0].clone(),
        members[1].clone(),
        DelayedMember(members[2].clone(), delay_s=0.30),
        members[3].clone(),
    ]
    demo_ens = VotingEnsemble(slow_members, mode="soft").fit(split.X_train, split.y_train)
    server = EnsembleServer(demo_ens, "demo-serving@1", settings=s)
    row = split.X_test[0]
    result = server.predict_one(row, per_member_timeout_ms=100.0)
    print(f"    request: label={result.label} p={result.probability:.3f} "
          f"latency={result.latency_ms:.1f} ms")
    print(f"    members used: {result.members_used}")
    print(f"    members dropped (timed out): {result.members_dropped}")
    print(f"    degraded={result.degraded} (served anyway — no failed request)")
    degraded_ok = result.degraded and result.members_dropped and result.label in (0, 1)

    # Canary / shadow: production is the promoted winner, the candidate
    # is the runner-up blending ensemble.
    deploy = DeploymentManager(
        production_key=best_key, candidate_key="blending-ensemble@1",
        canary_fraction=s.canary_fraction,
        shadow_agreement_floor=s.shadow_agreement_floor,
    )
    canary_ids = [f"req-{i}" for i in range(200)]
    canary_n = sum(1 for rid in canary_ids if deploy.is_canary(rid))
    active = platform.registry.active()
    active_ens = platform._ensembles[active.key]
    candidate_ens = platform._ensembles["blending-ensemble@1"]
    for rid, shadow_row in zip(canary_ids[:50], split.X_test[:50]):
        deploy.record_shadow(
            rid,
            active_ens.predict([shadow_row])[0],
            candidate_ens.predict([shadow_row])[0],
            active_ens.predict_proba([shadow_row])[0],
            candidate_ens.predict_proba([shadow_row])[0],
        )
    shadow = deploy.shadow_report()
    print(f"    canary: {canary_n}/200 requests routed to candidate "
          f"(target {s.canary_fraction:.0%})")
    print(f"    shadow: {shadow['n_shadowed']} scored, agreement "
          f"{shadow['agreement']:.3f} (floor {shadow['floor']:.2f})")

    print("\n[5] Drift — shifted input window")
    shifted = generate_dataset(n=400, n_features=s.n_features, seed=999)
    shifted_X = [[v + 2.0 if j == 0 else v for j, v in enumerate(row)] for row in shifted.X]
    active_members = active_ens.members
    current_member_probs = [m.predict_proba(shifted_X) for m in active_members]
    current_ens_probs = active_ens.predict_proba(shifted_X)
    drift = platform.check_drift(shifted_X, current_member_probs, current_ens_probs)
    worst_feature = max(drift.psi_per_feature, key=lambda k: drift.psi_per_feature[k])
    print(f"    worst feature: {worst_feature} PSI={drift.psi_per_feature[worst_feature]:.3f} "
          f"(max PSI {drift.max_psi:.3f}, max KS {drift.max_ks:.3f})")
    print(f"    agreement drift: {drift.agreement_drift:.3f} "
          f"(ref {drift.reference_agreement:.3f} -> current {drift.current_agreement:.3f})")
    print(f"    level={drift.level} flagged={drift.flagged} "
          f"— stale-ensemble fallback armed")
    print(f"\n    audit chain valid: {platform.audit.verify()} "
          f"({len(platform.audit.entries())} entries)")
    print(_line("="))
    print("Demo complete.")
    return {
        "stacking_beats_single": stacking_beats_single,
        "promoted": promoted,
        "best_key": best_key,
        "best_auc": best.ensemble_metrics.roc_auc,
        "best_single_auc": best.diversity.best_single_auc,
        "beat_single_delta": best.diversity.beat_single_delta,
        "mean_disagreement": best.diversity.mean_pairwise_disagreement,
        "ece": best.expected_calibration_error,
        "serving_degraded_gracefully": degraded_ok,
        "drift_flagged": drift.flagged,
        "drift_level": drift.level,
        "audit_valid": platform.audit.verify(),
    }
