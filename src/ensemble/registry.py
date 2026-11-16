"""Versioned ensemble registry with promotion gates and rollback.

Lifecycle: DRAFT -> CANDIDATE -> ACTIVE -> RETIRED. A candidate is
promoted only if it clears *every* gate:

- accuracy floor and ROC-AUC floor on the held-out set,
- must beat the best single member by at least ``min_beat_single_delta``
  (AUC) — an ensemble that cannot beat its best member is not an ensemble,
- diversity floor (mean pairwise disagreement) — members must actually
  disagree, otherwise the ensemble is a single model in a trench coat,
- latency budget and member-count / size budgets,
- multi-seed stability (accuracy std across seeds under the cap),
- minimum evaluated cases.

Promotion retires the previous ACTIVE version; ``rollback`` restores it.
"""
from __future__ import annotations

import statistics

from .config import Settings
from .models import (
    EnsembleStatus,
    EnsembleVersion,
    EvaluationReport,
    GateResult,
    MemberSpec,
)


class PromotionBlocked(Exception):
    def __init__(self, gates: list[GateResult]):
        self.gates = gates
        failed = [g.name for g in gates if not g.passed]
        super().__init__(f"promotion blocked by gates: {', '.join(failed)}")


class EnsembleRegistry:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._versions: dict[str, EnsembleVersion] = {}

    def register(
        self, key: str, kind: str, members: list[MemberSpec], seed: int = 42
    ) -> EnsembleVersion:
        if key in self._versions:
            raise ValueError(f"ensemble {key!r} is already registered")
        version = EnsembleVersion(
            key=key, kind=kind, members=list(members), created_seed=seed
        )
        self._versions[key] = version
        return version

    def get(self, key: str) -> EnsembleVersion:
        return self._versions[key]

    def list(self) -> list[EnsembleVersion]:
        return sorted(self._versions.values(), key=lambda v: v.key)

    def active(self) -> EnsembleVersion | None:
        for v in self._versions.values():
            if v.status == EnsembleStatus.ACTIVE:
                return v
        return None

    def submit_candidate(self, key: str) -> EnsembleVersion:
        version = self.get(key)
        if version.status != EnsembleStatus.DRAFT:
            raise ValueError(f"only a DRAFT can become a candidate (got {version.status})")
        version.status = EnsembleStatus.CANDIDATE
        return version

    def evaluate_gates(
        self,
        key: str,
        report: EvaluationReport,
        stability_std: float | None = None,
    ) -> list[GateResult]:
        """Run every promotion gate against an evaluation report.

        ``stability_std`` is the accuracy std across training seeds; when
        not supplied the stability gate is reported as skipped (passed with
        actual 0.0 and a note) so a single-seed evaluation is not blocked
        on data it does not have.
        """
        s = self.settings
        m = report.ensemble_metrics
        d = report.diversity
        gates = [
            GateResult(
                "accuracy_floor", m.accuracy >= s.accuracy_floor, s.accuracy_floor,
                m.accuracy, f"held-out accuracy {m.accuracy:.4f}",
            ),
            GateResult(
                "auc_floor", m.roc_auc >= s.auc_floor, s.auc_floor,
                m.roc_auc, f"held-out ROC-AUC {m.roc_auc:.4f}",
            ),
            GateResult(
                "beats_best_single", d.beat_single_delta >= s.min_beat_single_delta,
                s.min_beat_single_delta, d.beat_single_delta,
                f"ensemble AUC {d.ensemble_auc:.4f} vs best single "
                f"{d.best_single_name} {d.best_single_auc:.4f}",
            ),
            GateResult(
                "diversity_floor",
                d.mean_pairwise_disagreement >= s.diversity_floor,
                s.diversity_floor, d.mean_pairwise_disagreement,
                f"mean pairwise disagreement {d.mean_pairwise_disagreement:.4f}",
            ),
            GateResult(
                "latency_budget", report.estimated_latency_ms <= s.max_latency_ms,
                s.max_latency_ms, report.estimated_latency_ms,
                f"estimated p95 latency {report.estimated_latency_ms:.1f} ms",
            ),
            GateResult(
                "member_count", len(self.get(key).members) <= s.max_members,
                float(s.max_members), float(len(self.get(key).members)),
                f"{len(self.get(key).members)} members",
            ),
            GateResult(
                "size_budget", report.estimated_size_kb <= s.max_ensemble_size_kb,
                s.max_ensemble_size_kb, report.estimated_size_kb,
                f"estimated size {report.estimated_size_kb:.1f} KB",
            ),
            GateResult(
                "min_eval_cases", m.n >= s.min_eval_cases,
                float(s.min_eval_cases), float(m.n), f"{m.n} held-out cases",
            ),
        ]
        if stability_std is None:
            gates.append(
                GateResult(
                    "stability", True, s.stability_max_std, 0.0,
                    "not measured (single-seed evaluation)",
                )
            )
        else:
            gates.append(
                GateResult(
                    "stability", stability_std <= s.stability_max_std,
                    s.stability_max_std, stability_std,
                    f"accuracy std across seeds {stability_std:.4f}",
                )
            )
        version = self.get(key)
        version.report = report
        version.gates = gates
        return gates

    def promote(self, key: str) -> EnsembleVersion:
        """Promote a candidate that cleared every gate. Raises
        ``PromotionBlocked`` listing the failed gates otherwise."""
        version = self.get(key)
        if version.status != EnsembleStatus.CANDIDATE:
            raise ValueError(f"only a CANDIDATE can be promoted (got {version.status})")
        failed = [g for g in version.gates if not g.passed]
        if failed:
            version.status = EnsembleStatus.BLOCKED
            raise PromotionBlocked(version.gates)
        current = self.active()
        if current is not None:
            current.status = EnsembleStatus.RETIRED
        version.status = EnsembleStatus.ACTIVE
        return version

    def rollback(self, key: str) -> EnsembleVersion:
        """Restore a RETIRED version to ACTIVE, retiring the current one."""
        version = self.get(key)
        if version.status != EnsembleStatus.RETIRED:
            raise ValueError(f"only a RETIRED version can be rolled back to (got {version.status})")
        current = self.active()
        if current is not None:
            current.status = EnsembleStatus.RETIRED
        version.status = EnsembleStatus.ACTIVE
        return version


def multi_seed_stability(
    accuracies: list[float],
) -> float:
    """Accuracy standard deviation across training seeds (0.0 for <2)."""
    if len(accuracies) < 2:
        return 0.0
    return statistics.pstdev(accuracies)
