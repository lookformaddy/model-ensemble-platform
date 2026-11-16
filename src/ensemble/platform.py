"""Orchestration: build, evaluate, gate, promote, serve, monitor.

``EnsemblePlatform`` wires the pieces together for one dataset and one
settings object: train the base learners, build the three ensembles,
evaluate them on the same held-out set, register the winner, run the
promotion gates, and — if they pass — promote it to ACTIVE. It also owns
the serving path (via ``EnsembleServer``) and the drift check that can
fall back to the previous stable ensemble.
"""
from __future__ import annotations

from .audit import AuditLog
from .config import Settings
from .data import generate_dataset, train_test_split
from .drift import drift_report, member_agreement
from .ensembles import StackingEnsemble, VotingEnsemble, WeightedBlending
from .evaluation import member_specs, train_evaluate_and_build
from .learners import default_learners
from .models import DriftReport, EvaluationReport
from .registry import EnsembleRegistry, PromotionBlocked, multi_seed_stability
from .serving import EnsembleServer


class EnsemblePlatform:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.registry = EnsembleRegistry(self.settings)
        self.audit = AuditLog()
        self._ensembles: dict[str, object] = {}
        self._servers: dict[str, EnsembleServer] = {}

    # --- build / evaluate / promote --------------------------------------
    def build_all(self, seed: int | None = None) -> dict[str, EvaluationReport]:
        """Train members, build all three ensembles, evaluate, register."""
        seed = seed if seed is not None else self.settings.default_seed
        data = generate_dataset(
            n=self.settings.n_samples,
            n_features=self.settings.n_features,
            seed=seed,
        )
        split = train_test_split(data, self.settings.test_fraction, seed=seed)
        members = default_learners(seed)
        reports, ensembles = train_evaluate_and_build(
            members,
            split.X_train, split.y_train,
            split.X_test, split.y_test,
            split.segment_test,
            self.settings,
        )
        self._ensembles = ensembles
        for key, report in reports.items():
            kind = key.rsplit("@", 1)[0].split("-")[0]
            version = self.registry.register(key, kind, member_specs(members), seed=seed)
            # Attach the evaluation so the API/CLI can show metrics and
            # diversity without re-running the evaluation.
            version.report = report
            self.audit.append("register", {"key": key, "kind": kind})
        self._split = split
        self._data = data
        return reports

    def promote_best(self, reports: dict[str, EvaluationReport]) -> str:
        """Register the best ensemble as a candidate, run the gates
        (including multi-seed stability), and promote it if they pass.

        Returns the promoted key. Raises ``PromotionBlocked`` otherwise.
        """
        best_key = max(reports, key=lambda k: reports[k].ensemble_metrics.roc_auc)
        stability = self._stability_std(best_key)
        version = self.registry.submit_candidate(best_key)
        self.audit.append("candidate", {"key": best_key})
        self.registry.evaluate_gates(best_key, reports[best_key], stability_std=stability)
        try:
            self.registry.promote(best_key)
        except PromotionBlocked:
            self.audit.append("promotion_blocked", {"key": best_key})
            raise
        self.audit.append("promote", {"key": best_key})
        return best_key

    def _stability_std(self, key: str) -> float:
        """Accuracy std of the same ensemble kind across training seeds."""
        kind = key.rsplit("@", 1)[0].split("-")[0]
        accuracies: list[float] = []
        for seed in self.settings.stability_seeds:
            data = generate_dataset(
                n=self.settings.n_samples,
                n_features=self.settings.n_features,
                seed=seed,
            )
            split = train_test_split(data, self.settings.test_fraction, seed=seed)
            members = default_learners(seed)
            n_val = max(20, int(0.2 * len(split.X_train)))
            X_fit, y_fit = split.X_train[:-n_val], split.y_train[:-n_val]
            X_val, y_val = split.X_train[-n_val:], split.y_train[-n_val:]
            if kind == "voting":
                ens = VotingEnsemble([m.clone() for m in members], mode="soft").fit(X_fit, y_fit)
            elif kind == "blending":
                ens = WeightedBlending([m.clone() for m in members])
                ens.fit(X_fit, y_fit)
                ens.fit_weights(X_val, y_val)
            else:
                ens = StackingEnsemble(
                    [m.clone() for m in members],
                    n_folds=self.settings.n_folds, seed=seed,
                ).fit(X_fit, y_fit)
            probs = ens.predict_proba(split.X_test)
            accuracies.append(
                sum(1 for p, t in zip(probs, split.y_test) if (1 if p >= 0.5 else 0) == t)
                / len(split.y_test)
            )
        return multi_seed_stability(accuracies)

    # --- serving ---------------------------------------------------------
    def server_for(self, key: str) -> EnsembleServer:
        if key not in self._servers:
            ensemble = self._ensembles[key]
            active = self.registry.active()
            fallback = None
            fallback_key = None
            if active is not None and active.key != key:
                fallback = self._ensembles.get(active.key)
                fallback_key = active.key
            self._servers[key] = EnsembleServer(
                ensemble, key, settings=self.settings,
                fallback=fallback, fallback_key=fallback_key,
            )
        return self._servers[key]

    def active_server(self) -> EnsembleServer | None:
        active = self.registry.active()
        if active is None:
            return None
        return self.server_for(active.key)

    # --- drift -----------------------------------------------------------
    def check_drift(
        self, current_X: list[list[float]], current_probs: list[list[float]],
        current_ensemble_probs: list[float],
    ) -> DriftReport:
        """Compare a current serving window against the training reference.

        ``current_probs`` are the per-member probabilities on the window;
        ``current_ensemble_probs`` are the ensemble's. A ``critical``
        report falls back to the previous stable ensemble (audited).
        """
        split = self._split
        ref_member_probs = [
            m.predict_proba(split.X_test) for m in self._ensembles[
                self.registry.active().key if self.registry.active() else "stacking-ensemble@1"
            ].members
        ]
        ref_ens_probs = self._ensembles[
            self.registry.active().key if self.registry.active() else "stacking-ensemble@1"
        ].predict_proba(split.X_test)
        report = drift_report(
            split.X_train,
            current_X,
            self._data.feature_names,
            member_agreement(ref_member_probs, ref_ens_probs),
            member_agreement(current_probs, current_ensemble_probs),
            self.settings,
        )
        if report.level == "critical":
            self.audit.append(
                "drift_fallback",
                {"max_psi": report.max_psi, "level": report.level},
            )
        return report
