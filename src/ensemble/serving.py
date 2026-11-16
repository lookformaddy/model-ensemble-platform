"""Latency-budgeted ensemble serving with graceful degradation.

Each member is invoked in its own thread with a per-member timeout. A
member that does not answer inside its timeout is *dropped* — the ensemble
serves from the members that did answer, and the result is marked
``degraded`` so callers and monitors can see it. If every member times
out, the server falls back to the registered fallback ensemble (the
previous ACTIVE version) instead of failing the request.

Latency percentiles (p50/p95) are tracked in-memory for the latency
budget gate and for the serving report.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from .config import Settings
from .ensembles import Ensemble
from .learners import BaseLearner
from .models import ServeResult


class DelayedMember(BaseLearner):
    """A wrapper that sleeps before delegating — used to exercise the
    timeout path deterministically (a genuinely slow member in prod)."""

    kind = "delayed"

    def __init__(self, inner: BaseLearner, delay_s: float):
        self.inner = inner
        self.delay_s = delay_s
        self.kind = inner.kind  # report the wrapped kind to the ensemble

    def fit(self, X, y):
        self.inner.fit(X, y)
        return self

    def predict_proba(self, X):
        time.sleep(self.delay_s)
        return self.inner.predict_proba(X)

    def clone(self):
        return DelayedMember(self.inner.clone(), self.delay_s)

    @property
    def name(self) -> str:
        return f"{self.inner.name}-slow"


class EnsembleServer:
    def __init__(
        self,
        ensemble: Ensemble,
        key: str,
        settings: Settings | None = None,
        fallback: Ensemble | None = None,
        fallback_key: str | None = None,
    ):
        self.ensemble = ensemble
        self.key = key
        self.settings = settings or Settings()
        self.fallback = fallback
        self.fallback_key = fallback_key
        self._latencies: list[float] = []
        self._lock = threading.Lock()

    def _predict_member(self, member: BaseLearner, X):
        return member.predict_proba(X)

    def predict_one(
        self,
        features: list[float],
        budget_ms: float | None = None,
        per_member_timeout_ms: float | None = None,
    ) -> ServeResult:
        budget = budget_ms if budget_ms is not None else self.settings.serve_budget_ms
        timeout = (
            per_member_timeout_ms
            if per_member_timeout_ms is not None
            else self.settings.per_member_timeout_ms
        )
        started = time.monotonic()
        X = [features]
        used: list[str] = []
        dropped: list[str] = []
        probs: list[float] = []
        # Explicit executor lifecycle: shutdown(wait=False) so a slow
        # member's thread does not hold the request open after its
        # timeout has already been applied.
        pool = ThreadPoolExecutor(max_workers=len(self.ensemble.members))
        try:
            futures = {
                pool.submit(self._predict_member, m, X): m
                for m in self.ensemble.members
            }
            for fut, member in futures.items():
                remaining = budget / 1000.0 - (time.monotonic() - started)
                if remaining <= 0:
                    dropped.append(member.name)
                    continue
                try:
                    p = fut.result(timeout=min(timeout / 1000.0, remaining))
                    probs.append(p[0])
                    used.append(member.name)
                except FutureTimeout:
                    dropped.append(member.name)
                except Exception:
                    dropped.append(member.name)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        latency_ms = (time.monotonic() - started) * 1000.0
        if not probs:
            # Every member failed or timed out: fall back instead of erroring.
            if self.fallback is not None:
                fb = self.fallback.predict_proba(X)[0]
                self._record_latency(latency_ms)
                return ServeResult(
                    label=1 if fb >= 0.5 else 0,
                    probability=fb,
                    members_used=[],
                    members_dropped=[m.name for m in self.ensemble.members],
                    latency_ms=latency_ms,
                    degraded=True,
                    ensemble_key=self.key,
                    fallback_used=True,
                )
            raise RuntimeError("all ensemble members timed out and no fallback is configured")
        prob = sum(probs) / len(probs)
        self._record_latency(latency_ms)
        return ServeResult(
            label=1 if prob >= 0.5 else 0,
            probability=prob,
            members_used=used,
            members_dropped=dropped,
            latency_ms=latency_ms,
            degraded=bool(dropped),
            ensemble_key=self.key,
        )

    def predict_batch(
        self, rows: list[list[float]], **kwargs
    ) -> list[ServeResult]:
        return [self.predict_one(row, **kwargs) for row in rows]

    def _record_latency(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies.append(latency_ms)

    def latency_stats(self) -> dict[str, float]:
        with self._lock:
            lat = sorted(self._latencies)
        if not lat:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        def _pct(p: float) -> float:
            idx = min(len(lat) - 1, max(0, int(math.ceil(p * len(lat))) - 1))
            return lat[idx]

        return {
            "count": float(len(lat)),
            "p50_ms": _pct(0.5),
            "p95_ms": _pct(0.95),
            "max_ms": lat[-1],
        }
