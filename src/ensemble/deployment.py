"""Canary and shadow deployment for ensemble versions.

Routing is deterministic: the SHA-256 hash of the request id decides the
bucket, so the same request id always lands on the same arm — no sticky
sessions, no surprises in replay.

- **Canary** — a fraction of traffic (``canary_fraction``) is routed to the
  candidate; the rest stays on production.
- **Shadow** — the candidate scores every (or a sampled) request in the
  background but its answer is never served; agreement between production
  and candidate is recorded, and a candidate whose shadow agreement falls
  below the floor is not safe to promote.
"""
from __future__ import annotations

import hashlib

from .models import ShadowRecord


def hash_bucket(request_id: str) -> float:
    """Deterministic bucket in [0, 1) from a request id."""
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class DeploymentManager:
    def __init__(
        self,
        production_key: str,
        candidate_key: str | None = None,
        canary_fraction: float = 0.10,
        shadow_agreement_floor: float = 0.85,
    ):
        self.production_key = production_key
        self.candidate_key = candidate_key
        self.canary_fraction = canary_fraction
        self.shadow_agreement_floor = shadow_agreement_floor
        self._shadow: list[ShadowRecord] = []

    def route(self, request_id: str) -> str:
        """Return the ensemble key that should serve this request."""
        if self.candidate_key is None:
            return self.production_key
        if hash_bucket(request_id) < self.canary_fraction:
            return self.candidate_key
        return self.production_key

    def is_canary(self, request_id: str) -> bool:
        return self.route(request_id) == self.candidate_key

    def record_shadow(
        self, request_id: str, production_label: int, candidate_label: int,
        production_prob: float, candidate_prob: float,
    ) -> ShadowRecord:
        record = ShadowRecord(
            request_id=request_id,
            production_label=production_label,
            candidate_label=candidate_label,
            agree=production_label == candidate_label,
            production_prob=production_prob,
            candidate_prob=candidate_prob,
        )
        self._shadow.append(record)
        return record

    def shadow_agreement(self) -> float:
        if not self._shadow:
            return 1.0
        return sum(1 for r in self._shadow if r.agree) / len(self._shadow)

    def shadow_report(self) -> dict:
        agreement = self.shadow_agreement()
        return {
            "n_shadowed": len(self._shadow),
            "agreement": agreement,
            "floor": self.shadow_agreement_floor,
            "safe_to_promote": agreement >= self.shadow_agreement_floor,
        }

    def promote_candidate(self) -> str:
        """Swap the candidate into production (after the gates pass)."""
        if self.candidate_key is None:
            raise ValueError("no candidate deployed")
        self.production_key = self.candidate_key
        self.candidate_key = None
        return self.production_key
