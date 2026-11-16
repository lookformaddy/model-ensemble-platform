"""SHA-256 hash-chained audit log.

Every registry decision (register, candidate, promote, block, rollback)
and every drift-triggered fallback is appended with the hash of the
previous entry, so any tampering or deletion is detectable with
``verify()``. The chain rule is persistence-agnostic: the same entries can
be stored in a file or a database without changing the verification.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AuditEntry:
    seq: int
    timestamp: str
    event: str
    detail: dict
    prev_hash: str
    hash: str = ""


class AuditLog:
    def __init__(self):
        self._entries: list[AuditEntry] = []

    def append(self, event: str, detail: dict | None = None) -> AuditEntry:
        prev_hash = self._entries[-1].hash if self._entries else "GENESIS"
        entry = AuditEntry(
            seq=len(self._entries),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event=event,
            detail=detail or {},
            prev_hash=prev_hash,
        )
        payload = json.dumps(
            {
                "seq": entry.seq,
                "timestamp": entry.timestamp,
                "event": entry.event,
                "detail": entry.detail,
                "prev_hash": entry.prev_hash,
            },
            sort_keys=True,
        )
        entry.hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._entries.append(entry)
        return entry

    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def verify(self) -> bool:
        prev = "GENESIS"
        for entry in self._entries:
            if entry.prev_hash != prev:
                return False
            payload = json.dumps(
                {
                    "seq": entry.seq,
                    "timestamp": entry.timestamp,
                    "event": entry.event,
                    "detail": entry.detail,
                    "prev_hash": entry.prev_hash,
                },
                sort_keys=True,
            )
            if hashlib.sha256(payload.encode("utf-8")).hexdigest() != entry.hash:
                return False
            prev = entry.hash
        return True
