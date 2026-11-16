from ensemble.audit import AuditLog


def test_audit_chain_verifies():
    log = AuditLog()
    log.append("register", {"key": "stacking-ensemble@1"})
    log.append("promote", {"key": "stacking-ensemble@1"})
    assert log.verify() is True
    assert len(log.entries()) == 2
    assert log.entries()[0].prev_hash == "GENESIS"
    assert log.entries()[1].prev_hash == log.entries()[0].hash


def test_audit_chain_detects_tampering():
    log = AuditLog()
    log.append("register", {"key": "a"})
    log.append("promote", {"key": "a"})
    log.entries()[0].detail["key"] = "tampered"
    assert log.verify() is False


def test_audit_chain_detects_deletion():
    log = AuditLog()
    log.append("register", {"key": "a"})
    log.append("promote", {"key": "a"})
    log.append("rollback", {"key": "a"})
    del log._entries[1]
    assert log.verify() is False
