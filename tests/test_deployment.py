from ensemble.deployment import DeploymentManager, hash_bucket


def test_hash_bucket_is_deterministic():
    assert hash_bucket("req-123") == hash_bucket("req-123")
    assert 0.0 <= hash_bucket("req-123") < 1.0
    assert hash_bucket("req-123") != hash_bucket("req-124")


def test_route_without_candidate_is_always_production():
    deploy = DeploymentManager(production_key="voting-ensemble@1")
    assert deploy.route("req-1") == "voting-ensemble@1"
    assert deploy.is_canary("req-1") is False


def test_canary_fraction_is_approximately_honoured():
    deploy = DeploymentManager(
        production_key="voting-ensemble@1",
        candidate_key="stacking-ensemble@1",
        canary_fraction=0.10,
    )
    ids = [f"req-{i}" for i in range(1000)]
    canary = sum(1 for rid in ids if deploy.is_canary(rid))
    # Deterministic hashing: expect ~10% with binomial wiggle room.
    assert 50 <= canary <= 160
    # Same id always routes the same way.
    assert all(deploy.is_canary(rid) == deploy.is_canary(rid) for rid in ids[:50])


def test_shadow_agreement_scoring():
    deploy = DeploymentManager(
        production_key="voting-ensemble@1",
        candidate_key="stacking-ensemble@1",
        shadow_agreement_floor=0.85,
    )
    for i in range(10):
        deploy.record_shadow(f"req-{i}", 1, 1, 0.8, 0.75)
    deploy.record_shadow("req-10", 1, 0, 0.8, 0.4)
    assert deploy.shadow_agreement() == 10 / 11
    report = deploy.shadow_report()
    assert report["n_shadowed"] == 11
    assert report["safe_to_promote"] is True


def test_shadow_below_floor_is_not_safe():
    deploy = DeploymentManager(
        production_key="voting-ensemble@1",
        candidate_key="stacking-ensemble@1",
        shadow_agreement_floor=0.85,
    )
    for i in range(10):
        deploy.record_shadow(f"req-{i}", 1, 0, 0.8, 0.4)
    assert deploy.shadow_agreement() == 0.0
    assert deploy.shadow_report()["safe_to_promote"] is False


def test_promote_candidate_swaps_production():
    deploy = DeploymentManager(
        production_key="voting-ensemble@1",
        candidate_key="stacking-ensemble@1",
    )
    assert deploy.promote_candidate() == "stacking-ensemble@1"
    assert deploy.production_key == "stacking-ensemble@1"
    assert deploy.candidate_key is None
