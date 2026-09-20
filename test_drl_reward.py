"""
test_drl_reward.py

Unit tests for rewards/reward_engine.py. No simulator or Gymnasium needed.

Run either way, from the project root:
    python test_drl_reward.py
    python -m pytest test_drl_reward.py -v
"""

import math
import sys
import traceback
import unittest
from contextlib import contextmanager
from pathlib import Path

from rewards.reward_engine import (
    BENEFIT_TERMS,
    TERMS,
    RewardConfig,
    RewardEngine,
    RewardInputs,
)

REWARD_YAML = Path(__file__).resolve().parent / "config" / "reward_config.yaml"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

@contextmanager
def expect_raises(exc_type, contains=None):
    try:
        yield
    except exc_type as e:
        if contains is not None:
            assert contains in str(e), f"expected '{contains}' in error message, got: {e}"
    else:
        raise AssertionError(f"expected {exc_type.__name__} to be raised")


def only(**weights):
    """Config where every weight is 0 except the ones given (isolates a term)."""
    base = {t: 0.0 for t in TERMS}
    base.update(weights)
    return RewardConfig(weights=base)


def done(latency_ms=500.0, met=True, priority="medium"):
    return {"latency_ms": latency_ms, "met_deadline": met, "priority": priority}


def failed(reason="deadline_missed_while_running", priority="medium"):
    return {"reason": reason, "priority": priority}


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------

def test_no_events_gives_zero_costs():
    b = RewardEngine(RewardConfig()).compute(
        RewardInputs(load_scores=[0.3, 0.3, 0.3], cpu_utilization=[0, 0, 0], cpu_cores=[2, 4, 8])
    )
    for term in ("latency", "sla", "rejection", "balance", "energy", "utilization"):
        assert b.raw[term] == 0.0, term
    assert b.total == 0.0


def test_latency_term_is_normalised_and_capped():
    cfg = only(latency=1.0)
    eng = RewardEngine(cfg)
    # one task at exactly latency_ref -> raw 1.0, reward -1.0
    b = eng.compute(RewardInputs(completed=[done(latency_ms=cfg.latency_ref_ms)]))
    assert math.isclose(b.raw["latency"], 1.0)
    assert math.isclose(b.total, -1.0)
    # two tasks add up
    b = eng.compute(RewardInputs(completed=[done(500.0), done(1500.0)]))
    assert math.isclose(b.raw["latency"], 0.5 + 1.5)
    # a huge latency is clipped at latency_cap, not unbounded
    b = eng.compute(RewardInputs(completed=[done(latency_ms=10_000_000.0)]))
    assert math.isclose(b.raw["latency"], cfg.latency_cap)


def test_sla_counts_late_completions_and_deadline_failures_by_priority():
    cfg = only(sla=1.0)
    eng = RewardEngine(cfg)
    pw = cfg.priority_weights

    # on-time completion -> no SLA cost
    assert eng.compute(RewardInputs(completed=[done(met=True)])).raw["sla"] == 0.0

    # late completion, one per priority
    for prio in ("high", "medium", "low"):
        b = eng.compute(RewardInputs(completed=[done(met=False, priority=prio)]))
        assert math.isclose(b.raw["sla"], pw[prio]), prio
        assert b.counts["n_late_completions"] == 1

    # killed for missing deadline while running -> also SLA
    b = eng.compute(RewardInputs(failed=[failed("deadline_missed_while_running", "high")]))
    assert math.isclose(b.raw["sla"], pw["high"])
    assert b.counts["n_deadline_failures"] == 1
    assert b.raw["rejection"] == 0.0

    # high priority costs more than low priority
    hi = eng.compute(RewardInputs(completed=[done(met=False, priority="high")])).raw["sla"]
    lo = eng.compute(RewardInputs(completed=[done(met=False, priority="low")])).raw["sla"]
    assert hi > lo


def test_rejection_is_separate_from_sla():
    eng = RewardEngine(RewardConfig())
    for reason in ("server_full_or_incompatible", "queue_full_on_arrival", "invalid_server_id"):
        b = eng.compute(RewardInputs(failed=[failed(reason, "medium")]))
        assert b.raw["rejection"] == RewardConfig().priority_weights["medium"], reason
        assert b.raw["sla"] == 0.0, reason
        assert b.counts["n_rejected"] == 1
    # Ablating SLA must NOT make dropping tasks free.
    ablated = RewardEngine(RewardConfig().without("sla"))
    b = ablated.compute(RewardInputs(failed=[failed("server_full_or_incompatible")]))
    assert b.total < 0.0


def test_balance_is_zero_when_even_and_bounded():
    eng = RewardEngine(only(balance=1.0))
    assert eng.compute(RewardInputs(load_scores=[0.4, 0.4, 0.4, 0.4])).raw["balance"] == 0.0
    assert eng.compute(RewardInputs(load_scores=[0.9])).raw["balance"] == 0.0  # single server
    uneven = eng.compute(RewardInputs(load_scores=[0.0, 0.0, 1.0, 1.0])).raw["balance"]
    mild = eng.compute(RewardInputs(load_scores=[0.4, 0.5, 0.5, 0.6])).raw["balance"]
    assert 0.0 < mild < uneven <= 1.0
    # extreme overflow values still cannot exceed 1
    assert eng.compute(RewardInputs(load_scores=[0.0, 9.0])).raw["balance"] == 1.0


def test_energy_scales_linearly_and_rejects_negative():
    cfg = only(energy=1.0)
    eng = RewardEngine(cfg)
    a = eng.compute(RewardInputs(energy_delta_j=cfg.energy_ref_j)).raw["energy"]
    b = eng.compute(RewardInputs(energy_delta_j=3 * cfg.energy_ref_j)).raw["energy"]
    assert math.isclose(a, 1.0) and math.isclose(b, 3.0)
    with expect_raises(ValueError, "energy_delta_j"):
        eng.compute(RewardInputs(energy_delta_j=-1.0))


def test_utilization_is_capacity_weighted_and_a_benefit():
    assert "utilization" in BENEFIT_TERMS
    eng = RewardEngine(only(utilization=1.0))
    # 2-core server fully busy, 8-core idle -> 2/10 = 0.2 (NOT the plain mean 0.5)
    b = eng.compute(RewardInputs(cpu_utilization=[1.0, 0.0], cpu_cores=[2, 8]))
    assert math.isclose(b.raw["utilization"], 0.2)
    assert b.total > 0.0  # benefit is added, not subtracted
    with expect_raises(ValueError):
        eng.compute(RewardInputs(cpu_utilization=[1.0], cpu_cores=[2, 8]))


def test_zero_weight_disables_term_but_raw_is_still_reported():
    inputs = RewardInputs(energy_delta_j=200.0, load_scores=[0.0, 1.0])
    full = RewardEngine(RewardConfig()).compute(inputs)
    ablated = RewardEngine(RewardConfig().without("energy")).compute(inputs)
    assert ablated.raw["energy"] == full.raw["energy"] > 0.0
    assert ablated.weighted["energy"] == 0.0
    assert ablated.total > full.total  # removing a cost raises the reward
    assert math.isclose(ablated.total, full.total - full.weighted["energy"])


def test_total_is_scaled_sum_of_weighted_terms():
    cfg = RewardConfig(reward_scale=0.25)
    inputs = RewardInputs(
        completed=[done(800.0), done(2000.0, met=False, priority="high")],
        failed=[failed("queue_full_on_arrival", "low"), failed()],
        load_scores=[0.1, 0.6, 0.3],
        cpu_utilization=[0.5, 0.2, 0.9],
        cpu_cores=[4, 8, 2],
        energy_delta_j=150.0,
    )
    b = RewardEngine(cfg).compute(inputs)
    assert math.isclose(b.total, cfg.reward_scale * sum(b.weighted.values()))
    for term in TERMS:
        sign = 1.0 if term in BENEFIT_TERMS else -1.0
        assert math.isclose(b.weighted[term], sign * cfg.weights[term] * b.raw[term])


def test_worse_outcomes_always_score_lower():
    eng = RewardEngine(RewardConfig())
    good = eng.compute(RewardInputs(completed=[done(400.0, True)]))
    slow = eng.compute(RewardInputs(completed=[done(1800.0, True)]))
    late = eng.compute(RewardInputs(completed=[done(1800.0, False)]))
    dropped = eng.compute(RewardInputs(failed=[failed("server_full_or_incompatible")]))
    assert good.total > slow.total > late.total
    assert dropped.total < slow.total


def test_default_config_never_makes_failing_cheaper_than_serving():
    """Design invariant: the cheapest failure costs at least as much as the
    most expensive successful service, so the agent can never profit from
    letting tasks fail. If you retune weights, keep this true."""
    cfg = RewardConfig.from_yaml(REWARD_YAML)
    cheapest_failure = cfg.weights["sla"] * min(cfg.priority_weights.values())
    worst_service = cfg.weights["latency"] * cfg.latency_cap
    assert cheapest_failure >= worst_service, (cheapest_failure, worst_service)
    cheapest_rejection = cfg.weights["rejection"] * min(cfg.priority_weights.values())
    assert cheapest_rejection >= worst_service


def test_config_validation_catches_mistakes():
    with expect_raises(ValueError, "Unknown reward weight"):
        RewardConfig(weights={"latancy": 1.0})  # typo must not be silently ignored
    with expect_raises(ValueError, ">= 0"):
        RewardConfig(weights={"latency": -1.0})
    with expect_raises(ValueError, "Unknown reward config key"):
        RewardConfig.from_dict({"weight": {}})
    with expect_raises(ValueError):
        RewardConfig(latency_ref_ms=0)
    with expect_raises(ValueError, "Unknown term"):
        RewardConfig().with_weights(nonsense=1.0)
    with expect_raises(KeyError, "Unknown task priority"):
        RewardEngine(RewardConfig()).compute(
            RewardInputs(completed=[done(met=False, priority="urgent")])
        )


def test_yaml_loading_and_overrides():
    cfg = RewardConfig.from_yaml(REWARD_YAML)
    assert set(cfg.weights) == set(TERMS)
    assert all(w >= 0 for w in cfg.weights.values())
    # overrides merge into the weights dict instead of replacing it
    cfg2 = RewardConfig.from_yaml(REWARD_YAML, overrides={"weights": {"energy": 0.0}})
    assert cfg2.weights["energy"] == 0.0
    assert cfg2.weights["sla"] == cfg.weights["sla"]
    cfg3 = RewardConfig.from_yaml(REWARD_YAML, overrides={"reward_scale": 0.5})
    assert cfg3.reward_scale == 0.5


def test_engine_is_deterministic():
    inputs = RewardInputs(completed=[done(700.0)], load_scores=[0.2, 0.5], energy_delta_j=42.0)
    eng = RewardEngine(RewardConfig())
    assert eng.compute(inputs).total == eng.compute(inputs).total


# ----------------------------------------------------------------------
# tiny runner so this file also works as `python test_drl_reward.py`
# ----------------------------------------------------------------------

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except unittest.SkipTest as e:
            print(f"SKIP  {name}: {e}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed" + ("" if not failures else f", {failures} FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())