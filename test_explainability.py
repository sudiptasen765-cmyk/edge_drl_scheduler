"""Phase 7 explainability tests. Run: python -m pytest test_explainability.py -q"""
import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.edge_scheduling_env import EdgeSchedulingEnv
from explainability.explain_decision import build_reference, explain_decision, _masked_probs


@pytest.fixture(scope="module")
def setup():
    env = EdgeSchedulingEnv.from_config_files()
    vec = DummyVecEnv([lambda: Monitor(EdgeSchedulingEnv.from_config_files())])
    model = MaskablePPO("MlpPolicy", vec, n_steps=256, batch_size=64, seed=0, device="cpu", verbose=0)
    model.learn(1024)
    ref = build_reference(model, env, n_episodes=2)
    obs, _ = env.reset(options={"workload_type": "normal", "network_scenario": "normal", "task_seed": 7})
    return model, env, ref, obs


def test_probabilities_respect_mask_and_sum_to_one(setup):
    model, env, ref, obs = setup
    mask = np.array([True, False, True, False])
    p = _masked_probs(model, obs[None], mask)[0]
    assert p.sum() == pytest.approx(1.0, abs=1e-5)
    assert p[1] == 0 and p[3] == 0


def test_chosen_action_matches_model_predict(setup):
    model, env, ref, obs = setup
    mask = env.action_masks()
    e = explain_decision(model, obs, mask, env.feature_names, env.server_ids, ref)
    pred, _ = model.predict(obs, action_masks=mask, deterministic=True)
    assert e.chosen_server == env.server_ids[int(pred)]
    assert e.chosen_server in e.eligible_servers
    assert sum(e.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_effects_are_consistent_with_direct_recomputation(setup):
    """Recompute one reported effect by hand: it must match."""
    model, env, ref, obs = setup
    mask = np.ones(4, bool)
    e = explain_decision(model, obs, mask, env.feature_names, env.server_ids, ref, top_k=1)
    f = (e.supporting or e.opposing)[0]
    i = env.feature_names.index(f.name)
    ci, ri = env.server_ids.index(e.chosen_server), env.server_ids.index(e.runner_up_server)
    p0 = _masked_probs(model, obs[None], mask)[0]
    x2 = obs.copy(); x2[i] = ref[i]
    p1 = _masked_probs(model, x2[None], mask)[0]
    m0 = np.log(p0[ci] + 1e-12) - np.log(p0[ri] + 1e-12)
    m1 = np.log(p1[ci] + 1e-12) - np.log(p1[ri] + 1e-12)
    assert f.effect == pytest.approx(m0 - m1, abs=1e-4)


def test_no_effect_when_state_equals_reference(setup):
    model, env, ref, obs = setup
    e = explain_decision(model, ref.copy(), np.ones(4, bool), env.feature_names, env.server_ids, ref)
    assert not e.supporting and not e.opposing
    assert all(abs(v) < 1e-5 for v in e.group_effects.values())


def test_single_eligible_server_is_reported_honestly(setup):
    model, env, ref, obs = setup
    mask = np.array([False, False, True, False])
    e = explain_decision(model, obs, mask, env.feature_names, env.server_ids, ref)
    assert e.chosen_server == env.server_ids[2] and e.runner_up_server is None
    assert "only eligible" in e.text and e.supporting == []


def test_groups_cover_task_and_servers_and_text_is_present(setup):
    model, env, ref, obs = setup
    e = explain_decision(model, obs, env.action_masks(), env.feature_names, env.server_ids, ref)
    assert set(e.group_effects) <= {"task", "server1", "server2", "server3", "server4"}
    assert "Chose server" in e.text and "not a causal explanation" in e.text
    assert e.to_dict()["chosen_server"] == e.chosen_server   # JSON-ready for the dashboard


def test_input_validation(setup):
    model, env, ref, obs = setup
    with pytest.raises(ValueError):
        explain_decision(model, obs, env.action_masks(), env.feature_names[:-1], env.server_ids, ref)


def test_flip_features_really_change_the_choice(setup):
    model, env, ref, obs = setup
    mask = env.action_masks()
    e = explain_decision(model, obs, mask, env.feature_names, env.server_ids, ref, top_k=50)
    for f in e.flip_features:
        i = env.feature_names.index(f["name"])
        x2 = obs.copy(); x2[i] = ref[i]
        p = _masked_probs(model, x2[None], mask)[0]
        assert env.server_ids[int(np.argmax(p))] == f["becomes_server"] != e.chosen_server
    # any feature NOT reported as flipping must leave the argmax unchanged
    flipped = {f["name"] for f in e.flip_features}
    for i, name in enumerate(env.feature_names):
        if name in flipped:
            continue
        x2 = obs.copy(); x2[i] = ref[i]
        p = _masked_probs(model, x2[None], mask)[0]
        assert env.server_ids[int(np.argmax(p))] == e.chosen_server


def test_text_states_robustness_when_nothing_flips(setup):
    model, env, ref, obs = setup
    e = explain_decision(model, ref.copy(), np.ones(4, bool), env.feature_names, env.server_ids, ref)
    assert e.flip_features == [] and "No single factor group" in e.text


def test_families_partition_the_features_and_group_across_servers(setup):
    from explainability.explain_decision import feature_families
    model, env, ref, obs = setup
    fams = feature_families(env.feature_names)
    idx = sorted(i for v in fams.values() for i in v)
    assert idx == list(range(len(env.feature_names)))               # every feature exactly once
    assert len(fams["queue_fill (all servers)"]) == len(env.server_ids)
    assert len(fams["task priority"]) == 3


def test_family_effects_match_direct_recomputation_and_flips(setup):
    from explainability.explain_decision import feature_families
    model, env, ref, obs = setup
    mask = np.ones(4, bool)
    e = explain_decision(model, obs, mask, env.feature_names, env.server_ids, ref)
    fams = feature_families(env.feature_names)
    p0 = _masked_probs(model, obs[None], mask)[0]
    ci, ri = env.server_ids.index(e.chosen_server), env.server_ids.index(e.runner_up_server)
    m0 = np.log(p0[ci] + 1e-12) - np.log(p0[ri] + 1e-12)
    for f in e.family_effects:
        x2 = obs.copy(); x2[fams[f["family"]]] = ref[fams[f["family"]]]
        p1 = _masked_probs(model, x2[None], mask)[0]
        m1 = np.log(p1[ci] + 1e-12) - np.log(p1[ri] + 1e-12)
        assert f["effect"] == pytest.approx(m0 - m1, abs=1e-4)
        assert (f["becomes_server"] is not None) == (env.server_ids[int(np.argmax(p1))] != e.chosen_server)