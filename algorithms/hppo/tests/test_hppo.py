"""Regression tests for the hPPO implementation.

Each test pins down a property that is either non-obvious from the source or
easy to break silently. The set mirrors algorithms/ppo/tests/test_ppo.py for everything
hPPO inherited from flat PPO -- the GAE indexing convention, the value
standardisation round trip, the self-describing checkpoint, the metric
definitions -- and adds the ones specific to the hierarchy: the manager's
derived gamma**c discount, the goal encoding, and the fact that the two heads
are updated independently.

See HPPOAgent in algorithms/hppo/hppo.py, and the thesis write-up's PPO
chapter (kept outside this repo), for the reasoning ported
from there.
"""

import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

from hppo import (
    HPPOAgent,
    RolloutBuffer,
    VecRolloutBuffer,
    WorkerActor,
    normalize_goal,
)

OBS_DIM, GOAL_DIM, ACT_DIM = 4, 2, 2
OBS_LOW = [-1.0, -2.0, -2.0, -2.0]
OBS_HIGH = [11.0, 2.0, 2.0, 2.0]
LOW, HIGH = torch.tensor(-1.0), torch.tensor(1.0)


def _agent(**kwargs):
    kwargs.setdefault("obs_low", OBS_LOW)
    kwargs.setdefault("obs_high", OBS_HIGH)
    return HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu", **kwargs)


def _filled_buffer(num_steps=16, obs_dim=OBS_DIM, act_dim=GOAL_DIM, seed=0, fill=None):
    """A buffer with plausible, reproducible contents, filled to `fill` steps.

    Partial filling is the norm here, not an edge case: the manager writes one
    transition per c worker steps into a buffer sized for the worker.
    """
    torch.manual_seed(seed)
    buf = RolloutBuffer(num_steps, obs_dim, act_dim, "cpu")
    buf.states = torch.randn(num_steps, obs_dim)
    buf.actions = torch.rand(num_steps, act_dim) * 2 - 1
    buf.logprobs = torch.randn(num_steps)
    buf.rewards = torch.randn(num_steps) * 100
    buf.values = torch.randn(num_steps) * 100
    buf.dones = (torch.rand(num_steps) < 0.3).float()
    buf.step = num_steps if fill is None else fill
    return buf


# --------------------------------------------------------------------------
# Input encoding
# --------------------------------------------------------------------------

def test_agent_without_bounds_refuses_to_normalize():
    """Silently returning raw physical units is the failure this prevents."""
    agent = HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu")
    with pytest.raises(ValueError, match="observation bounds"):
        agent.normalize_obs(np.zeros(4, dtype=np.float32))


def test_goal_encoding_round_trips_through_the_manager_action_box():
    """scale_goal is the rollout's manager-action -> metres map; normalize_goal
    is what the worker is fed. They must be exact inverses, or the worker sees
    a goal in different units from the one the geometry decays."""
    agent = _agent(max_goal_bound=10.0)
    manager_action = np.array([1.0, -0.35], dtype=np.float32)
    physical = agent.scale_goal(manager_action)
    assert physical == pytest.approx([10.0, -3.5])
    assert agent.normalize_goal(physical) == pytest.approx(manager_action)


def test_a_fresh_goal_lands_on_the_worker_input_box():
    """The manager's action box is [-1, 1]^2, so a freshly issued goal occupies
    the same range as the normalized observation it is concatenated onto."""
    assert normalize_goal(np.array([10.0, -10.0]), 10.0) == pytest.approx([1.0, -1.0])


@pytest.mark.parametrize("head", ["manager", "worker"])
def test_act_is_the_deterministic_action_without_a_critic_pass(head):
    """Flat PPO's `act`, per head: evaluation and the solved-check only need
    the action, deterministically, and never the value."""
    agent = _agent()
    in_dim = OBS_DIM if head == "manager" else OBS_DIM + GOAL_DIM
    state = torch.randn(3, in_dim)
    expected = getattr(agent, f"{head}_policy_forward")(state, deterministic=True)[0]

    def no_critic(*_):
        raise AssertionError(f"{head}_act() must not evaluate the critic")

    getattr(agent, f"{head}_critic").forward = no_critic
    assert torch.equal(getattr(agent, f"{head}_act")(state), expected)


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------

def test_gae_matches_transition_indexed_reference():
    """dones[t] means 'transition t ended an episode', so the mask is 1-dones[t].

    CleanRL stores the other convention and masks with dones[t+1]; this test
    pins ours down so the difference is never 'fixed' by mistake.
    """
    buf = _filled_buffer()
    next_value = torch.tensor(3.0)
    gamma, lam = 0.99, 0.95
    buf.compute_returns_and_advantage(next_value, gamma, lam)

    expected = torch.zeros(buf.step)
    last = 0.0
    for t in reversed(range(buf.step)):
        nonterminal = 1.0 - buf.dones[t]
        next_v = next_value if t == buf.step - 1 else buf.values[t + 1]
        delta = buf.rewards[t] + gamma * next_v * nonterminal - buf.values[t]
        last = delta + gamma * lam * nonterminal * last
        expected[t] = last
    assert torch.allclose(buf.advantages[:buf.step], expected, atol=1e-4)


def test_gae_returns_identity_holds():
    """returns = advantages + values, which the update relies on."""
    buf = _filled_buffer()
    buf.compute_returns_and_advantage(torch.tensor(0.0), 0.99, 0.95)
    assert torch.allclose(buf.returns, buf.advantages + buf.values, atol=1e-4)


def test_gae_stops_bootstrapping_at_a_done():
    """A terminal transition must not leak value from the next episode."""
    buf = RolloutBuffer(2, OBS_DIM, GOAL_DIM, "cpu")
    buf.rewards = torch.tensor([5.0, 7.0])
    buf.values = torch.zeros(2)
    buf.dones = torch.tensor([1.0, 0.0])  # transition 0 ended the episode
    buf.step = 2
    buf.compute_returns_and_advantage(torch.tensor(100.0), 0.99, 0.95)
    # Step 0 is terminal: its advantage is its own reward and nothing else.
    assert buf.advantages[0].item() == pytest.approx(5.0, abs=1e-4)


def test_gae_does_not_bootstrap_past_a_done_on_the_last_transition():
    """The case a separate `next_done` argument used to cover: when the last
    stored transition ended its episode, the state after the rollout belongs
    to the next episode, and its value must not leak in. dones[-1] alone has
    to say so."""
    buf = RolloutBuffer(2, OBS_DIM, GOAL_DIM, "cpu")
    buf.rewards = torch.tensor([5.0, 7.0])
    buf.values = torch.zeros(2)
    buf.dones = torch.tensor([0.0, 1.0])  # the last transition ended the episode
    buf.step = 2
    buf.compute_returns_and_advantage(torch.tensor(100.0), 0.99, 0.95)
    assert buf.advantages[1].item() == pytest.approx(7.0, abs=1e-4)


def test_gae_only_touches_the_filled_prefix():
    """The manager's buffer is sized for the worker and left mostly empty."""
    buf = _filled_buffer(num_steps=16, fill=5)
    buf.compute_returns_and_advantage(torch.tensor(0.0), 0.99, 0.95)
    assert torch.all(buf.advantages[5:] == 0.0)
    assert torch.any(buf.advantages[:5] != 0.0)


def test_gae_parameters_are_required():
    """They used to default to 0.99/0.95 while the manager ran at gamma**c."""
    buf = _filled_buffer()
    with pytest.raises(TypeError):
        buf.compute_returns_and_advantage(torch.tensor(0.0))


# --------------------------------------------------------------------------
# Worker actor: the plain softplus + 1 floor (no cap -- see ManagerActor's
# MAX_CONCENTRATION docstring for why the manager alone needed one).
# Architecturally identical to flat PPO's ActorNetwork (algorithms/ppo/tests/
# test_ppo.py's test_actor_concentrations_never_drop_below_one/
# test_actor_initial_policy_is_near_uniform), ported here because hppo.py's
# WorkerActor is its own class, not a reuse of ActorNetwork.
# --------------------------------------------------------------------------

def test_worker_actor_concentrations_never_drop_below_one():
    """softplus + 1 keeps the density bounded and unimodal for any logits."""
    actor = WorkerActor(OBS_DIM, GOAL_DIM, ACT_DIM)
    with torch.no_grad():
        for layer in actor.net:
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(-50.0)  # drive softplus to ~0
    alpha, beta = actor(torch.randn(64, OBS_DIM + GOAL_DIM))
    assert alpha.min().item() >= 1.0 and beta.min().item() >= 1.0


def test_worker_actor_initial_policy_is_near_uniform():
    """std=0.01 output init gives alpha = beta = softplus(0) + 1, i.e. broad."""
    torch.manual_seed(0)
    alpha, beta = WorkerActor(OBS_DIM, GOAL_DIM, ACT_DIM)(torch.zeros(1, OBS_DIM + GOAL_DIM))
    expected = math.log(2.0) + 1.0
    assert alpha.flatten().tolist() == pytest.approx([expected] * ACT_DIM, abs=1e-4)
    assert beta.flatten().tolist() == pytest.approx([expected] * ACT_DIM, abs=1e-4)


# --------------------------------------------------------------------------
# The two discounts
# --------------------------------------------------------------------------

def test_manager_discount_is_derived_from_the_cadence():
    """One manager step is c environment steps. This used to be computed in the
    training script while the agent stored gamma_manager=gamma, so the agent's
    own attribute disagreed with the discounting actually used."""
    agent = _agent(gamma=0.99, manager_freq=10)
    assert agent.gamma_worker == pytest.approx(0.99)
    assert agent.gamma_manager == pytest.approx(0.99 ** 10)


def test_agent_is_the_single_source_of_each_head_discount():
    agent = _agent(gamma=0.9, manager_freq=3, gae_lambda=0.5)
    buf_m, buf_w, reference = _filled_buffer(), _filled_buffer(), _filled_buffer()
    next_value = torch.tensor(1.5)

    agent.compute_manager_returns_and_advantage(buf_m, next_value)
    reference.compute_returns_and_advantage(next_value, 0.9 ** 3, 0.5)
    assert torch.allclose(buf_m.advantages, reference.advantages, atol=1e-5)

    agent.compute_worker_returns_and_advantage(buf_w, next_value)
    reference.compute_returns_and_advantage(next_value, 0.9, 0.5)
    assert torch.allclose(buf_w.advantages, reference.advantages, atol=1e-5)

    # And the two are genuinely different discounts, not the same one twice.
    assert not torch.allclose(buf_m.advantages, buf_w.advantages, atol=1e-5)


def test_agent_default_discount_is_the_documented_one():
    """The default must track hppo_train.py's --gamma (PPO ch. 11.1)."""
    agent = HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu")
    assert agent.gamma == pytest.approx(0.99)


# --------------------------------------------------------------------------
# Value-target standardisation
# --------------------------------------------------------------------------

def test_get_value_inverts_the_standardisation_for_both_heads():
    """Each critic learns in standardised space; get_*_value is the way back."""
    agent = _agent()
    agent.manager_ret_rms.mean, agent.manager_ret_rms.var = 350.0, 53570.0
    agent.worker_ret_rms.mean, agent.worker_ret_rms.var = 0.4, 0.09

    obs = torch.randn(8, OBS_DIM)
    raw = agent.manager_critic(obs).squeeze(-1)
    expected = raw * agent.manager_ret_rms.std + agent.manager_ret_rms.mean
    assert torch.allclose(agent.get_manager_value(obs), expected, atol=1e-4)

    obs_goal = torch.randn(8, OBS_DIM + GOAL_DIM)
    raw = agent.worker_critic(obs_goal).squeeze(-1)
    expected = raw * agent.worker_ret_rms.std + agent.worker_ret_rms.mean
    assert torch.allclose(agent.get_worker_value(obs_goal), expected, atol=1e-4)


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

def test_checkpoint_is_self_describing():
    """Both ret_rms, the observation bounds, the goal bound and the cadence must
    survive a save/load round trip, or a reloaded hierarchy's inputs, outputs
    and re-planning schedule are all wrong."""
    agent = _agent(max_goal_bound=10.0, manager_freq=7, gamma=0.99)
    agent.manager_ret_rms.mean, agent.manager_ret_rms.var = 350.0, 53570.0
    agent.worker_ret_rms.mean, agent.worker_ret_rms.var = 0.4, 0.09
    obs = torch.randn(8, OBS_DIM)
    obs_goal = torch.randn(8, OBS_DIM + GOAL_DIM)
    manager_before = agent.get_manager_value(obs)
    worker_before = agent.get_worker_value(obs_goal)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        reloaded = HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu")
        reloaded.load(path)  # must not need weights_only=False

    assert reloaded.obs_low == pytest.approx(OBS_LOW)
    assert reloaded.obs_high == pytest.approx(OBS_HIGH)
    assert reloaded.max_goal_bound == pytest.approx(10.0)
    assert reloaded.manager_freq == 7
    assert reloaded.gamma_manager == pytest.approx(0.99 ** 7)
    assert torch.allclose(reloaded.get_manager_value(obs), manager_before, atol=1e-4)
    assert torch.allclose(reloaded.get_worker_value(obs_goal), worker_before, atol=1e-4)
    assert reloaded.normalize_obs(np.array(OBS_HIGH, dtype=np.float32)) == pytest.approx([1.0] * 4)


def test_reloaded_policies_act_identically():
    agent = _agent()
    obs = torch.randn(4, OBS_DIM)
    obs_goal = torch.randn(4, OBS_DIM + GOAL_DIM)
    manager_before = agent.manager_act(obs)
    worker_before = agent.worker_act(obs_goal)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        reloaded = HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu")
        reloaded.load(path)

    assert torch.allclose(reloaded.manager_act(obs), manager_before, atol=1e-6)
    assert torch.allclose(reloaded.worker_act(obs_goal), worker_before, atol=1e-6)


def test_checkpoint_from_the_autotuning_era_still_loads():
    """Every checkpoint trained before the entropy autotuner was removed
    carries both heads' log_ent_coef and their optimiser states. Loading one
    must restore the networks and leave the fixed ent_coefs alone, not raise."""
    agent = _agent()
    obs = torch.randn(4, OBS_DIM)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pt")
        agent.save(path)
        checkpoint = torch.load(path, weights_only=False)
        for head in ("manager", "worker"):
            log_ent_coef = torch.tensor(math.log(0.3), requires_grad=True)
            checkpoint[f"{head}_log_ent_coef"] = math.log(0.3)
            checkpoint[f"{head}_ent_coef_optimizer"] = torch.optim.Adam([log_ent_coef], lr=3e-4).state_dict()
        torch.save(checkpoint, path)

        reloaded = HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu")
        reloaded.load(path)  # must not raise, and must not need weights_only=False

    assert reloaded.ent_coef_manager == 0.01 and reloaded.ent_coef_worker == 0.01
    assert torch.allclose(reloaded.manager_act(obs),
                          agent.manager_act(obs), atol=1e-6)


# --------------------------------------------------------------------------
# What the worker observes (--worker-obs)
# --------------------------------------------------------------------------

def test_default_worker_view_is_the_observation_itself():
    """The default worker's input must be bit-identical to what it was before
    the choice existed: the view is the array itself, not a copy."""
    agent = _agent()
    obs_norm = np.random.default_rng(0).uniform(-1, 1, size=(8, OBS_DIM)).astype(np.float32)
    assert agent.worker_obs == "full"
    assert agent.worker_obs_view(obs_norm) is obs_norm
    assert agent.worker_input_dim == OBS_DIM + GOAL_DIM


def test_velocity_worker_sees_only_the_velocity():
    agent = _agent(worker_obs="velocity")
    obs_norm = np.random.default_rng(0).uniform(-1, 1, size=(8, OBS_DIM)).astype(np.float32)
    assert np.array_equal(agent.worker_obs_view(obs_norm), obs_norm[:, 2:4])
    assert agent.worker_input_dim == 2 + GOAL_DIM
    assert agent.worker_actor.net[0].in_features == 2 + GOAL_DIM
    assert agent.worker_critic.net[0].in_features == 2 + GOAL_DIM
    # The manager still sees everything.
    assert agent.manager_actor.net[0].in_features == OBS_DIM


def test_velocity_worker_acts_the_same_wherever_it_is():
    """The point of the option: moving the agent without changing its
    velocity or its goal cannot change what the worker does."""
    from hppo_train import worker_input
    agent = _agent(worker_obs="velocity")
    obs = np.array([[1.0, -1.0, 0.8, 0.3]] * 2, dtype=np.float32)
    obs[1, :2] = [9.0, 1.5]
    goal = np.array([[4.0, -2.0]] * 2, dtype=np.float32)
    actions = agent.worker_act(worker_input(agent, agent.normalize_obs(obs), goal))
    assert torch.equal(actions[0], actions[1])


def test_unknown_worker_obs_is_rejected():
    with pytest.raises(ValueError, match="worker_obs"):
        _agent(worker_obs="position")


def test_worker_obs_travels_with_the_checkpoint():
    """A reloaded blind worker must be rebuilt blind: the choice sizes its
    networks, so it is read from the checkpoint before they are built, and an
    agent built the other way refuses the checkpoint instead of failing on a
    shape mismatch."""
    from hppo import checkpoint_worker_obs
    agent = _agent(worker_obs="velocity")
    obs_goal = torch.randn(4, 2 + GOAL_DIM)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        assert checkpoint_worker_obs(path) == "velocity"
        reloaded = HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu", worker_obs=checkpoint_worker_obs(path))
        reloaded.load(path)
        with pytest.raises(ValueError, match="worker_obs"):
            HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu").load(path)
    assert torch.allclose(reloaded.worker_act(obs_goal), agent.worker_act(obs_goal), atol=1e-6)


def test_checkpoint_from_before_worker_obs_loads_as_full():
    from hppo import checkpoint_worker_obs
    agent = _agent()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pt")
        agent.save(path)
        checkpoint = torch.load(path, weights_only=False)
        del checkpoint["worker_obs"]
        torch.save(checkpoint, path)
        assert checkpoint_worker_obs(path) == "full"
        HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu").load(path)  # must not raise


# --------------------------------------------------------------------------
# The update
# --------------------------------------------------------------------------

def _prepared(head, agent=None, fill=None):
    """An agent plus a buffer whose logprobs come from the current policy, so
    the first epoch's ratio is exactly 1."""
    agent = agent or _agent()
    if head == "manager":
        buf = _filled_buffer(num_steps=32, obs_dim=OBS_DIM, act_dim=GOAL_DIM, fill=fill)
        forward = agent.manager_policy_forward
        compute = agent.compute_manager_returns_and_advantage
    else:
        buf = _filled_buffer(num_steps=32, obs_dim=OBS_DIM + GOAL_DIM, act_dim=ACT_DIM, fill=fill)
        forward = agent.worker_policy_forward
        compute = agent.compute_worker_returns_and_advantage
    # An empty buffer has nothing to seed logprobs from, and torch's `Beta`
    # rejects a batch of size 0 outright (a distribution-validation quirk
    # unrelated to the code under test here).
    if buf.step > 0:
        with torch.no_grad():
            _, logprobs, _ = forward(buf.states[:buf.step], buf.actions[:buf.step])
        buf.logprobs[:buf.step] = logprobs
    compute(buf, torch.tensor(0.0))
    return agent, buf


@pytest.mark.parametrize("head", ["manager", "worker"])
def test_explained_variance_uses_pre_update_values(head):
    """Scoring a critic after its own 10 epochs on the batch measures
    training-set fit, not prediction. The metric must use the values that
    actually produced the advantages -- the previous implementation re-ran the
    critic after the update."""
    agent, buf = _prepared(head)
    returns = buf.returns[:buf.step].numpy()
    values = buf.values[:buf.step].numpy()
    expected = 1 - np.var(returns - values) / np.var(returns)

    update = agent.update_manager if head == "manager" else agent.update_worker
    metrics = update(buf, 4, 10)
    assert metrics[f"{head}/explained_variance"] == pytest.approx(expected, abs=1e-4)
    assert metrics[f"{head}/value_bias"] == pytest.approx(float(np.mean(values - returns)), abs=1e-2)


@pytest.mark.parametrize("head", ["manager", "worker"])
def test_update_only_consumes_the_filled_prefix(head):
    agent, buf = _prepared(head, fill=9)
    update = agent.update_manager if head == "manager" else agent.update_worker
    assert update(buf, 4, 2)[f"{head}/batch_size"] == 9


@pytest.mark.parametrize("fill, num_minibatches, expected_sizes", [
    (9, 4, [3, 2, 2, 2]),  # the old fixed size, 9 // 4 = 2, made 2, 2, 2, 2 and a runt of 1
    (32, 4, [8, 8, 8, 8]),  # an even batch: the partition fixed-size slicing gave
    (2, 4, [1, 1]),         # fewer samples than minibatches: no empty minibatch
])
def test_update_splits_the_batch_into_num_minibatches_near_equal_parts(
        fill, num_minibatches, expected_sizes):
    """The manager's batch size is whatever the rollout produced, so a fixed
    minibatch size left a runt of 1-3 samples on most updates -- each one
    still a full Adam step. The update takes a count instead."""
    agent, buf = _prepared("manager", fill=fill)
    sizes = []
    forward = agent.manager_policy_forward

    def recording_forward(states, actions):
        sizes.append(states.shape[0])
        return forward(states, actions)

    agent.manager_policy_forward = recording_forward
    agent.update_manager(buf, num_minibatches, update_epochs=3)
    assert sizes == expected_sizes * 3


@pytest.mark.parametrize("head", ["manager", "worker"])
def test_update_with_empty_batch_does_not_crash(head):
    """A head's buffer can end a rollout with zero transitions: a rollout
    short relative to manager_freq, or a run of environments that neither hit
    a c-step boundary nor terminated. The update must degrade to a metrics-only
    no-op rather than crash -- it once did, via the since-removed target_kl
    check indexing an empty list. (The same empty batch also used to poison the since-removed entropy
    autotuner's log_ent_coef with a NaN.)"""
    agent, buf = _prepared(head, fill=0)
    update = agent.update_manager if head == "manager" else agent.update_worker
    metrics = update(buf, 4, 2)
    assert metrics[f"{head}/batch_size"] == 0
    assert math.isnan(metrics[f"{head}/loss_policy"])


@pytest.mark.parametrize("head", ["manager", "worker"])
def test_empty_batch_metrics_have_the_same_keys_as_a_real_update(head):
    """Regression guard: it is easy to add a new metric to the non-empty path
    (as adv_std_raw/approx_kl_max/ratio_max_dev were) and forget the empty-
    batch short-circuit above it, leaving wandb.log fed inconsistent columns
    across updates."""
    agent, buf = _prepared(head)
    empty_agent, empty_buf = _prepared(head, agent=_agent(), fill=0)
    update = agent.update_manager if head == "manager" else agent.update_worker
    empty_update = empty_agent.update_manager if head == "manager" else empty_agent.update_worker
    assert set(update(buf, 4, 2)) == set(empty_update(empty_buf, 4, 2))


@pytest.mark.parametrize("head", ["manager", "worker"])
def test_update_reports_the_expected_metrics(head):
    agent, buf = _prepared(head)
    update = agent.update_manager if head == "manager" else agent.update_worker
    metrics = update(buf, 4, 2)
    assert set(metrics) == {
        f"{head}/loss_policy", f"{head}/loss_value", f"{head}/entropy",
        f"{head}/approx_kl", f"{head}/approx_kl_max", f"{head}/ratio_max_dev",
        f"{head}/clipfrac",
        f"{head}/value_bias", f"{head}/value_target_mean",
        f"{head}/value_target_std", f"{head}/explained_variance",
        f"{head}/adv_std_raw", f"{head}/batch_size",
    }
    # np.mean over a float32 batch returns np.float32, which is not a float
    # and serialises awkwardly.
    assert all(type(v) is float for v in metrics.values())


@pytest.mark.parametrize("head", ["manager", "worker"])
def test_each_heads_actor_and_critic_share_no_parameters(head):
    """The premise behind HPPOAgent having no vf_coef: with disjoint
    parameters a critic's gradient comes from its value loss alone, so a
    constant weight on that loss is undone by Adam instead of trading the
    head's actor and critic off against each other."""
    agent = _agent()
    actor, critic = getattr(agent, f"{head}_actor"), getattr(agent, f"{head}_critic")
    assert {id(p) for p in actor.parameters()}.isdisjoint({id(p) for p in critic.parameters()})


def test_update_changes_both_networks_of_the_head_and_neither_of_the_other():
    """The heads have separate optimisers and separate losses: a manager update
    must not move the worker, or the two are no longer independently tunable."""
    agent, buf = _prepared("manager")
    manager_before = [p.clone() for p in agent.manager_actor.parameters()]
    critic_before = [p.clone() for p in agent.manager_critic.parameters()]
    worker_before = [p.clone() for p in agent.worker_actor.parameters()]

    agent.update_manager(buf, 4, 2)

    assert any(not torch.allclose(a, b) for a, b in zip(manager_before, agent.manager_actor.parameters()))
    assert any(not torch.allclose(a, b) for a, b in zip(critic_before, agent.manager_critic.parameters()))
    assert all(torch.allclose(a, b) for a, b in zip(worker_before, agent.worker_actor.parameters()))


def test_a_head_update_is_flat_ppos_update():
    """With value-loss clipping gone (2026-09-24) a head's update is flat
    PPO's, step for step. The worker is architecturally flat PPO's actor and
    critic over the (obs, goal) input, so given the same weights, the same
    rollout and the same seed the two must end on identical weights and
    report identical metrics -- which pins the two implementations together
    far more tightly than any single property could."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ppo")))
    import ppo

    in_dim, T, N = OBS_DIM + GOAL_DIM, 8, 4
    hppo_agent = _agent()
    ppo_agent = ppo.PPOAgent(in_dim, ACT_DIM, device="cpu")
    ppo_agent.actor.load_state_dict(hppo_agent.worker_actor.state_dict())
    ppo_agent.critic.load_state_dict(hppo_agent.worker_critic.state_dict())

    torch.manual_seed(0)
    worker_buf = VecRolloutBuffer(T, N, in_dim, ACT_DIM, "cpu")
    ppo_buf = ppo.RolloutBuffer(T, N, in_dim, ACT_DIM, "cpu")
    for _ in range(T):
        states = torch.randn(N, in_dim)
        with torch.no_grad():
            actions, logprobs, _ = hppo_agent.worker_policy_forward(states)
        step = (states, actions, logprobs, torch.randn(N) * 10, torch.randn(N) * 10,
                (torch.rand(N) < 0.2).float())
        worker_buf.add(*step)
        ppo_buf.add(*step)
    next_value = torch.randn(N) * 10
    hppo_agent.compute_worker_returns_and_advantage(worker_buf, next_value)
    ppo_agent.compute_returns_and_advantage(ppo_buf, next_value)

    torch.manual_seed(1)
    hppo_metrics = hppo_agent.update_worker(worker_buf, 4, 3)
    torch.manual_seed(1)
    ppo_metrics = ppo_agent.update(ppo_buf, 4, 3)

    for a, b in zip(list(hppo_agent.worker_actor.parameters()) + list(hppo_agent.worker_critic.parameters()),
                    list(ppo_agent.actor.parameters()) + list(ppo_agent.critic.parameters())):
        assert torch.equal(a, b)
    renamed = {"loss_policy": "policy_loss", "loss_value": "value_loss"}
    for key, value in hppo_metrics.items():
        name = key.split("/", 1)[1]
        if name != "batch_size":
            assert value == ppo_metrics["loss/" + renamed.get(name, name)], key


# --------------------------------------------------------------------------
# Manager-collapse diagnostics (see ManagerActor's docstring)
# --------------------------------------------------------------------------


def test_manager_update_reports_the_pre_normalization_advantage_std():
    """adv_std_raw is the diagnostic the manager-collapse hypothesis says to
    watch; it must reflect the buffer's own GAE advantages, not something
    post-normalization or otherwise disconnected from them."""
    agent, buf = _prepared("manager")
    expected = float(buf.advantages[:buf.step].std())
    metrics = agent.update_manager(buf, 4, 2)
    assert metrics["manager/adv_std_raw"] == pytest.approx(expected, rel=1e-4)


# --------------------------------------------------------------------------
# Optimiser parameter groups
# --------------------------------------------------------------------------

def test_optimisers_split_actor_and_critic_into_separate_groups():
    """Every parameter must still be covered by exactly one group -- ported
    from algorithms/ppo/tests/test_ppo.py's id()-set check, absent here before this."""
    agent = _agent(lr_manager=1e-3, lr_worker=2e-3, critic_lr_mult=3.0)
    for optimizer, base, actor, critic in (
        (agent.manager_optimizer, 1e-3, agent.manager_actor, agent.manager_critic),
        (agent.worker_optimizer, 2e-3, agent.worker_actor, agent.worker_critic),
    ):
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(base * 3.0)

        actor_ids = {id(p) for p in actor.parameters()}
        critic_ids = {id(p) for p in critic.parameters()}
        assert {id(p) for p in optimizer.param_groups[0]["params"]} == actor_ids
        assert {id(p) for p in optimizer.param_groups[1]["params"]} == critic_ids


def test_critic_lr_mult_of_one_restores_a_single_rate():
    agent = _agent(lr_manager=1e-3, lr_worker=1e-3, critic_lr_mult=1.0)
    for optimizer in (agent.manager_optimizer, agent.worker_optimizer):
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-3, 1e-3])


def test_annealing_every_group_keeps_the_critic_ratio():
    """Annealing only param_groups[0] would silently leave each critic at its
    full rate -- the bug the two-group split introduces if the training loop is
    not updated with it."""
    agent = _agent(lr_manager=1e-3, lr_worker=1e-3, critic_lr_mult=3.0)
    for optimizer in (agent.manager_optimizer, agent.worker_optimizer):
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = 0.5 * base_lr
        lrs = [group["lr"] for group in optimizer.param_groups]
        assert lrs[1] / lrs[0] == pytest.approx(3.0)
        assert lrs[0] == pytest.approx(5e-4)


