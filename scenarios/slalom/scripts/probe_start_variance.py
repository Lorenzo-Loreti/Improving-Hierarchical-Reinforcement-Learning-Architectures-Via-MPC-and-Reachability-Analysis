"""How much of flat PPO's policy-gradient noise comes from where its training
episodes start? The mechanism behind the init-sampler experiment
(study_init_sampler.py, docs/init-sampler.md), measured directly.

    python scenarios/slalom/scripts/probe_start_variance.py
    python scenarios/slalom/scripts/probe_start_variance.py --seeds 1-3 --batches 128 --steps 10240,30720,51200,204800

A Sobol' sampler can only remove the part of an estimator's variance that the
randomness of the starts causes. By the law of total variance, the policy
gradient g estimated from one training batch splits as

    Var(g) = Var_starts( E[g | starts] ) + E_starts( Var(g | starts) ),

the first term from which starts the batch happened to draw, the second from
everything else: the sampled actions, and through them which states follow.
This probe estimates the gradient of one PPO update's surrogate at fixed
checkpoints, many times over, under three ways of drawing the starts:

    uniform   independent starts, as flat PPO trained until 2026-10-03
    sobol     a scrambled Sobol' stream, rescrambled for every batch
    fixed     the very same start sequence in every batch: the first term is
              zero, so this is the floor no start sampler can go below

Every condition uses the same per-batch action-noise seeds. The share of the
gradient's variance due to the starts is then 1 - Var(fixed) / Var(uniform),
and Var(sobol) / Var(uniform) is how much of the whole the sampler actually
removes.

What is estimated. One batch exactly as flat PPO collects it (8 environments x
128 steps, the training loop's truncation bootstrap and GAE), the advantages
normalized over the batch as in PPOAgent.update, and the gradient of the
surrogate with respect to the actor's parameters at the collection policy
(ratio = 1, where clipping is inactive): -mean(A * grad log pi(a|s)), the
direction the first minibatch step of the update follows. The entropy bonus
and the critic are left out: neither depends on the advantages. Its variance
across batches is reported as the total variance tr Cov(g), and as a
noise-to-signal ratio tr Cov(g) / |E g|^2.

The checkpoints are the eval_*.pt files of the experiment's uniform PPO arm,
which is the policy flat PPO actually passes through, at a few points of its
run (before, around and after the first solve, ~51k steps). The probe changes
no training code and draws its own seeds. Results go to
scenarios/slalom/studies/init_sampler/probe/, which git ignores.

Result on the current dynamics (2026-10-06; docs/disk-limits-and-effort.md):
seeds 1-3 of the 1M-step independent PPO arm, checkpoints at 102k, 307k, 614k
and 1024k steps (flat PPO now first solves at 338k-973k, if at all; seed 1
at 338k, seeds 2 and 3 never), 128 batches per condition. Over all 12
checkpoints, Sobol' / independent = 0.94 (0.85-1.02) and fixed / independent
= 0.96 (0.87-1.05): the starts are +4 % (-5 % to +13 %) of the gradient's
variance, as before. Run with --steps 102400,307200,614400,1024000.

Result on the old dynamics (2026-10-03, git tag box-limits-final): seeds 1-3,
checkpoints at 10k, 31k, 51k and 205k steps,
128 batches per condition. Over all 12 checkpoints (geometric mean, 95 %
bootstrap interval), Sobol' / independent = 0.94 (0.88-1.02) and fixed /
independent = 0.97 (0.89-1.05): the starts are +3 % (-5 % to +11 %) of the
gradient's variance, and no single checkpoint's interval excludes 1. This is
why the sampler changed nothing in study_init_sampler.py: almost all of the
noise comes from the sampled actions. See docs/init-sampler.md, section 8.2.
"""
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Scenario root, for `envs`; then algorithms/ppo and algorithms/, for the
# flat modules imported by bare name.
sys.path.append(os.path.abspath(os.path.join(HERE, '..')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms', 'ppo')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms')))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import study_plots
from envs.slalom_env import SlalomEnv
from envs.vec_slalom_env import SlalomVecEnv
from ppo import RolloutBuffer
from ppo_train import load_agent
from script_ppo import make_env_config
from study import _parse_seeds

CONDITIONS = ("uniform", "sobol", "fixed")
CONDITION_LABEL = {"uniform": "independent", "sobol": "Sobol'", "fixed": "fixed starts (floor)"}
CONDITION_COLOR = {"uniform": "#9a9893", "sobol": "#2a78d6", "fixed": study_plots.INK}

# Flat PPO's rollout at its defaults (algorithms/ppo/ppo_train.py).
NUM_ENVS, NUM_STEPS = 8, 128
# Seeds of the probe's own batches, clear of every training seed. The
# "fixed" condition reuses FIXED_START_SEED's start sequence in every batch.
BATCH_SEED_BASE = 100_000
FIXED_START_SEED = 99_999

DEFAULT_RUNS = os.path.abspath(os.path.join(HERE, '..', 'studies', 'init_sampler', 'ppo_uniform', 'runs'))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, '..', 'studies', 'init_sampler', 'probe'))


def batch_gradient(agent, vec_env, start_seed, action_seed):
    """The surrogate's actor gradient from one freshly collected batch, as a
    flat numpy vector."""
    torch.manual_seed(action_seed)
    buffer = RolloutBuffer(NUM_STEPS, NUM_ENVS, 4, 2, "cpu")
    obs, _ = vec_env.reset(seed=start_seed)
    with torch.no_grad():
        for _ in range(NUM_STEPS):
            obs_tensor = torch.tensor(agent.normalize_obs(obs), dtype=torch.float32)
            action, logprob, value = agent.get_action_and_value(obs_tensor)
            next_obs, reward, terminated, truncated, info = vec_env.step(action.numpy())
            # The training loop's truncation bootstrap (ppo_train.py).
            trunc_only = truncated & ~terminated
            if np.any(trunc_only):
                final_obs = agent.normalize_obs(info["final_observation"][trunc_only])
                reward[trunc_only] += agent.gamma * agent.get_value(
                    torch.tensor(final_obs, dtype=torch.float32)).numpy()
            done = terminated | truncated
            buffer.add(obs_tensor, action, logprob, torch.tensor(reward, dtype=torch.float32), value,
                       torch.tensor(done.astype(np.float32)))
            obs = next_obs
        next_value = agent.get_value(torch.tensor(agent.normalize_obs(obs), dtype=torch.float32))
        agent.compute_returns_and_advantage(buffer, next_value)
    states, actions, logprobs, _, _, advantages = buffer.get()
    adv_std, adv_mean = torch.std_mean(advantages)
    advantages = (advantages - adv_mean) / (float(adv_std) + 1e-8)
    _, newlogprob, _ = agent.policy_forward(states, actions)
    loss = -(advantages * (newlogprob - logprobs).exp()).mean()
    grads = torch.autograd.grad(loss, list(agent.actor.parameters()))
    return torch.cat([g.reshape(-1) for g in grads]).numpy().astype(np.float64)


def variance_stats(grads, rng, n_boot=1000):
    """Total variance, noise-to-signal ratio, and bootstrap resamples of the
    total variance (over batches) for confidence intervals of ratios."""
    grads = np.asarray(grads)
    total = float(grads.var(axis=0, ddof=1).sum())
    mean = grads.mean(axis=0)
    boot = np.array([grads[idx].var(axis=0, ddof=1).sum()
                     for idx in rng.integers(0, len(grads), size=(n_boot, len(grads)))])
    return {"total_var": total, "nsr": total / float(mean @ mean), "boot": boot}


def probe_checkpoint(path, batches):
    torch.set_num_threads(1)
    env = SlalomEnv(config=make_env_config())
    agent = load_agent(path, env)
    result = {}
    for condition in CONDITIONS:
        sampler = "uniform" if condition == "uniform" else "sobol"
        vec_env = SlalomVecEnv(num_envs=NUM_ENVS, config=make_env_config(init_sampler=sampler))
        grads = []
        for k in range(batches):
            start_seed = FIXED_START_SEED if condition == "fixed" else BATCH_SEED_BASE + k
            grads.append(batch_gradient(agent, vec_env, start_seed, action_seed=BATCH_SEED_BASE + k))
        result[condition] = np.asarray(grads)
    return result


def find_checkpoint(runs_dir, seed, step):
    matches = glob.glob(os.path.join(runs_dir, f"ppo_slalom_{seed}_*", f"eval_{step:07d}.pt"))
    if not matches:
        sys.exit(f"no eval_{step:07d}.pt for seed {seed} in {runs_dir}: train the experiment's "
                 f"uniform PPO arm first (study_init_sampler.py)")
    return sorted(matches)[-1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=str, default=DEFAULT_RUNS,
        help="the runs/ directory of a flat-PPO seed study with eval checkpoints")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=str, default="1-3", help="training seeds whose checkpoints to probe")
    parser.add_argument("--steps", type=str, default="10240,30720,51200,204800",
        help="checkpoint steps: early, mid-learning, around the median first solve (51k), end")
    parser.add_argument("--batches", type=int, default=128, help="batches per condition and checkpoint")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="checkpoints probed in parallel, one process each")
    parser.add_argument("--plot-only", action="store_true", help="redraw from probe.pkl")
    args = parser.parse_args()
    args.seed_list = _parse_seeds(args.seeds)
    args.step_list = [int(s) for s in args.steps.split(",")]
    return args


def run(args):
    rng = np.random.default_rng(0)
    pairs = [(seed, step) for seed in args.seed_list for step in args.step_list]
    paths = [find_checkpoint(args.runs, seed, step) for seed, step in pairs]
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(pairs)))) as pool:
        all_grads = list(pool.map(probe_checkpoint, paths, [args.batches] * len(paths)))
    rows = []
    for (seed, step), grads in zip(pairs, all_grads):
        stats = {c: variance_stats(g, rng) for c, g in grads.items()}
        row = {"seed": seed, "step": step, **{c: stats[c] for c in CONDITIONS}}
        u, s, f = (stats[c]["total_var"] for c in CONDITIONS)
        print(f"seed {seed} step {step}: tr Cov uniform {u:.3e}, sobol {s:.3e} ({s / u:.2f}), "
              f"fixed {f:.3e} ({f / u:.2f}); start share {1 - f / u:+.2f}")
        rows.append(row)
    return {"rows": rows, "batches": args.batches}


def summarize(probe, out):
    rows = probe["rows"]
    lines = [
        "# Where flat PPO's policy-gradient noise comes from: the starts or the rest",
        "",
        f"{probe['batches']} batches per condition and checkpoint (one batch = 8 x 128 steps, as in "
        "training). tr Cov: total variance of the surrogate's actor gradient across batches. "
        "Ratios are against the independent starts, with 95 % bootstrap intervals over batches. "
        "Start share = 1 - fixed / independent: the part of the variance due to which starts a "
        "batch drew, the most any start sampler could remove.",
        "",
        "| seed | checkpoint | tr Cov, independent | Sobol' / independent | fixed / independent "
        "| start share | noise-to-signal, independent |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        u = r["uniform"]
        ratio = lambda c: (r[c]["total_var"] / u["total_var"],
                           *np.percentile(r[c]["boot"] / u["boot"], [2.5, 97.5]))
        s, f = ratio("sobol"), ratio("fixed")
        lines.append(f"| {r['seed']} | {r['step'] / 1e3:.0f}k | {u['total_var']:.3e} "
                     f"| {s[0]:.2f} ({s[1]:.2f}-{s[2]:.2f}) | {f[0]:.2f} ({f[1]:.2f}-{f[2]:.2f}) "
                     f"| {1 - f[0]:+.2f} | {u['nsr']:.1f} |")

    # Every checkpoint at once: the geometric mean of the ratios, its interval
    # from the per-checkpoint bootstraps (independent across checkpoints, so
    # they combine index by index).
    def pooled(c):
        point = np.exp(np.mean([np.log(r[c]["total_var"] / r["uniform"]["total_var"]) for r in rows]))
        boot = np.exp(np.mean([np.log(r[c]["boot"] / r["uniform"]["boot"]) for r in rows], axis=0))
        return point, *np.percentile(boot, [2.5, 97.5])
    s, f = pooled("sobol"), pooled("fixed")
    lines.append(f"| **all {len(rows)}** | | | **{s[0]:.2f}** ({s[1]:.2f}-{s[2]:.2f}) "
                 f"| **{f[0]:.2f}** ({f[1]:.2f}-{f[2]:.2f}) | **{1 - f[0]:+.2f}** "
                 f"({1 - f[2]:+.2f} to {1 - f[1]:+.2f}) | |")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(out, 'summary.md')}")


def fig_variance(probe, out):
    """Total gradient variance per condition, relative to independent starts,
    at each checkpoint step; one marker per seed."""
    rows = probe["rows"]
    steps = sorted({r["step"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for i, condition in enumerate(CONDITIONS):
        for j, step in enumerate(steps):
            values = [r[condition]["total_var"] / r["uniform"]["total_var"] for r in rows if r["step"] == step]
            x = j + (i - 1) * 0.22
            ax.scatter([x] * len(values), values, s=16, color=CONDITION_COLOR[condition],
                       label=CONDITION_LABEL[condition] if j == 0 else None, zorder=3)
    ax.axhline(1.0, color=study_plots.MUTED, lw=0.8, zorder=1)
    ax.set_xticks(range(len(steps)), [f"{s / 1e3:.0f}k" for s in steps])
    ax.set_xlabel("checkpoint (environment steps)")
    ax.set_ylabel("tr Cov(g) / independent")
    ax.set_ylim(bottom=0)
    ax.set_title("Policy-gradient variance by how the batch's starts are drawn (flat PPO, slalom)")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    study_plots._save(fig, os.path.join(out, "figures"), "gradient_variance")


def main():
    args = parse_args()
    path = os.path.join(args.out, "probe.pkl")
    if args.plot_only:
        with open(path, "rb") as fh:
            probe = pickle.load(fh)
    else:
        probe = run(args)
        os.makedirs(args.out, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(probe, fh)
    summarize(probe, args.out)
    with plt.rc_context(study_plots.STYLE):
        fig_variance(probe, args.out)


if __name__ == "__main__":
    main()
