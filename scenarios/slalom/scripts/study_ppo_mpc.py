"""PPO+MPC seed study on the slalom: train the seeds, then plot what they
learned against the oracle. The phases and flags are algorithms/study.py's,
the same protocol as study_ppo.py and study_hppo.py, so the three studies
overlay in study_compare.py.

    python scenarios/slalom/scripts/study_ppo_mpc.py --seeds 1-10
    python scenarios/slalom/scripts/study_ppo_mpc.py analyze --seeds 1-10

Output goes to scenarios/slalom/studies/ppo_mpc/, which git ignores. The
controller is the pair the checkpoint describes: the manager and a tube-MPC
worker rebuilt with the settings it was trained with (ppo_mpc_train.load_agent).
Runs trained with a disturbance are refused by study.py; see
compare_disturbed.py for those.

Result (2026-10-04, the current dynamics: speed and thrust bounded in
norm, the effort term, the gates' faces walls; docs/disk-limits-and-effort.md).
PPO+MPC seeds 1-10 at 204 800 steps; flat PPO and hPPO seeds 1-20 at
1 024 000, since without the old clamp through the gates' faces they need
far more steps to find the way through (section 7 of that document).
p: Mann-Whitney on the first-solve steps.

                                   PPO+MPC      hPPO           PPO
  solved within the budget         8/10         0/20           9/20
  first solve, median / mean       118k / 125k  --             748k / 701k (p = 6e-4)
  first solve, range               102k-195k    --             338k-973k
  solved at the last evaluation    4/10         0/20           6/20
  solved-checks passed after the
    first solve                    50%          --             52%
  training contacts per episode    0            0.28           0.28
  grid starts on the oracle's step 8%           0%             1%
  grid mean extra steps            +1.36        never arrives  +37.9
  grid effort per episode          -0.073       -0.398         -0.052   (oracle -0.040)

- PPO+MPC is the only algorithm that solves the slalom at the old budget,
  and its worker still never touches a wall. It is less precise than on the
  box: the gates now cost time, and arriving on the oracle's step takes goals
  placed where the time-optimal path threads them; it ends 1.4 steps behind
  the oracle on average, and the strict check (gap <= 5 at all 25 starts)
  flickers, passed by 4 of 10 seeds at the last evaluation.
- hPPO never gets through: every seed ends stopped, waiting out the clock --
  17 between the gates in front of gate 2, 2 in front of gate 1, 1 inside
  gate 1. On the
  tunnel it solves 20/20 at 31k, so it is the gates' faces it cannot get
  past, not the hierarchy that is broken. With the old clamp it reached the
  goal from every start, by taking the jump (~1 contact per episode).
- Flat PPO finds the way through on 14 of 20 seeds, late (first solve
  338k-973k), and then reaches the goal from every start; 6 seeds stay
  stopped in front of a gate. Of the 14, 9 passed the strict check.
- Effort: PPO+MPC's tube worker spends 1.8 times the oracle's effort (its
  tracking cost weighs the input lightly, r = 0.1 against 10 on position
  errors); flat PPO is close to the oracle; hPPO's stopped workers keep
  thrusting to hold position.
- Wall clock, 7 runs in parallel: ~90 min per PPO+MPC run, ~30 min per
  1M-step PPO run and ~35 per hPPO run.

Result (2026-09-29, per-axis limits, no effort term, the old contact rule:
the environments of git tag box-limits-final): seeds 1-10, 204 800 steps, no
disturbance, set against
the 20-seed PPO and hPPO studies of 2026-09-24 (same protocol, same
environment). The p values are Mann-Whitney tests on the first-solve steps.

                                   PPO+MPC      hPPO                  PPO
  solved within the budget         10/10        20/20                 20/20
  first solve, median / mean       72k / 68k    72k / 79k (p = 0.43)  51k / 53k (p = 1.5e-4)
  first solve, range               61k-72k      51k-133k              41k-133k
  solved at the last evaluation    10/10        19/20                 19/20
  solved-checks passed after the
    first solve                    100%         86%                   97%
  training contacts per episode    0            1.09                  0.61
  grid starts on the oracle's step 80%          8%                    79%
  grid mean extra steps            +0.20        +0.96                 +0.21

- Sample efficiency is hPPO's, but without its tail. The median matches,
  and every seed has solved by 72k, where hPPO's run to 133k. Flat PPO
  remains ~20k steps faster.
- It is the most precise at the end. Every seed still passes the strict
  check at the last evaluation, and every check after the first solve
  passed. Its arrival steps match flat PPO's, and are about one step
  ahead of hPPO's.
- No training episode of any seed touched a wall. The worker only ever
  plans admissible states, from the first episode on, whatever goal the
  manager samples. The learned hierarchies average 0.6-1.1 contacts per
  training episode over the run.
- Without a disturbance Z = {0}, and the only clearance left is the margin
  rho = 1 mm. The worker uses it: after gate 2, seed 1's grid rollouts ride
  the lower wall at p_y = -1.999, which is time-optimal and legal in the
  environment. A real plant would need a nonzero W, or a larger rho.
- Wall clock: ~28 min per run, with 10 in parallel.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
# Scenario root, for `envs`; then algorithms/ppo_mpc and algorithms/, for the
# flat `ppo_mpc_train`/`ppo_mpc` and shared modules imported by bare name. This
# script's own directory (for `script_ppo_mpc`) is already on sys.path as the
# running script's.
sys.path.append(os.path.abspath(os.path.join(HERE, '..')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms', 'ppo_mpc')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms')))
from envs.slalom_env import SlalomEnv
from envs.visualize_env import plot_env
from script_ppo_mpc import make_env_config
from ppo_mpc_train import load_agent, make_policy_fn
from study import Study, main


def make_env(run_config):
    """The environment a run trained on, from its config.json: the contact
    penalty and u_max can be overridden, as in hPPO's script."""
    overrides = {"contact_penalty": run_config["contact_penalty"]}
    if run_config.get("env_u_max") is not None:
        overrides["u_max"] = run_config["env_u_max"]
    return SlalomEnv(config=make_env_config(**overrides))


def load_policy(path, env):
    """One worker per checkpoint, shared by every episode replayed from it:
    make_policy_fn resets its slot at the start of each episode, and study.py
    replays episodes one at a time."""
    agent, worker = load_agent(path, env)
    return lambda goal_trace=None: make_policy_fn(agent, worker, goal_trace=goal_trace)


STUDY = Study(
    label="PPO+MPC",
    key="ppo_mpc",
    scenario="slalom",
    train_script=os.path.join(HERE, "script_ppo_mpc.py"),
    make_env=make_env,
    plot_env=plot_env,
    load_policy=load_policy,
    hierarchical=True,
    # The reference palette's third categorical slot (aqua), after PPO's blue
    # and hPPO's orange, in its fixed order.
    color="#1baf7a",
    out_dir=os.path.abspath(os.path.join(HERE, '..', 'studies', 'ppo_mpc')),
)

if __name__ == "__main__":
    main(STUDY)
