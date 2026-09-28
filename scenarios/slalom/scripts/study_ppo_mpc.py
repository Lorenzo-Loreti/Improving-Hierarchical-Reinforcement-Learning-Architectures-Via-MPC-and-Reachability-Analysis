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

Result (2026-09-29): seeds 1-10, 204 800 steps, no disturbance, set against
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
