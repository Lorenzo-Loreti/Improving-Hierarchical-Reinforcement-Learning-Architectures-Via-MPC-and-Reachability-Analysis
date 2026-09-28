"""The hPPO training loop, shared by every scenario's script_hppo.py.

scenarios/slalom/scripts/script_hppo.py and scenarios/tunnel/scripts/
script_hppo.py used to be ~1100-line copies of this loop whose code was
identical: they differed only in which environment classes they built, three
defaults and their comments and help texts -- and those had drifted. The
slalom's carried the full diagnoses and ablations, the tunnel's pointed back
to them and added a few tunnel-specific notes, and one had simply gone stale
(the tunnel's --worker-extrinsic-coef help quoted the slalom's +20 terminal,
where the tunnel's goal_reward of 200 gave +4; since 2026-09-24 the tunnel
uses the slalom's 1000, so +20 now holds for both). The texts below are the
slalom's, with the tunnel's notes merged in; each script is now a thin wrapper
that describes its scenario in a `Scenario` and hands it to `main`, as flat
PPO's do (algorithms/ppo/ppo_train.py).

Deliberately a separate module from ppo_train.py rather than one loop with an
`if hierarchical` in it: each algorithm's loop stays self-contained, so one
can be changed for an ablation without touching another's (see
algorithms/common.py).

Imported by bare name, like `hppo`, once algorithms/hppo and algorithms/ are
on sys.path -- each script_hppo.py puts them there.
"""

import argparse
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch
import wandb

from hppo import HPPOAgent, VecRolloutBuffer, ManagerVecRolloutBuffer, WORKER_OBS, checkpoint_worker_obs
from optimal_solver import spawn_grid, precompute_optimal_grid, MinTimeSolver
from solved_check import check_solved
from metrics_log import MetricsLog


# `type=bool` would be a no-op for the boolean flags below: argparse applies
# it to the *string*, and bool("False") is True, so a flag could only ever be
# turned on.
def _str2bool(x):
    return x.lower() in ['true', '1', 't', 'y', 'yes']


@dataclass(frozen=True)
class Scenario:
    """Everything the training loop needs to know about one scenario."""
    name: str                   # "slalom" / "tunnel": exp-name default and help texts
    env_cls: type               # plain env, for evaluation and the solved-check
    vec_env_cls: type           # batched, auto-resetting env, for rollouts
    make_env_config: Callable   # **overrides (u_max, contact_penalty) -> env config
    total_timesteps: int        # default training budget
    wandb_project: str
    env_u_max_help: str         # scenario-specific: each is a different arm of the regime study
    script_dir: str             # checkpoints are written relative to the scenario's script


def parse_args(scenario):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=f"hppo_{scenario.name}",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=_str2bool, default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=_str2bool, default=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", action="store_true",
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default=scenario.wandb_project,
        help="the wandb's project name")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
        help="directory (relative to the scenario's script) to save model checkpoints in")
    parser.add_argument("--save-eval-checkpoints", type=_str2bool, default=False,
        help="if toggled, also save the agent at every evaluation, as "
             "eval_<global_step>.pt next to best.pt and final.pt, so the "
             "hierarchy can be replayed at any point of its learning curve (the "
             "study scripts' progression figure, algorithms/study.py, uses "
             "them). Off by default: each is ~250 kB. Saving touches no random "
             "number generator, so the run itself is unchanged")

    # Reward-scale overrides. Default to the scenario env config's own defaults, so
    # omitting these reproduces exactly the environment every other script
    # (flat PPO, PPO+MPC) trains against -- passing them is how a single run
    # can test a different risk/reward balance without moving that shared
    # default out from under everyone else. See envs/config.py.
    parser.add_argument("--env-u-max", type=float, default=None,
        help=scenario.env_u_max_help)
    parser.add_argument("--contact-penalty", type=float, default=scenario.make_env_config().contact_penalty,
        help="small penalty applied every step the agent is in contact with a "
             "wall, flat regardless of impact speed -- no longer a terminal "
             "reward: a contact clamps the agent to the wall and absorbs "
             "v_y, but the episode continues. Defaults to the env config's "
             "own default, so omitting this reproduces the shared environment "
             "every other script trains against -- see envs/config.py")

    parser.add_argument("--noise-bound-p", type=float, default=0.0,
        help="half-width, in metres, of the uniform disturbance added to each "
             "position every step (the env config's noise_bound_p, since "
             "2026-09-28). 0.0, the default, is the deterministic environment "
             "every earlier run trained on, unchanged bit for bit. The same "
             "flag as PPO+MPC's (algorithms/ppo_mpc/ppo_mpc_train.py), so the "
             "algorithms can be compared on one disturbed environment; the "
             "level chosen there is 0.005 with --noise-bound-v 0.05")
    parser.add_argument("--noise-bound-v", type=float, default=0.0,
        help="half-width, in m/s, of the uniform disturbance added to each "
             "velocity every step (the env config's noise_bound_v); see "
             "--noise-bound-p")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=scenario.total_timesteps,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate-manager", type=float, default=3e-4,
        help="the learning rate of the manager's actor. Its critic runs at "
             "--critic-lr-mult times this")
    parser.add_argument("--learning-rate-worker", type=float, default=3e-4,
        help="the learning rate of the worker's actor. Its critic runs at "
             "--critic-lr-mult times this")
    parser.add_argument("--critic-lr-mult", type=float, default=3.0,
        help="each critic's learning rate as a multiple of its head's rate. "
             "Every critic gets its own optimiser parameter group; 1.0 "
             "restores the single-rate behaviour")
    parser.add_argument("--anneal-lr", type=_str2bool, default=False,
        help="if toggled, the learning rates decay linearly over training, in "
             "every parameter group of both optimisers. Off by default: on "
             "the slalom, with a decaying LR, every seed that solved the task "
             "in testing did so by escaping an intermediate local optimum (a 'rush and "
             "crash' policy that ignores the walls) at some update that "
             "varies by seed, and annealing to ~0 by the end of a fixed "
             "training budget starved that escape of gradient signal before "
             "it could happen on several seeds. See envs/config.py's "
             "contact_penalty comment for the fuller story. Annealing all "
             "the way to 0 also freezes the last updates (approx_kl "
             "collapses to ~0), burning the tail of --total-timesteps on a "
             "policy that can no longer move; a --lr-floor-frac that stopped "
             "the decay short of 0 existed for that, and was removed along "
             "with the other options no run enabled by default. That escape "
             "predates the worker-reward fix; re-measured after it "
             "(2026-09-24, 13 slalom seeds, full 500k budget, every early "
             "stop disabled, with clip_vloss still on), annealing reached the "
             "first solve no faster (median / mean 82k / 89k steps against "
             "92k / 97k, p = 0.22) and held the strict solved criterion on "
             "fewer checks after it (80%% against 91%%; one seed not solved "
             "at the end), though no evaluation fell below 95%% of the "
             "oracle in either arm. So it stays off -- while flat PPO keeps "
             "annealing, which does help it hold the solved criterion (see "
             "--num-steps in algorithms/ppo/ppo_train.py)")
    parser.add_argument("--manager-freq", type=int, default=10,
        help="c: the number of steps the manager's goal is valid for. Also "
             "fixes the manager's discount, gamma**c")
    parser.add_argument("--max-goal-bound", type=float, default=10.0,
        help="the manager's [-1, 1] action is scaled by this to a physical "
             "goal displacement in metres, and divided by it again to form "
             "the worker's input. On x only when --max-goal-bound-y is set")
    parser.add_argument("--max-goal-bound-y", type=float, default=None,
        help="a separate bound for the goal's lateral (y) axis, in metres; "
             "default: --max-goal-bound on both axes. The two axes ask "
             "different things of a goal: forward, a far goal makes the worker "
             "move at full speed; laterally, a reachable one says 'be at this "
             "y', and a short box shrinks the manager's lateral sampling noise "
             "(its floor is ~0.1 of the box, see ManagerActor's "
             "MAX_CONCENTRATION). One box cannot give both; see the "
             "--worker-obs comment for the measurements that led here")
    # --- worker termination-avoidance ------------------------------------
    #
    # The worker is trained on a purely *intrinsic* reward: the per-step
    # reduction in the distance to the manager's goal. That reward stream has
    # no terminal term, so in the worker's own MDP reaching the environment's
    # goal line is an absorbing state of value exactly 0, while *not* reaching
    # it keeps paying ~v_max*dt approx 0.12 per step forever (the manager hands
    # out a fresh, effectively unreachable goal every --manager-freq steps, so
    # the stream never runs dry). Discounted at --gamma that is a value of
    # O(10) for never finishing against 0 for finishing: the worker's optimal
    # policy under its own reward is to approach the goal line and stall.
    #
    # That is exactly what a 500k-step slalom run does. Around update ~50 (of
    # 244) the worker starts braking near p_x = L, episode length climbs
    # 80 -> 200, and eval success falls 1.00 -> 0.00 over ~30 updates and
    # never recovers. Probing the collapsed checkpoint at (p_x, p_y, v) =
    # (9.5, 0, 1.2, 0): the worker's deterministic a_x is between -1.6 and
    # -2.1 for *every goal in the manager's entire action box* -- no goal the
    # manager can emit makes it cross the line, which is why the hierarchy has
    # no in-run recovery path. Flat PPO and both PPO+MPC variants are immune
    # for the same reason: neither has a *learned* worker optimizing a
    # reward stream that termination cuts off (flat PPO optimizes the
    # environment's own reward, whose terminal is +goal_reward; PPO+MPC's
    # worker is a QP, not a policy).
    #
    # The pathology is a property of this worker reward, not of the slalom,
    # so every scenario carries the same knob at the same default. The
    # tunnel does not discriminate either way, being solved long before the
    # pathology can bind within its budget; its default is set to match the
    # slalom's anyway, so that the two scenarios compare *the same algorithm*.
    #
    # There used to be two knobs, two different ways to put the environment's
    # terminal back into the worker's return, 0.0 restoring the pre-fix reward
    # exactly for either: --worker-extrinsic-coef (below) and
    # --worker-success-bonus (removed 2026-09-24; its record is further down).
    # A 4-variant x 2-seed, 500k-step ablation with
    # --early-stop-success-rate and --solved-early-stop *disabled* (so a fix
    # that merely delays the collapse cannot be mistaken for one that removes
    # it) settled which is needed:
    #
    #   variant                          clean solves   final eval return
    #   ------------------------------   ------------   ------------------
    #   baseline (both knobs 0.0)            0/2        -153 / -997
    #   --ent-coef-lr 0.05                   0/2        -181 /  615
    #   --worker-success-bonus 20            2/2        1018 / 1019
    #   --worker-extrinsic-coef 0.02         2/2        1019 / 1019
    #
    # (--ent-coef-lr no longer exists: the entropy autotuner it sped up was
    # later removed as inert at its default -- see HPPOAgent's class comment.)
    #
    # against a mean oracle optimum of ~1013 (see the "Solved-check:
    # precomputed optimal returns" line printed at startup); "clean solve"
    # is the final evaluation sitting at >= 95% of that, which separates a
    # genuine solve from the two ways an unfixed run ends -- stalling short
    # of the line at 200 steps/episode (seed 1), or reaching it by smashing
    # through both gates (seed 2: success_rate 1.00 at return -997). Two
    # things that
    # ablation settles beyond the headline: (1) the pathology is the *worker's*
    # reward, not the manager's exploration -- fixing the entropy autotuner
    # (whose dual-ascent step is lr-bounded at ~3e-4 per update, so at the
    # default it moves ent_coef by 8% over a whole run while the manager's
    # entropy falls from +1.2 to -2.2 nats) changes when the collapse happens
    # and not whether it does; (2) both fixes do not merely prevent the
    # collapse, they close the optimality gap -- the unfixed hierarchy peaked
    # at ~745 with 3-8 wall contacts per episode even *before* collapsing,
    # because the worker had no reason to care about anything but the goal
    # vector.
    #
    # --worker-extrinsic-coef is the default, and the reasoning that first
    # picked --worker-success-bonus instead is worth recording because the
    # evidence overturned it. The success bonus is the *minimal* intervention:
    # it changes the terminal only and leaves the worker otherwise blind to
    # the environment's reward, which is what keeps this a feudal hierarchy
    # rather than two agents optimizing the same objective at different
    # rates. On the 2-seed ablation above the two tie on final quality, so
    # that argument decided it.
    #
    # It does not survive the full benchmark. Re-running both arms at 3 seeds
    # under the benchmark protocol (uniform 500k budget, --solved-early-stop
    # left on and --early-stop-success-rate 2.0 to disable the looser stop)
    # separates them on samples rather than on quality:
    #
    #   hppo, slalom   success-bonus  3/3 solved, 266k / 296k / 399k steps,
    #                                 final return 1018 / 1019 / 1017
    #                  extrinsic      3/3 solved,  71k /  81k /  81k steps,
    #                                 final return 1015 / 1016 / 1016
    #
    # A 3-5x difference in samples at the same final policy, and the reason
    # is what the success bonus does *not* do. It repairs the termination
    # incentive, so the worker does finish -- success_rate 1.00 on every seed
    # -- but nothing in its reward charges it for a wall or for the clock, so
    # every collision still has to be steered out by a manager that only acts
    # every --manager-freq steps, and that indirection is expensive.
    # Restoring the terminal alone makes the worker's MDP *sound*; it does
    # not make it *aligned*.
    #
    # The size of that gap also changes the headline comparison: at
    # 71k-81k steps this hierarchy is no longer measurably slower than flat
    # PPO (51k-81k) on the slalom, where under the success bonus it read as a
    # clean 4-6x hierarchy tax. See docs/benchmark.md section 3.
    #
    # So: feudal purity was the right prior and the wrong call. 0.02 is the
    # default.
    #
    # --worker-success-bonus, removed 2026-09-24. It was kept for a while
    # after the comparison above as the minimal-intervention ablation arm,
    # the one to use when the question is specifically "what does the
    # terminal alone fix?": a one-off bonus added to the worker's reward on
    # the step the episode terminated successfully,
    #
    #   worker_succeeded = terminated & info["final_info"]["is_success"]
    #   worker_reward[worker_succeeded] += args.worker_success_bonus
    #
    # applied right after the extrinsic mix below. It read the success flag
    # rather than `terminated` alone, so that adding another terminal
    # condition later could not silently start paying it. It was sized
    # against the value the worker forgoes by crossing, not against the
    # environment's reward: the worker earns about v_max*dt per step
    # indefinitely, so what it gives up by terminating is v_max*dt/(1-gamma)
    # approx 12 at the defaults, which the 20.0 of the runs above cleared
    # with ~1.7x margin (to be re-derived if --gamma, --manager-freq or the
    # velocity limit move). Setting it together with the extrinsic mix was
    # double-counting, since the mix already carries goal_reward into the
    # worker's return. It defaulted to 0.0 and no run used it after this
    # comparison, so it went; the numbers above are its record, and
    # docs/worker-termination-avoidance.md section 5.1 has the rest.
    parser.add_argument("--worker-extrinsic-coef", type=float, default=0.02,
        help="FeUdal-Networks-style mixing coefficient: add this times the "
             "environment's own reward to the worker's intrinsic reward, so "
             "the worker sees r_int + coef * r_env instead of r_int alone. "
             "The broader alternative to the removed --worker-success-bonus "
             "(see above): it fixes the "
             "same termination-avoidance pathology (goal_reward enters the "
             "worker's return) and additionally makes the worker itself aware "
             "of wall contacts and of the clock, instead of leaving every "
             "collision the manager's problem. At the env config defaults, "
             "which both scenarios share since 2026-09-24, a "
             "coefficient of 0.02 puts the terminal at +20 (it was +4 on the "
             "tunnel while its goal_reward was 200), a contact at -1.0 and a "
             "step at -0.02 in the worker's units, "
             "against an intrinsic stream of ~0.12/step -- which is why it "
             "is the default: unlike the success bonus, it makes the "
             "worker itself avoid walls, and the _reach variants need that. "
             "The coefficient has a floor: "
             "stalling in front of the line is worth about "
             "(r_orbit - coef)/(1-gamma) to the worker, where r_orbit is the "
             "goal progress an orbit harvests per step (the step penalty is "
             "the only extrinsic term that survives an orbit), and crossing it "
             "about coef*goal_reward, so the terminal only wins above "
             "coef = (r_orbit/(1-gamma)) / (goal_reward + 1/(1-gamma)). "
             "Bounding r_orbit by v_max*dt = 0.12 gives ~0.011, and that bound "
             "proved conservative. "
             "The sweep (scenarios/slalom/scripts/sweep_extrinsic_coef.py, "
             "2026-09-24; 10 seeds per arm at 500k with every early stop off) "
             "found 0.01 and 0.005 as sound as 0.02: 10/10 solved, first "
             "solve median 72k / 82k against 72k (p = 0.78 / 0.46), 10/10 "
             "still solved at the end, and no collapse. The collapsed runs at "
             "0 harvest r_orbit ~0.055, which puts the measured floor near "
             "0.005 (no margin left there). It applies to both scenarios now "
             "that they share goal_reward. At the tunnel's former 200 the "
             "floor was ~0.018, and its 0.02 sat just above it. The same "
             "sweep found the cost of a larger coefficient on the "
             "hierarchy side: at 0.02 the worker threads the slalom almost "
             "unaided, with ~1 wall contact per episode when the manager is "
             "replaced by a fixed forward goal, against ~6 at 0.005. See "
             "that script's summary. "
             "0.0 restores the pre-fix reward, which collapses (2/10 seeds "
             "ever solved in the sweep, 0/10 at the end)")
    # --- worker observation: a strict feudal separation (2026-09-26) -------
    #
    # The extrinsic-coef sweep (scenarios/slalom/scripts/sweep_extrinsic_coef.py)
    # found that at 0.02 the manager is close to ornamental. With the manager
    # replaced by a fixed forward goal, the worker still threads the slalom
    # with ~1 wall contact per episode, and on 2/10 seeds loses nothing at
    # all. Seed 1's manager emits nearly the same goal, (5.7, -4.5) m, in
    # every state, even in front of gate 1 at y = +1. The worker has learned
    # the task itself, which it can because it sees the full state. Lowering
    # the coefficient only moves the balance (~6 contacts without the manager
    # at 0.005, near the coefficient's floor).
    #
    # --worker-obs velocity removes the channel instead. The worker sees
    # (v_x, v_y, goal) and not (p_x, p_y). The argument for why that is enough:
    # the plant is a double integrator, invariant under translation, so
    # reaching a relative goal needs the velocity and the goal and nothing
    # else. Only two things break the invariance, the walls and the goal line,
    # and both are the task -- the manager's business. A worker that cannot
    # see them can learn to reach goals but not to do the slalom.
    #
    # Two predictions this makes, both to be measured rather than assumed:
    #
    # - The termination-avoidance pathology (the block above) needs the
    #   worker to recognise that it is near the line, which it did from p_x.
    #   A worker blind to position cannot stall selectively there, and
    #   stalling everywhere costs it its intrinsic reward. If so, the
    #   extrinsic term loses its first reason, and --worker-extrinsic-coef 0
    #   becomes viable: a hierarchy that is feudal in reward as well as in
    #   information. The residual risk is that the manager's goals themselves
    #   give away that the line is near; charts/worker_ax_near_goal is the
    #   direct test.
    # - It also loses its second reason, making the worker avoid walls,
    #   because a blind worker cannot see them. To it a contact penalty is
    #   unpredictable noise, which could make it timid laterally.
    #
    # Where it can fail, most likely first:
    #
    # 1. Lateral precision. ManagerActor caps both Beta concentrations at
    #    MAX_CONCENTRATION = 50, which floors the sampling std of a goal
    #    coordinate at ~0.1 of the action box: ~1 m at --max-goal-bound 10.
    #    Trained managers sample goal_y with a std of 1.3-1.85 m, measured
    #    2026-09-26. A gate's half-width is 0.75 m. A sighted worker filters
    #    that noise because it knows where the gates are. A blind one
    #    executes it, so training rollouts fill with contacts and the
    #    manager's advantages with noise. Deterministic evaluation, which
    #    uses the mean, can still be clean. A reachable goal box (1.5x reach,
    #    ~1.8 m; see docs/goal-box-saturation.md) floors it at ~0.18 m. That
    #    is a second variable, so it gets its own arm.
    # 2. Cadence. A segment is 1 s and at most 1.2 m, and a gate is 1 m long.
    #    Going from gate 1 (y = +1) to gate 2 (y = -1) takes 2 m of lateral
    #    travel in 2 m of forward travel, a diagonal across two segments,
    #    which has to be set up before each gate. The oracle does it at no
    #    time cost. A manager that commits once per second may not, so the
    #    strict solved-check can become unreachable at 100% success.
    # 3. Co-adaptation. Until the worker can track goals, the manager's
    #    goals do not produce predictable motion. A blind worker is
    #    task-agnostic, so it could be pretrained on random goals in an empty
    #    arena and then frozen; that is not done here.
    # 4. A partially observable worker critic. Its return depends on walls,
    #    the line and future goals it cannot see. Goal reaching is easy, so
    #    this is probably minor.
    #
    # First measurement (2026-09-26): slalom, 10 seeds per arm, 1M steps,
    # every early stop off, goal box 10 m; the output is in
    # scenarios/slalom/studies/blind_worker. The sighted arms are the 500k
    # extrinsic sweep; hPPO's constant learning rate makes a 500k run a prefix
    # of a 1M one.
    #
    #   worker   coef   ever solved  first solve (median)  solved at end  gap, manager -> forward goal
    #   sighted  0.02      10/10          72k                  10/10          0 -> 53
    #   sighted  0          2/10         271k                   0/10     (collapsed at the line)
    #   blind    0.02       8/10         461k                   7/10          0 -> 437
    #   blind    0          9/10         195k                   5/10          6 -> 1801
    #
    # - The separation holds. With a blind worker at 0, the manager is
    #   indispensable: replacing it with a forward goal costs ~1800 of
    #   return, and a mirrored goal costs ~4600. The goal explains 96% of
    #   the worker's action variance.
    # - The first prediction holds. No blind seed stalls at the line; with
    #   the sighted worker at 0, every seed does. The one blind failure
    #   (seed 10 at 0) is a different one. Its manager never learned gate 2:
    #   it orbits between the gates at x ~ 6, alternating backward-down and
    #   forward-up goals, the old "advance, then idle" local optimum, now
    #   the manager's rather than the worker's.
    # - The cost is samples and precision. At 0 the blind hierarchy reaches
    #   the oracle (best logged gap 0.4-1.3) and then drifts to 2-9 steps
    #   slower, with no contacts. That looked like risk 1 at work -- round 3
    #   below contradicts it. Sampled goals
    #   give the blind worker 0.66 contacts per training episode, against
    #   0.13 for the sighted one at 0.02. The manager compensates with
    #   caution: it takes the gates at y ~ +-0.7, where the oracle takes
    #   them at +0.29 / -0.41, and it slows down (26 steps below
    #   v_x = 1.1 m/s, against the oracle's 7).
    # - 0.02 does not help a blind worker. It learns 2.4x slower (p = 0.07),
    #   and the progress term gives the worker a forward bias of its own. The
    #   goal then explains only 64% of its action variance, which is a weak
    #   leak of the task back into the worker. In exchange it ends within ~1
    #   step of the oracle.
    #
    # The next step this points to is a reachable goal box (--max-goal-bound
    # ~1.8) with a blind worker at 0. It lowers risk 1's noise floor from ~1 m
    # to ~0.18 m.
    #
    # Measured (2026-09-28; 10 seeds, 1M steps,
    # scenarios/slalom/studies/blind_worker_box1.8), it fails outright.
    # - 0/10 seeds ever solve, and none reaches the goal in any evaluation.
    #   Early sampled rollouts did cross (12-44% training success), but every
    #   manager converged to the same orbit: it takes gate 1, then oscillates
    #   at x ~ 6-6.5 in front of gate 2 with near-zero lateral goals. The
    #   return is ~-150 with no contacts, the same local optimum as round
    #   1's seed 10, now on every seed.
    # - The worker is not the problem. It has become a proper waypoint
    #   follower: at v_x = 1.2 it brakes (a_x -1.7) for a goal 0.3 m ahead
    #   and accelerates (+2.1) for one 1.8 m ahead.
    # - That is also what the reachable box changes. Every goal is now a
    #   point to stop at, so the hierarchy cruises at ~0.8 m/s, not 1.2.
    #   The lateral transfer to gate 2 (~2.2 m) has to be held at the edge
    #   of the box over consecutive segments. A 10 m box got it from any
    #   clearly lateral goal, because a far goal makes the worker move at
    #   full speed.
    # - A shorter box therefore lowers the lateral noise at the price of the
    #   forward drive and of lateral authority. The two axes want different
    #   bounds: a far forward goal (move at full speed) and a reachable
    #   lateral one (be at this y). That points to a per-axis goal box, say
    #   10 m in x and ~2.5 m in y (lateral noise floor ~0.25 m, one goal
    #   enough for the transfer), as the next arm: --max-goal-bound-y.
    #
    # Round 3 (2026-09-28): the per-axis box, --max-goal-bound-y 2.5, with a
    # blind worker at 0; 10 seeds, 1M steps,
    # scenarios/slalom/studies/blind_worker_boxy2.5. It fixes round 2's stall
    # but is worse than round 1:
    #
    #   box (x / y)   ever solved  first solve  solved at end  grid gap  contacts/ep  training contacts
    #   10 / 10          9/10         195k          5/10          6.4       0.00          0.72
    #   10 / 2.5         7/10         358k          2/10         37.2       0.66          1.19
    #
    # - The lateral noise did drop as designed. The manager's goal_y
    #   sampling std at visited states went from a median of 1.47 m to
    #   0.21 m. Contacts rose anyway, in training and in the final policies
    #   (283 against 27 on the grid, from gate 1's entry at x ~ 4 to past
    #   gate 2 at x ~ 9.3).
    # - So round 1's attribution of the drift to risk 1 ("That is risk 1 at
    #   work", above) is not supported. A sevenfold drop in lateral noise
    #   made precision worse, not better. What is left of the list is risk 2,
    #   a manager that has to do all the fine lateral control at one
    #   decision per second through a worker that cannot see the walls.
    #   That is untested here.
    # - The separation is intact in every round-3 policy. Replacing the
    #   manager with a forward goal costs ~2600 of return and mirroring its
    #   goal ~10100, and the goal explains 94% of the worker's action
    #   variance.
    #
    # Round 1 (both bounds 10 m) stays the best blind configuration measured.
    #
    # Cadence (2026-09-28): round 1's setting (blind worker at 0, 10 m box)
    # with --manager-freq 5, two decisions a second; 10 seeds, 1M steps,
    # scenarios/slalom/studies/blind_worker_c5. Not a clean single-variable
    # test: the manager's discount moves with the cadence (gamma**c). It is
    # also worse than round 1:
    #
    #   --manager-freq  ever solved  first solve  solved at end  grid gap  stuck at gate 2
    #   10                 9/10         195k          5/10          6.4        1/10
    #   5                  6/10         118k          2/10         50.4        4/10
    #
    # - Twice the decisions did not buy precision. The seeds that solve pass
    #   only 21% of the solved-checks after their first solve (51% at 10),
    #   and they end 0.6-57 of return behind the oracle.
    # - It made the manager's local optimum four times as common. Seeds 5-8
    #   never reach the line in any evaluation, and end at x ~ 6.5-7.2,
    #   at gate 2, with no contacts. That is round 1's seed 10 and round 2's
    #   failure again, not a stall at the line.
    # - So neither named suspect, lateral noise (round 3) or cadence (this
    #   run), explains the blind hierarchy's late drift. That stays open.
    #
    # Where this leaves the strict separation: --worker-obs velocity
    # --worker-extrinsic-coef 0, with every other flag at its default
    # (round 1). It holds as a hierarchy: the manager is indispensable, and no
    # seed stalls at the line without any extrinsic reward. Against the
    # sighted default it costs:
    #
    # - ~2.7x the samples to a first solve (median 195k against 72k);
    # - 5/10 seeds against 10/10 still passing the strict solved-check at the
    #   end, the rest drifting 2-9 steps behind the oracle;
    # - 1/10 seeds stuck in front of gate 2.
    #
    # The default stays the sighted worker at 0.02.
    parser.add_argument("--worker-obs", type=str, default="full", choices=sorted(WORKER_OBS),
        help="what the worker observes next to its goal: 'full' (the whole observation, "
             "[p_x, p_y, v_x, v_y]) or 'velocity' ([v_x, v_y] only, blind to where it is, "
             "so it can learn to reach goals but not the task). See the comment above "
             "this flag. Saved in the checkpoint; load_agent reads it back")
    parser.add_argument("--num-envs", type=int, default=8,
        help=f"the number of parallel {scenario.name} environments to collect "
             "rollouts from. Rollouts come from a batched vector env; each "
             "environment carries its own goal, manager cadence and segment "
             "accumulator")
    parser.add_argument("--num-steps-worker", type=int, default=2048,
        help="the number of worker (= environment) steps per rollout, summed "
             "over all --num-envs environments. Unchanged in meaning by "
             "vectorization -- it is still the worker's batch size, so 2048 "
             "keeps the minibatch at 256 and the update count where they were; "
             "what changed is that the 2048 now come from --num-envs parallel "
             "streams of 2048/--num-envs steps instead of one serial stream. "
             "Must be divisible by --num-envs. A total over environments, as "
             "in the PPO+MPC scripts' flag of the same name -- where flat "
             "PPO's --num-steps counts steps *per* environment (its 1024-step "
             "batch is --num-steps 128 x 8 envs). The name is kept for the "
             "hierarchical family's sake rather than renamed to flat PPO's. "
             "Its 1024-step batch was measured here too (2026-09-24, 13 "
             "slalom seeds, full 500k budget, every early stop disabled, "
             "clip_vloss still on): --num-steps-worker 1024 with "
             "--num-minibatches 4, --num-minibatches-manager 2 and "
             "--eval-freq 10, so the minibatch sizes and the evaluation "
             "cadence in steps stay put, solved no faster (median / mean "
             "82k / 91k steps against 92k / 97k, p = 0.29) and passed fewer "
             "solved-checks afterwards (80%% against 91%%, p = 0.07; 11/13 "
             "solved at the end). Flat PPO on this 2048-step setting did worse "
             "too, so the two keep different rollouts on purpose")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the environment-step discount factor. 0.99, not the 0.999 this "
             "script used to train at: see the thesis PPO chapter, 11.1. The "
             "manager is discounted at gamma**manager_freq")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches the *worker's* rollout is split "
             "into. 8 against 2048 holds the minibatch at 256, as flat PPO "
             "does; the previous 32 gave minibatches of 64")
    parser.add_argument("--num-minibatches-manager", type=int, default=4,
        help="the number of mini-batches the *manager's* rollout is split "
             "into, near-equal whatever that rollout's manager batch size "
             "turns out to be. Separate from --num-minibatches because the "
             "manager's buffer is roughly manager_freq times smaller: "
             "splitting it 32 ways left ~6 samples per gradient step")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update each policy")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef-manager", type=float, default=0.01,
        help="coefficient of the manager's entropy bonus, fixed for the whole "
             "run (see HPPOAgent for why there is no autotuning)")
    parser.add_argument("--ent-coef-worker", type=float, default=0.01,
        help="coefficient of the worker's entropy bonus, fixed for the whole run")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--eval-freq", type=int, default=5,
        help="evaluate the agent every eval_freq updates. 5 rather than 1 "
             "because an evaluation of --eval-episodes episodes costs up to "
             "eval_episodes*max_steps environment steps, which at every "
             "update would dominate the wall clock over the 2048 steps of "
             "training it is measuring")
    parser.add_argument("--eval-episodes", type=int, default=20,
        help="number of episodes to evaluate the agent")
    parser.add_argument("--solved-early-stop", type=_str2bool, default=True,
        help="if toggled, actually stop training once the solved criterion below "
             "is met. The check, its logging, and the one-time solved.pt checkpoint "
             "still happen either way -- this only gates the early `break`, so runs "
             "meant to be plotted against each other on the same x-axis (fixed "
             "--total-timesteps for every seed) can set this false and still see "
             "where each seed crossed the threshold. Independent of "
             "--early-stop-success-rate below (off by default) -- either can "
             "stop training")
    parser.add_argument("--solved-tolerance", type=float, default=5.0,
        help="stop training once the agent's return is within this many reward "
             "units of the oracle's optimal return (see algorithms/optimal_solver.py) "
             "at every point of the fixed evaluation grid below")
    parser.add_argument("--solved-grid-nx", type=int, default=5,
        help="number of p_x0 points in the fixed grid the solved-check evaluates from")
    parser.add_argument("--solved-grid-ny", type=int, default=5,
        help="number of p_y0 points in the fixed grid the solved-check evaluates from")
    parser.add_argument("--solved-consecutive", type=int, default=2,
        help="require the solved criterion to hold for this many consecutive "
             "eval passes in a row before actually stopping training -- guards "
             "against stopping on a momentary crossing while the policy is "
             "still moving")
    parser.add_argument("--early-stop-success-rate", type=float, default=None,
        help="stop training once eval/success_rate has been >= this AND "
             "eval/episodic_return >= --early-stop-optimal-frac, both for "
             "--early-stop-patience consecutive evaluations. Off by default "
             "(None) since 2026-09-24; it used to default to 1.0, for this "
             "reason: on the slalom task, seed 1's manager reliably reaches 100%% "
             "eval success by ~1/5 of a 500k-step run and then, under "
             "continued training, its goal distribution keeps sharpening "
             "past that point -- one goal dimension's Beta collapses toward "
             "a state-insensitive skew, the other's state-dependent sign "
             "flip (needed to alternate between the slalom's two "
             "oppositely-offset gates) erodes, and eval success falls back "
             "to 0%% within a few evaluations and never recovers for the "
             "rest of the budget. This has no in-run recovery mechanism, so "
             "rather than fight it, stop as soon as the policy has "
             "demonstrably solved the task. On the tunnel the manager can "
             "likewise reach 100%% eval success while still threading the "
             "corridor with avoidable wall contacts, well short of the return "
             "--solved-early-stop's tight tolerance needs. That collapse was "
             "later traced to the worker's reward and fixed (see the "
             "termination-avoidance block above), and a default of 1.0 then "
             "only meant that every default run stopped at ~95%% of the "
             "oracle, under a looser rule than flat PPO, which stops on "
             "--solved-early-stop alone -- the benchmark protocol "
             "(docs/benchmark.md) had to pass 2.0 to disable it. Pass 1.0 "
             "to restore the old rule; any value > 1.0 also leaves it off")
    parser.add_argument("--early-stop-optimal-frac", type=float, default=0.95,
        help="the eval/episodic_return floor --early-stop-success-rate also "
             "requires, as a fraction of the solved-check oracle's mean "
             "optimal return (see the 'Solved-check: precomputed optimal "
             "returns' line printed at startup) rather than a fixed reward "
             "value. success_rate alone is not enough: the manager reaches "
             "100%% success as early as ~40k steps by smashing through both "
             "gates rather than threading them -- the env clamps a wall "
             "contact and continues rather than ending the episode (see "
             "envs/config.py's contact_penalty comment), so reaching the "
             "goal line at all does not mean reaching it cleanly. A prior "
             "version of this flag was a fixed absolute value (150.0) "
             "picked off one seed's trajectory at one goal_reward/"
             "contact_penalty balance; it went stale the moment that "
             "balance changed -- a seed-1-10 sweep at a 5x larger "
             "goal_reward found it let a policy with ~16 collisions/episode "
             "(return=215, far from the ~1013 mean optimum) trigger early "
             "stop. Deriving the floor from the oracle's own optimal return "
             "keeps it meaningful across reward-scale changes")
    parser.add_argument("--early-stop-patience", type=int, default=3,
        help="consecutive qualifying evaluations required before "
             "--early-stop-success-rate stops training; see its help")
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error(f"--num-envs must be > 0, got {args.num_envs}")
    if args.num_steps_worker % args.num_envs != 0:
        parser.error(
            f"--num-steps-worker ({args.num_steps_worker}) must be divisible by "
            f"--num-envs ({args.num_envs}); otherwise the worker's batch is not "
            f"the step budget it names")
    args.num_steps_per_env = args.num_steps_worker // args.num_envs
    return args


def worker_input(agent, obs_norm, goal_phys):
    """The worker's network input: the normalized observation columns it
    sees (all of them, or only the velocity -- see --worker-obs) ++ the
    normalized goal.

    Concatenates on the last axis, so it takes either a single
    (obs_dim,)/(goal_dim,) pair -- what evaluation has -- or the
    (num_envs, obs_dim)/(num_envs, goal_dim) batch the rollout carries.
    """
    return torch.tensor(
        np.concatenate([agent.worker_obs_view(obs_norm), agent.normalize_goal(goal_phys)], axis=-1),
        dtype=torch.float32, device=agent.device
    )


def make_policy_fn(agent, goal_trace=None, replan_goal=None):
    """The controller evaluation scores, as a fresh closure per episode:
    one raw observation in, the worker's deterministic action out. The
    manager issues a deterministic goal at the episode's first step and
    every manager_freq steps after it, and the goal decays by the agent's
    displacement in between -- the rollout's cadence and goal transition
    exactly, without the sampling. The random-start evaluation, the
    solved-check and the study scripts (algorithms/study.py) all run exactly
    this, so they cannot drift apart. Flat PPO's counterpart is memoryless
    (algorithms/ppo/ppo_train.py).

    A *factory*, not a single shared closure: the re-plan cadence
    (state["goal"]/state["step_in_c"]) is per-episode state, so every
    caller builds a fresh policy_fn right after each env.reset() -- see
    algorithms/solved_check.py.

    `goal_trace`, if given, is a list the policy appends one
    `(replanned, goal)` pair to per step: whether the manager issued a fresh
    goal on that step, and the physical goal displacement the worker acted
    on. Only the study scripts' figures read it; it changes nothing the
    controller does.

    `replan_goal`, if given, maps the manager's physical goal to the one the
    worker is handed at each re-plan (it then decays as usual), so the
    manager can be knocked out -- replaced by a fixed goal, or overruled --
    while the worker runs under exactly this controller. Only the manager
    knock-out probes of scenarios/slalom/scripts/sweep_extrinsic_coef.py
    pass it; evaluation and the solved-check never do.

    The evaluation loop used to carry its own inline copy of this
    controller, a second implementation to keep in step with this one
    and with the rollout. That copy had, earlier still, reused the names
    the rollout carries its in-flight segment in (`current_goal`,
    `current_pos`, `manager_obs_tensor`), so each evaluation silently
    overwrote the training state with whatever its last episode ended on
    and the rollout resumed against a goal from a different episode.
    Vectorizing turned that into a shape error, which is how it was
    found; it was a live bug in the single-environment version too. With
    the controller's state inside this closure it cannot recur. It is at
    module level, rather than a closure inside `train`, so that a
    hierarchy reloaded from a checkpoint (`load_agent`) is driven by this
    same code instead of a third copy. Its cadence is the agent's own
    `manager_freq`, which `train` constructs from --manager-freq and every
    checkpoint carries.
    """
    state = {}

    def policy_fn(obs):
        obs_norm = agent.normalize_obs(obs)
        pos = np.array([obs[0], obs[1]])
        if state:
            state["step_in_c"] += 1
        with torch.no_grad():
            replanned = not state or state["step_in_c"] == agent.manager_freq
            if replanned:
                # First step of the episode or of a new segment.
                manager_obs = torch.tensor(obs_norm, dtype=torch.float32, device=agent.device).unsqueeze(0)
                state["goal"] = agent.scale_goal(agent.manager_act(manager_obs).cpu().numpy()[0])
                if replan_goal is not None:
                    state["goal"] = np.asarray(replan_goal(state["goal"]), dtype=np.float32)
                state["step_in_c"] = 0
            else:
                state["goal"] = state["goal"] - (pos - state["pos"])
            state["pos"] = pos
            action = agent.worker_act(worker_input(agent, obs_norm, state["goal"]).unsqueeze(0))
        if goal_trace is not None:
            goal_trace.append((replanned, np.array(state["goal"], dtype=float)))
        return action.cpu().numpy()[0]

    return policy_fn


def load_agent(path, env, device="cpu"):
    """A trained hierarchy rebuilt from the checkpoint at `path`, for
    evaluation.

    `env` is an instance of the environment it was trained on: its spaces give
    the network sizes and the worker's action limits, the same way `train`
    reads them. The observation and goal maps and the manager's cadence
    travel in the checkpoint itself (see HPPOAgent.save), and so does what the
    worker observes, which is read first because it sizes the worker's
    networks."""
    agent = HPPOAgent(
        obs_dim=env.observation_space.shape[0],
        goal_dim=2,  # delta x, delta y, as in `train`
        act_dim=env.action_space.shape[0],
        worker_act_limit_low=float(env.action_space.low[0]),
        worker_act_limit_high=float(env.action_space.high[0]),
        worker_obs=checkpoint_worker_obs(path),
        device=device,
    )
    agent.load(path)
    for net in (agent.manager_actor, agent.manager_critic, agent.worker_actor, agent.worker_critic):
        net.eval()
    return agent


def main(scenario):
    train(parse_args(scenario), scenario)


def train(args, scenario):
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    env_config = scenario.make_env_config(
        contact_penalty=args.contact_penalty,
        noise_bound_p=args.noise_bound_p,
        noise_bound_v=args.noise_bound_v,
        **({} if args.env_u_max is None else {"u_max": args.env_u_max}),
    )

    if args.env_u_max is not None:
        _T = args.manager_freq * env_config.dt
        print(f"REGIME STUDY: env u_max overridden to {env_config.u_max} "
              f"(canonical {scenario.make_env_config().u_max}); agility ratio rho = "
              f"v_max/(u_max*T) = {env_config.v_max / (env_config.u_max * _T):.2f}")

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
        # save_code only captures the entry script, which is now the thin
        # scenario wrapper; log the code that actually trains alongside it.
        wandb.run.log_code(
            root=os.path.dirname(os.path.abspath(__file__)),
            include_fn=lambda path: os.path.basename(path) in ("hppo.py", "hppo_train.py"),
        )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Env setup: a batched, auto-resetting vector env for rollout collection,
    # a plain env for evaluation -- the same split flat PPO uses. Evaluation
    # runs one seeded episode at a time and has no throughput problem to solve.
    #
    # No wrappers on either. `RecordEpisodeStatistics` is replaced by counting
    # returns and lengths in the loop, which also gets the success and
    # collision flags out of `info` -- the two metrics that distinguish a
    # stalled run from a converged one, and which the wrapper does not carry.
    #
    # `NormalizeReward` is gone for a substantive reason, not tidiness. It
    # scaled the reward stream by a running estimate of the return's standard
    # deviation to keep the critic's regression target O(1); the agent now
    # standardises that target directly (hppo.py, RunningMeanStd), which
    # achieves the same thing without altering the rewards GAE sees, without a
    # wrapper statistic that has to be checkpointed to reload a policy, and --
    # unlike the wrapper, which normalises the *environment* reward only --
    # symmetrically for the worker's intrinsic reward too. Runs from before
    # this change saw a different reward stream and are not directly
    # comparable. (This used to be the docstring of a one-line `make_env`
    # helper, inlined for parity with flat PPO's loop.)
    vec_env = scenario.vec_env_cls(num_envs=args.num_envs, config=env_config)
    eval_env = scenario.env_cls(config=env_config)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    goal_dim = 2 # delta x, delta y
    max_goal_bound = args.max_goal_bound
    obs_low = eval_env.observation_space.low
    obs_high = eval_env.observation_space.high

    # Solved-check setup: a fixed grid of initial conditions (not random
    # draws), so the oracle's optimal return for each point is solved once
    # here and reused on every later eval pass instead of being recomputed.
    # Any MinTimeSolver infeasibility raises here, at startup, rather than
    # mid-training.
    # The oracle replays open-loop actions through an env, which a
    # disturbance would knock off course, so it gets an undisturbed copy of
    # the environment (since 2026-09-28; the same one as before whenever
    # --noise-bound-p/-v are 0). Under a disturbance its returns are an upper
    # reference; evaluation and the solved-check still run on the disturbed
    # eval_env, reseeded per episode so every pass sees the same disturbances.
    oracle_env = scenario.env_cls(config=replace(env_config, noise_bound_p=0.0, noise_bound_v=0.0))
    solved_grid = spawn_grid(oracle_env, args.solved_grid_nx, args.solved_grid_ny)
    optimal_grid = precompute_optimal_grid(oracle_env, solved_grid, solver=MinTimeSolver())
    optimal_returns = np.array([r.total_return for _, r in optimal_grid])
    print(f"Solved-check: precomputed optimal returns for {len(optimal_grid)} fixed initial "
          f"conditions (mean={optimal_returns.mean():.1f}, min={optimal_returns.min():.1f}, "
          f"max={optimal_returns.max():.1f}); solved-tolerance={args.solved_tolerance}")
    # See --early-stop-optimal-frac: derived from the oracle's mean optimal
    # return rather than a fixed value, so it stays meaningful whatever
    # goal_reward/contact_penalty balance this run is training against.
    early_stop_min_return = args.early_stop_optimal_frac * float(optimal_returns.mean())
    if args.early_stop_success_rate is not None:
        print(f"Early-stop return floor: {early_stop_min_return:.1f} "
              f"({args.early_stop_optimal_frac:.0%} of mean optimal)")
    consecutive_solved = 0
    solved_checkpoint_saved = False

    # Agent and Buffer
    agent = HPPOAgent(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        act_dim=act_dim,
        worker_act_limit_low=float(eval_env.action_space.low[0]),
        worker_act_limit_high=float(eval_env.action_space.high[0]),
        lr_manager=args.learning_rate_manager,
        lr_worker=args.learning_rate_worker,
        # One environment-step discount. The manager's gamma**c is derived
        # inside the agent from manager_freq, so the discount used for its GAE
        # cannot drift from the cadence the rollout actually runs at.
        gamma=args.gamma,
        manager_freq=args.manager_freq,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef_manager=args.ent_coef_manager,
        ent_coef_worker=args.ent_coef_worker,
        max_grad_norm=args.max_grad_norm,
        critic_lr_mult=args.critic_lr_mult,
        # Carried into the checkpoint so a reloaded hierarchy knows the
        # observation and goal maps it was trained under.
        obs_low=obs_low,
        obs_high=obs_high,
        max_goal_bound=max_goal_bound,
        max_goal_bound_y=args.max_goal_bound_y,
        worker_obs=args.worker_obs,
        device=device,
    )

    # One base learning rate per parameter group (actor, then critic) for each
    # head. The anneal below scales each group against its own base; scaling
    # every group by the head's rate would silently reset its critic to the
    # actor's rate on the first update.
    base_lrs_manager = [group["lr"] for group in agent.manager_optimizer.param_groups]
    base_lrs_worker = [group["lr"] for group in agent.worker_optimizer.param_groups]

    # The worker acts every step in every environment, so its rollout is a
    # full (num_steps_per_env, num_envs) grid.
    worker_buffer = VecRolloutBuffer(
        args.num_steps_per_env, args.num_envs, agent.worker_input_dim, act_dim, device)
    # The manager writes one transition per c worker steps *per environment*,
    # so each column needs only ceil(num_steps_per_env / c) slots -- plus a
    # margin, because a segment also ends early whenever an episode does. Sized
    # for the worst case (an episode ending on every step) rather than reasoned
    # about, since the buffer is a few thousand floats either way.
    manager_buffer = ManagerVecRolloutBuffer(
        args.num_steps_per_env, args.num_envs, obs_dim, goal_dim, device)

    checkpoint_dir = os.path.join(scenario.script_dir, args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eval_return = -float("inf")

    # Initialize environments. Every quantity the serial rollout held as a
    # scalar is now a per-environment array: the environments run independent
    # episodes, so they sit at different points in their manager cadence, carry
    # different partially-consumed goals, and accumulate different segment
    # rewards.
    obs, _ = vec_env.reset(seed=args.seed)
    obs_norm = agent.normalize_obs(obs)
    current_pos = obs[:, :2].copy()

    worker_step_in_c = np.zeros(args.num_envs, dtype=np.int64)
    accumulated_env_reward = np.zeros(args.num_envs, dtype=np.float32)

    # Initial manager action, for every environment at once.
    manager_obs_norm = obs_norm.copy()
    with torch.no_grad():
        manager_action, manager_logprob, manager_value = agent.get_manager_action_and_value(
            torch.tensor(manager_obs_norm, dtype=torch.float32, device=device)
        )
    current_goal = agent.scale_goal(manager_action.cpu().numpy())

    global_step = 0
    start_time = time.time()

    num_updates = args.total_timesteps // args.num_steps_worker
    actual_timesteps = num_updates * args.num_steps_worker
    # Informational only: the update takes the minibatch *count*.
    worker_minibatch_size = args.num_steps_worker // args.num_minibatches
    print(f"num_envs={args.num_envs} num_steps_per_env={args.num_steps_per_env} "
          f"worker_batch_size={args.num_steps_worker} "
          f"worker_minibatch_size={worker_minibatch_size} "
          f"num_updates={num_updates} timesteps={actual_timesteps}"
          + (f" (requested {args.total_timesteps}; {args.total_timesteps - actual_timesteps} "
             f"dropped by integer division)" if actual_timesteps != args.total_timesteps else ""))
    if args.track:
        wandb.config.update({"num_updates": num_updates, "actual_timesteps": actual_timesteps,
                             "worker_minibatch_size": worker_minibatch_size,
                             "num_steps_per_env": args.num_steps_per_env})

    # A local copy of every W&B log call, written whether or not --track is
    # on; see algorithms/metrics_log.py.
    metrics_log = MetricsLog(checkpoint_dir, config={
        "scenario": scenario.name, "algorithm": "hppo", "run_name": run_name,
        "num_updates": num_updates, "actual_timesteps": actual_timesteps, **vars(args)})

    def log(metrics, step):
        metrics_log.log(metrics, step)
        if args.track:
            wandb.log(metrics, step=step)

    ep_reward = np.zeros(args.num_envs, dtype=np.float64)
    ep_length = np.zeros(args.num_envs, dtype=np.int64)

    # See --early-stop-success-rate: counts consecutive qualifying
    # evaluations, reset by any evaluation that falls short.
    consecutive_high_success = 0

    for update in range(1, num_updates + 1):
        # Linear LR decay: once the policies have converged a constant LR keeps
        # injecting noise into all four heads off a shrinking advantage signal.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for group, base_lr in zip(agent.manager_optimizer.param_groups, base_lrs_manager):
                group["lr"] = frac * base_lr
            for group, base_lr in zip(agent.worker_optimizer.param_groups, base_lrs_worker):
                group["lr"] = frac * base_lr

        completed_returns = []
        completed_lengths = []
        completed_successes = []
        completed_collision_counts = []
        completed_collision_impacts = []
        # Direct instrumentation of the termination-avoidance mechanism (see
        # the "worker termination-avoidance" block in parse_args): what the
        # worker actually does in the strip of states immediately before the
        # goal line. A healthy worker keeps
        # a_x positive there and crosses; the collapsed one brakes with
        # a_x approx -2 and orbits. Logged per update so the onset is visible
        # in the same plot as the eval success rate it precedes by ~30 updates.
        near_goal_ax = []
        near_goal_value = []
        # What the worker's reward is made of: the intrinsic goal-closing
        # term against the extrinsic mix (--worker-extrinsic-coef times the
        # environment's reward), summed over every worker step of the rollout
        # and, for the share, in absolute value. Logged so that whether the
        # extrinsic term swamps the intrinsic one -- the worst case for the
        # hierarchy being a worker that no longer needs its manager -- is a
        # curve rather than a guess. The intrinsic mean is also the worker's
        # goal progress per step (about v_max*dt = 0.12 m at most; the speed
        # limit is per axis, so diagonal motion can exceed it slightly, up to
        # ~0.126 measured): it falling while the return holds would mean the
        # worker has stopped following goals.
        # Read-only, like the near-goal diagnostic.
        reward_intrinsic_sum = reward_intrinsic_abs = 0.0
        reward_extrinsic_sum = reward_extrinsic_abs = 0.0

        for step in range(0, args.num_steps_per_env):
            global_step += args.num_envs

            # Worker action selection, one batched forward pass over all envs
            obs_goal_tensor = worker_input(agent, obs_norm, current_goal)
            with torch.no_grad():
                worker_action, worker_logprob, worker_value = agent.get_worker_action_and_value(
                    obs_goal_tensor
                )

            action_np = worker_action.cpu().numpy()

            # See near_goal_ax above. `obs` is the pre-step observation, so
            # this is "the action taken from a state within 1 m of the goal
            # line", which is exactly the decision the pathology corrupts.
            near_goal_mask = obs[:, 0] >= (env_config.tunnel_length - 1.0)
            if np.any(near_goal_mask):
                near_goal_ax.append(action_np[near_goal_mask, 0])
                near_goal_value.append(worker_value.cpu().numpy()[near_goal_mask])

            # Environment step (the vector env auto-resets envs that finish)
            next_obs, env_reward, terminated, truncated, info = vec_env.step(action_np)
            done = terminated | truncated
            next_obs_norm = agent.normalize_obs(next_obs)

            # The auto-reset is why this is not a mechanical substitution. For
            # a finished env `next_obs` is already the *next* episode's first
            # observation, so every quantity describing the transition that
            # just happened -- the goal decrement, the worker's progress
            # reward, both truncation bootstraps -- has to read the pre-reset
            # state instead. Taking it from `next_obs` would score the worker
            # on a teleport back to the entrance.
            term_obs = next_obs.copy()
            if np.any(done):
                term_obs[done] = info["final_observation"][done]
            term_obs_norm = agent.normalize_obs(term_obs)

            ep_reward += env_reward
            ep_length += 1

            accumulated_env_reward += (agent.gamma ** worker_step_in_c) * env_reward

            # Update goals manually, against the pre-reset position
            term_pos = term_obs[:, :2]
            next_goal = current_goal - (term_pos - current_pos)

            worker_reward = (np.linalg.norm(current_goal, axis=-1)
                             - np.linalg.norm(next_goal, axis=-1))

            # See reward_intrinsic_sum above. Read here, before either term
            # below is added: with the coefficient at 0, `worker_reward` is
            # still this very array when the truncation bootstrap adds into
            # it in place.
            extrinsic_reward = args.worker_extrinsic_coef * env_reward
            reward_intrinsic_sum += float(worker_reward.sum())
            reward_intrinsic_abs += float(np.abs(worker_reward).sum())
            reward_extrinsic_sum += float(extrinsic_reward.sum())
            reward_extrinsic_abs += float(np.abs(extrinsic_reward).sum())

            # Put the environment's own outcome back into the worker's return
            # -- see --worker-extrinsic-coef for the termination-avoidance
            # pathology this exists to remove. Applied here, before the
            # truncation bootstrap below, so the
            # bootstrapped value and the reward it is folded into are on the
            # same (mixed) scale. (The removed --worker-success-bonus was
            # applied right after it; see the comment above that flag's
            # former place in parse_args.)
            if args.worker_extrinsic_coef != 0.0:
                worker_reward = worker_reward + extrinsic_reward

            # Handle truncation bootstrapping for the worker. The episode did
            # not end, the clock did: fold the value of the state it was cut at
            # into the reward, then mark the transition done so GAE stops
            # there. Discounts come from the agent, which is also what ran GAE.
            #
            # The value is taken under the decayed `next_goal`. That is the
            # goal the worker would have carried on with, except when the cut
            # falls on a segment's last step: there the continuation would
            # have started under a fresh manager goal instead, which is not
            # sampled just for a bootstrap. An approximation on ~1 in
            # manager_freq truncations, and only while episodes still run
            # into max_steps at all.
            trunc_only = truncated & ~terminated
            if np.any(trunc_only):
                with torch.no_grad():
                    true_next_worker_value = agent.get_worker_value(
                        worker_input(agent, term_obs_norm[trunc_only], next_goal[trunc_only])
                    )
                worker_reward[trunc_only] += (
                    agent.gamma_worker * true_next_worker_value.cpu().numpy())

            # Store worker transitions (all envs, every step). `done` stops GAE
            # at a truncation too, as in flat PPO: the continuation value is
            # already in the reward. (This used to go through a `worker_done`
            # alias, "`terminated`, or forced True where truncated" -- which,
            # the vector env making the two exclusive, is exactly `done`.)
            worker_buffer.add(
                obs_goal_tensor,
                worker_action,
                worker_logprob,
                worker_reward,
                worker_value,
                done.astype(np.float32)
            )

            worker_step_in_c += 1

            # Which managers act now? An environment's segment ends after c
            # worker steps or when its episode does, and the environments do
            # not agree on when. Those are the only two boundaries, so every
            # done=False boundary is exactly manager_freq steps long -- which
            # is what the manager's fixed gamma**manager_freq GAE recursion
            # assumes.
            #
            # A wall contact used to be a third boundary (--replan-on-collision,
            # on by default): the manager re-planned the instant the worker
            # hit a wall, on the grounds that a hit can clamp position and
            # zero velocity in a way that leaves the in-flight goal
            # unreachable for the rest of the segment. It was removed because
            # (1) it was a train/eval mismatch -- evaluation and the
            # solved-check only ever re-planned on the manager_freq boundary,
            # so the policy was trained under a controller it was never
            # evaluated with; (2) it mattered while collision rates were 8-24
            # per episode and is close to moot at the 0-1 the worker-reward fix
            # brought (docs/worker-termination-avoidance.md, open item 1); and
            # (3) its segments needed a pseudo-done plus a continuation value
            # folded into the reward, and when such a segment was the last one
            # a column stored in a rollout, the GAE bootstrap (next_done taken
            # from the real `done`, not the pseudo-done) added that
            # continuation a second time.
            manager_act_now = (worker_step_in_c == args.manager_freq) | done
            if np.any(manager_act_now):
                # Handle truncation bootstrapping for the manager. Its segment
                # is worker_step_in_c environment steps long, so the cut-off
                # continuation is discounted by gamma**that -- not by the
                # manager's own gamma**c, which assumes a full segment. That
                # exponent is now per environment too.
                manager_reward = accumulated_env_reward.copy()
                manager_trunc = manager_act_now & trunc_only
                if np.any(manager_trunc):
                    with torch.no_grad():
                        true_next_manager_value = agent.get_manager_value(
                            torch.tensor(term_obs_norm[manager_trunc],
                                         dtype=torch.float32, device=device)
                        )
                    manager_reward[manager_trunc] += (
                        (agent.gamma ** worker_step_in_c[manager_trunc])
                        * true_next_manager_value.cpu().numpy())

                manager_buffer.add(
                    manager_act_now,
                    manager_obs_norm,
                    manager_action,
                    manager_logprob,
                    manager_reward,
                    manager_value,
                    done.astype(np.float32)
                )

                worker_step_in_c[manager_act_now] = 0
                accumulated_env_reward[manager_act_now] = 0.0

                # A fresh goal for every environment whose segment just ended.
                # The serial loop split this in two -- re-plan at the next
                # observation, or re-plan at the reset observation after an
                # episode ended -- but under an auto-resetting vector env both
                # are `next_obs`, since a done environment has already been
                # reset into the state the new segment starts from. Note that
                # `done` implies `manager_act_now`, so no environment is left
                # holding a goal from a finished episode.
                sel = manager_act_now
                sel_t = torch.as_tensor(sel, device=device)
                with torch.no_grad():
                    new_action, new_logprob, new_value = agent.get_manager_action_and_value(
                        torch.tensor(next_obs_norm[sel], dtype=torch.float32, device=device)
                    )
                manager_obs_norm[sel] = next_obs_norm[sel]
                manager_action[sel_t] = new_action
                manager_logprob[sel_t] = new_logprob
                manager_value[sel_t] = new_value
                current_goal[sel] = agent.scale_goal(new_action.cpu().numpy())

            for i in np.flatnonzero(done):
                completed_returns.append(ep_reward[i])
                completed_lengths.append(ep_length[i])
                completed_successes.append(bool(info["final_info"]["is_success"][i]))
                completed_collision_counts.append(int(info["final_info"]["collision_count"][i]))
                completed_collision_impacts.extend(info["final_info"]["collision_impacts"][i])
                ep_reward[i] = 0.0
                ep_length[i] = 0

            # Environments still mid-segment carry their partially consumed
            # goal forward; the rest were just given a fresh one above.
            carry = ~manager_act_now
            current_goal[carry] = next_goal[carry]

            obs = next_obs
            obs_norm = next_obs_norm
            # Post-auto-reset for finished envs, pre-reset terminal for the
            # rest -- which is what the next step's goal decrement needs.
            current_pos = next_obs[:, :2].copy()

        # Bootstrap values for the states after the rollout. GAE bootstraps
        # from them only where a column's last stored transition did not end
        # its episode -- each buffer reads that from its own last done flag.
        with torch.no_grad():
            # Worker bootstrap, per environment
            next_worker_value = agent.get_worker_value(worker_input(agent, obs_norm, current_goal))
            agent.compute_worker_returns_and_advantage(worker_buffer, next_worker_value)

            # Manager bootstrap. `manager_value` is the value of each
            # environment's segment currently in flight, whose start is exactly
            # the state that environment's last stored manager transition led
            # to -- so it is already the per-column bootstrap the ragged GAE
            # wants.
            #
            # That in-flight segment is stored only when it ends, early in the
            # next rollout, so it straddles the update below: its goal,
            # log-probability and value all come from the pre-update networks.
            # The log-probability is the one to keep -- it is the behaviour
            # policy's, which is what the importance ratio needs; the ratio
            # just does not start at exactly 1 for that transition. The stale
            # value biases nothing: the transition is its column's first in
            # that rollout, and GAE's return target for step t does not depend
            # on V(s_t), so it acts only as that one transition's baseline.
            # One transition per environment, 8 of the manager's ~215 per
            # update at the defaults. Flat PPO and the worker have no such
            # straddle: each of their transitions is a single step.
            agent.compute_manager_returns_and_advantage(manager_buffer, manager_value)

        # Optimize the policies. Each head's batch is split into a fixed
        # *number* of minibatches, not a fixed size: the manager's batch size
        # varies from rollout to rollout (see HPPOAgent._update_head).
        worker_metrics = agent.update_worker(worker_buffer, args.num_minibatches, args.update_epochs)
        manager_metrics = agent.update_manager(manager_buffer, args.num_minibatches_manager, args.update_epochs)

        # Logging
        sps = int(global_step / (time.time() - start_time))
        metrics = {**worker_metrics, **manager_metrics}
        # Group 0 is each actor, group 1 its critic (see HPPOAgent.__init__).
        metrics["charts/manager_lr"] = agent.manager_optimizer.param_groups[0]["lr"]
        metrics["charts/manager_critic_lr"] = agent.manager_optimizer.param_groups[-1]["lr"]
        metrics["charts/worker_lr"] = agent.worker_optimizer.param_groups[0]["lr"]
        metrics["charts/worker_critic_lr"] = agent.worker_optimizer.param_groups[-1]["lr"]
        metrics["charts/SPS"] = sps
        metrics["charts/num_episodes"] = len(completed_returns)
        if near_goal_ax:
            metrics["charts/worker_ax_near_goal"] = float(np.mean(np.concatenate(near_goal_ax)))
            metrics["charts/worker_value_near_goal"] = float(np.mean(np.concatenate(near_goal_value)))
            metrics["charts/near_goal_samples"] = float(sum(a.size for a in near_goal_ax))
        metrics["worker/reward_intrinsic"] = reward_intrinsic_sum / args.num_steps_worker
        metrics["worker/reward_extrinsic"] = reward_extrinsic_sum / args.num_steps_worker
        metrics["worker/extrinsic_share"] = (
            reward_extrinsic_abs / max(reward_intrinsic_abs + reward_extrinsic_abs, 1e-12))

        log_line = (f"update={update} global_step={global_step} SPS={sps} "
                    f"w_ev={metrics['worker/explained_variance']:.3f} "
                    f"m_ev={metrics['manager/explained_variance']:.3f} "
                    f"m_v_bias={metrics['manager/value_bias']:.1f} "
                    # Manager-collapse diagnostics (see ManagerActor's
                    # docstring in algorithms/hppo/hppo.py): printed inline,
                    # not just logged to W&B, so the update where these spike
                    # or crater is visible without leaving the console.
                    f"m_adv_std={metrics['manager/adv_std_raw']:.4f} "
                    f"m_kl_max={metrics['manager/approx_kl_max']:.4f} "
                    f"m_ratio_max={metrics['manager/ratio_max_dev']:.3f}")
        if "charts/worker_ax_near_goal" in metrics:
            log_line += (f" w_ax@goal={metrics['charts/worker_ax_near_goal']:+.2f} "
                         f"w_v@goal={metrics['charts/worker_value_near_goal']:.2f}")
        if completed_returns:
            mean_return = float(np.mean(completed_returns))
            mean_length = float(np.mean(completed_lengths))
            success_rate = float(np.mean(completed_successes))
            collision_count_mean = float(np.mean(completed_collision_counts))
            metrics["charts/episodic_return"] = mean_return
            metrics["charts/episodic_length"] = mean_length
            metrics["charts/success_rate"] = success_rate
            metrics["charts/collision_count_mean"] = collision_count_mean
            log_line += (f" return={mean_return:.1f} length={mean_length:.0f} "
                         f"success_rate={success_rate:.2f} collisions/ep={collision_count_mean:.2f} "
                         f"(n={len(completed_returns)})")
            if completed_collision_impacts:
                # Individual per-contact penalties (not summed), so both the
                # typical and the single worst contact this update are visible.
                metrics["charts/collision_impact_mean"] = float(np.mean(completed_collision_impacts))
                metrics["charts/collision_impact_worst"] = float(np.min(completed_collision_impacts))
        print(log_line)

        log(metrics, global_step)

        worker_buffer.reset()
        manager_buffer.reset()

        # Evaluation
        if update % args.eval_freq == 0:
            agent.manager_actor.eval()
            agent.manager_critic.eval()
            agent.worker_actor.eval()
            agent.worker_critic.eval()
            eval_returns = []
            eval_lengths = []
            eval_successes = []
            eval_collision_counts = []
            eval_collision_impacts = []
            for i in range(args.eval_episodes):
                eval_obs, info_eval = eval_env.reset(seed=args.seed + i)
                policy_fn = make_policy_fn(agent)
                eval_ep_reward = 0
                eval_ep_length = 0
                done_eval = False
                while not done_eval:
                    eval_obs, reward_eval, terminated_eval, truncated_eval, info_eval = eval_env.step(policy_fn(eval_obs))
                    eval_ep_reward += reward_eval
                    eval_ep_length += 1
                    done_eval = terminated_eval or truncated_eval
                eval_returns.append(eval_ep_reward)
                eval_lengths.append(eval_ep_length)
                eval_successes.append(bool(info_eval["is_success"]))
                # collision_count/collision_impacts are cumulative over the
                # whole episode (see the env's step()), so the last step's info
                # already carries every contact the episode had.
                eval_collision_counts.append(info_eval["collision_count"])
                eval_collision_impacts.extend(info_eval["collision_impacts"])

            mean_eval_return = float(np.mean(eval_returns))
            success_rate = float(np.mean(eval_successes))
            collision_count_mean = float(np.mean(eval_collision_counts))
            print(f"Evaluation at update {update}: return={mean_eval_return:.2f}, length={np.mean(eval_lengths):.2f}, "
                  f"success_rate={success_rate:.2f}, collisions/ep={collision_count_mean:.2f}")
            eval_metrics = {
                "eval/episodic_return": mean_eval_return,
                "eval/episodic_length": np.mean(eval_lengths),
                "eval/success_rate": success_rate,
                "eval/collision_count_mean": collision_count_mean,
            }
            if eval_collision_impacts:
                eval_metrics["eval/collision_impact_mean"] = float(np.mean(eval_collision_impacts))
                eval_metrics["eval/collision_impact_worst"] = float(np.min(eval_collision_impacts))
            log(eval_metrics, global_step)

            if args.save_eval_checkpoints:
                agent.save(os.path.join(checkpoint_dir, f"eval_{global_step:07d}.pt"))

            if mean_eval_return > best_eval_return:
                best_eval_return = mean_eval_return
                agent.save(os.path.join(checkpoint_dir, "best.pt"))

            # Solved-check: the same fixed grid and the same deterministic
            # controller as the random eval above on every pass, compared
            # point-by-point against the oracle's precomputed optimal return
            # for that exact starting position -- see algorithms/solved_check.py.
            solved_result = check_solved(lambda: make_policy_fn(agent), eval_env, optimal_grid, args.solved_tolerance,
                                         seed=args.seed)
            print(f"Solved-check at update {update}: solved={solved_result.solved} "
                  f"worst_gap={solved_result.worst_gap:.2f} at "
                  f"p_x0={solved_result.worst_point[0]:.2f} p_y0={solved_result.worst_point[1]:.2f} "
                  f"(consecutive={consecutive_solved})")
            log({
                "solved/is_solved": float(solved_result.solved),
                "solved/worst_gap": solved_result.worst_gap,
                "solved/mean_gap": float(solved_result.gaps.mean()),
            }, global_step)

            consecutive_solved = consecutive_solved + 1 if solved_result.solved else 0

            agent.manager_actor.train()
            agent.manager_critic.train()
            agent.worker_actor.train()
            agent.worker_critic.train()

            # See --early-stop-success-rate. Checked after best.pt so the
            # qualifying evaluation is always banked before we might stop.
            if (args.early_stop_success_rate is not None
                    and success_rate >= args.early_stop_success_rate
                    and mean_eval_return >= early_stop_min_return):
                consecutive_high_success += 1
            else:
                consecutive_high_success = 0
            if consecutive_high_success >= args.early_stop_patience:
                print(f"Early stop at update {update}: eval/success_rate >= "
                      f"{args.early_stop_success_rate} and eval/episodic_return "
                      f">= {early_stop_min_return:.1f} ({args.early_stop_optimal_frac:.0%} "
                      f"of mean optimal) for {consecutive_high_success} consecutive evaluations")
                log({"charts/early_stopped_at_update": update}, global_step)
                break

            # Independent of --early-stop-success-rate above: either one can
            # stop training. Saved once, the first time the criterion holds,
            # regardless of --solved-early-stop -- marks the moment the
            # policy became near-optimal even on a run left to run to its
            # full --total-timesteps budget for a coherent cross-seed plot.
            if consecutive_solved >= args.solved_consecutive:
                if not solved_checkpoint_saved:
                    print(f"Solved criterion held for {consecutive_solved} consecutive evals "
                          f"(tolerance={args.solved_tolerance}) at global_step={global_step}.")
                    agent.save(os.path.join(checkpoint_dir, "solved.pt"))
                    solved_checkpoint_saved = True
                    log({"solved/first_solved_step": global_step}, global_step)
                if args.solved_early_stop:
                    print("--solved-early-stop is on -- stopping training early.")
                    break

    agent.save(os.path.join(checkpoint_dir, "final.pt"))
    print(f"Saved checkpoints to {checkpoint_dir}")

    metrics_log.close()
    eval_env.close()
    if args.track:
        wandb.finish()
