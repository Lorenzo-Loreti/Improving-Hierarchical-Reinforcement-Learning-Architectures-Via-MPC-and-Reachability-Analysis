# Alignment tracker

Living record of the alignment of this repository with the thesis specification. Read it before starting a step and update it in the same pull request that changes the code.

- **Target ("spec v1").** Problem setup, architectures and rules in the Claude project's instructions, plus the design decisions in its decision log (`claude/decision-log.md`), as of 2026-10-06.
- **Starting point.** Tag `pre-alignment` (commit `ff50d7c`).
- **Last updated.** 2026-10-08: steps 1–6 closed; step 7 in progress on branch `align/step-07-ppo` (code committed; after the first acceptance runs the progress reward is undiscounted, DL-D22; acceptance runs to be repeated).

Contents: [1 How to use](#1-how-to-use-this-file) · [2 Workflow](#2-workflow-and-status) · [3 Baseline](#3-baseline-step-1) · [4 Audit summary](#4-audit-summary-step-2) · [5 Open decisions](#5-open-decisions) · [6 Matrix](#6-traceability-matrix) · [7 Inventory](#7-inventory-of-the-pre-alignment-code) · [8 Old results](#8-pre-alignment-results-reference-only) · [9 Change log](#9-change-log)

## 1. How to use this file

- **Where things live.**
  - The Claude project holds the specification and the design decisions.
  - This repository holds the code and the tests.
  - Each is the master for its own content. This file maps one onto the other and tracks progress. It is not a second copy of the specification.
- **One branch and one pull request per step** (`align/step-NN-<topic>`). The PR description lists the matrix rows it closes and the tests that prove it. A step is done when:
  1. the tests pass;
  2. its rows in §6 are updated;
  3. every design choice made during the step is recorded in the decision log;
  4. §9 has a line for it.
- **Conflicts.** When code and specification disagree, the specification wins by default. When the code exposes a flaw in the specification, update the decision log first, then the code, and never silently.
- **Open decisions** (Q-IDs, §5) are closed by a decision-log entry. Write that entry's ID into the Status column.
- **Notation introduced here.**
  - Matrix rows: M (model), E (environment step), R (contact, reward, metrics), AM (action map), AR (architectures), X (experiments), C (code).
  - Open decisions: Q.
  - Decision-log entries carry the prefix DL (DL-D1, DL-A4, DL-F3), so they cannot be confused with the row IDs.
  - Code names that differ from the spec: `u_max` is $a_{\max}$, `dt` is $T_s$, and `manager_freq` (also written $c$) is $H$.
- **Status legend.**
  - OK: conforms to the specification.
  - PARTIAL: partly conforms.
  - MISSING: not implemented.
  - CONFLICT: contradicts the specification.
  - IMPLICIT: the specification does not define it, so the code decides.
- **Class legend.**
  - S: semantic, changes the results.
  - F: missing feature or metric.
  - Y: code rules or style.
- Closed rows stay in the matrix with status OK, for traceability.

## 2. Workflow and status

| # | Step | Definition of done | Status | Branch / PR | Notes |
|---|---|---|---|---|---|
| 1 | Freeze the starting point | Tag; baseline test run recorded | **Done** (2026-10-06) | tag `pre-alignment` | 470 tests passed (§3). The old results stay reachable from the tag; removing them from `main` is part of step 4 (Q14) |
| 2 | Audit and traceability matrix | Inventory, matrix with evidence, open decisions | **Done** (2026-10-06) | `align/step-02-audit` | This file; evidence re-checked before closing |
| 3 | Ratify implicit choices; close the decisions that block steps 4–7 | Q1–Q9 and Q14–Q16 closed in the decision log | **Done** (2026-10-07) | `align/step-03-decisions` | Q1–Q9 and Q14–Q16 ratified as DL-D5–D15; Q10–Q13 stay open until steps 8–11 |
| 4 | Minimal infrastructure | Package layout, pinned dependencies, YAML configs, seeding, logging against the sample counter, pytest and CI, DL-F1–F6 as tests, cleanup of dead code and old results | **Done** (2026-10-07) | `align/step-04-infrastructure` | Package `src/hrlmpc` with YAML configs, seeding, run logging and the DL-F1–F6 tests; CI for `tests/`; every dependency locked at its pre-alignment version; old results untracked; dead code removed. The full suite passes in the locked virtual environment and CI is green |
| 5 | Physics module, tests first | One pure, batched step function implementing E1–E6 on polytopic geometry; tests of rows M and E | **Done** (2026-10-07) | `align/step-05-physics` | DL-D9–D12, DL-D16, DL-D17. `src/hrlmpc/geometry.py` and `physics.py` with their tests, reviewed adversarially. The full suite passes in the locked virtual environment (515 tests) and CI is green |
| 6 | MDP wrapper | Observation, goal, reward with the contact penalty, termination, disturbance sampling, action map, metrics R3–R4 | **Done** (2026-10-07) | `align/step-06-mdp` | DL-D13–D15, DL-D18. `src/hrlmpc/env.py`, `action_map.py` and the Gymnasium adapter `gym_env.py`, with their tests, reviewed adversarially. The full suite passes in the locked virtual environment (557 tests) and CI is green |
| 7 | Flat PPO on the new environment | Learns on a trivial layout; becomes the reference for logging and configs | **In progress** | `align/step-07-ppo` | DL-D19–D22, DL-F7. `src/hrlmpc/ppo.py` (bit-identical to the pre-alignment update), `rollout.py`, `evaluation.py`, `paths.py`, `train_ppo.py`, `analysis.py` and `configs/agent/ppo.yaml`, with their tests, reviewed adversarially. First acceptance runs: tunnel accepted, slalom failed (every seed learned to stand still); progress reward undiscounted since DL-D22. Pending: the author's test run and the repeated acceptance runs of DL-D20 (tunnel and slalom, 5 seeds × 204 800 samples) |
| 8 | Common hierarchical runner, then hPPO | One Manager/Worker loop; the Worker and the target space are interchangeable components | Not started | | Q10 |
| 9 | MPC Worker | Shared model module, $W = BD$, inter-sample constraints, no wall reaction in closed loop | Not started | | Q11 |
| 10 | Reachability | AR4 rebuilt on the tube MPC; inner approximation verified | Not started | | Q12 |
| 11 | Experimental protocol | Budget, seeds, tuning, sample-efficiency metric and statistics fixed before the final runs | Not started | | Q13 |
| 12 | Consolidation | README, lock file, reproduction script, tag `thesis-v1` | Not started | | |

## 3. Baseline (step 1)

- **Tags.**
  - `pre-alignment` points to commit `ff50d7c` on `main`, which is in sync with `origin/main` (checked 2026-10-06). It is an annotated tag (object `01ab953`).
  - The previous tag, `box-limits-final` (`b2959a4`), is the last version with per-axis speed and acceleration limits.
- **Tracked files.** 2 452 in total:
  - 99 are source, tests and docs;
  - about 2 350 are study outputs under `scenarios/*/studies/` (logs, metrics, configs, PDF figures);
  - checkpoints (`*.pt`, `scripts/checkpoints/`) are ignored.
- **Untracked file.** `docs/stato-ppo-mpc.md` is in the working tree: an Italian memo for the supervisors dated 2026-09-29, partly stale (see Q15).
- **Pre-alignment environment.** The package versions of the global Python 3.12 environment are recorded in `requirements-pre-alignment.txt` (pip freeze of 2026-10-06, committed in step 4). The reference stack is now pinned in `requirements.txt` (DL-D5).
- **Dependencies were not declared** at the tag (no `requirements.txt`, no `pyproject.toml`). What the code imports:
  - numpy, scipy, torch, gymnasium, matplotlib, pytest;
  - cvxpy with Clarabel;
  - gurobipy, which needs a licence and is required even for the tunnel;
  - osqp, used only in dead code;
  - wandb, optional.

  The author's machine runs Python 3.12.
- **Test suite.** 405 test functions in 24 files, 470 cases with parametrization.
  - **Baseline run** by the author on 2026-10-06 with `python -m pytest -q` from the repository root, on Windows with Python 3.12 (packages installed in the global interpreter, not in a virtual environment): `470 passed, 2 warnings in 64.54s`.
  - **The two warnings are harmless.** Gymnasium reports "Overriding environment SlalomEnv-v0 / TunnelEnv-v0 already in registry". Both scenarios name their package `envs`, and the test `conftest.py` files remove it from `sys.modules` so that the other scenario's copy can be imported. Each re-import calls `register(...)` again (`scenarios/*/envs/__init__.py`). The warning will disappear with the single environment package (E7).
  - The audit sandbox could not run the suite (its network policy blocks the package index), so this run is the reference.
- **Old results.** Every study under `scenarios/*/studies/` was produced on the pre-alignment dynamics, some of them on `box-limits-final`. They remain reproducible from the tags, but must never be mixed with post-alignment results (Q14).

## 4. Audit summary (step 2)

The findings that drive the plan. Evidence is in §6.

1. **The physics departs from the specification at every stage of the step** (rows M2–M6, E2–E6):
   - there is no viscous friction;
   - the discretization is the exact ZOH (a $\tfrac{1}{2}T_s^2 u$ position term) instead of semi-implicit Euler;
   - the disturbance acts on position and velocity instead of acceleration;
   - the speed limit acts on the nominal velocity, before the disturbance;
   - walls are handled by clamping the end position and zeroing the normal velocity, instead of projecting the velocity with continuous collision detection. A step that cuts a gate corner between two samples therefore goes undetected;
   - there is no wall friction;
   - the boundary at $p_x = -1$ is a position clip with no contact.
2. **The physics is implemented four times** (a scalar and a vector environment for each of the two scenarios, plus two copies of `actuation.py`). The model is hard-coded again in `tube_mpc.py`, `mpc_worker.py`, the helpers of `optimal_solver.py` and `stress_disturbed.py` (row E7).
3. **The geometry is a piecewise-constant corridor width, not polytopes** (M6). The slalom is equivalent to an arena rectangle with four rectangular obstacles, which is how the tube MPC already models it.
4. **The policies use a Beta distribution on the box $[-a_{\max}, a_{\max}]^2$, not a Gaussian** (AM1).
   - The environment then projects the box radially onto the disk $U$, so the corners (21.5% of the box's area) collapse onto the boundary of $U$.
   - Flat PPO and the hPPO Worker use the same map.
5. **Contact costs a flat −50 per contact step** (R2–R4).
   - The metrics record only the number of contact steps and a constant "impact".
   - Neither the cancelled normal velocity nor the physical correction of the commanded action is measured.
6. **hPPO** (AR2, AR5):
   - $H = 10$;
   - the goal is a 2-D relative goal in a ±10 m box, about 8 times the reach of one segment;
   - Worker reward = goal progress + 0.02 × environment reward.

   On the current dynamics it never passes the slalom gates (0/20 seeds).
7. **PPO_MPC uses a sound tube MPC**, but with the frictionless model, a position-and-velocity disturbance set and no inter-sample constraints (AR3a, AR3c, AR3d). Its sound parts:
   - a big-M MIQP for the obstacles;
   - exact disks;
   - a terminal rest set;
   - a fallback chain.
8. **PPO_MPC_reachability is disabled** (`raise SystemExit`) and stale (AR4):
   - it uses per-axis limits;
   - it uses the old QP worker;
   - it has no disturbance;
   - its "inner" approximation is no longer inner once the speed limit is accounted for.
9. **Protocol** (X2–X7). Samples are counted as environment steps everywhere, which is right but not yet declared. However:
   - the reported slalom comparison gives PPO and hPPO 1 024 000 steps × 20 seeds, but PPO_MPC 204 800 steps × 10 seeds;
   - there is no tuning protocol;
   - curves show the median and IQR;
   - the comparison tooling's only statistical test is a Mann–Whitney U that ranks unsolved seeds at $+\infty$, which is invalid when budgets differ (bootstrap intervals appear only in a one-off script).
10. **Code rules** (C1–C5):
    - no type hints;
    - no external configuration files;
    - no declared stack;
    - no CI;
    - GPU on by default without deterministic algorithms;
    - heavy duplication across scenarios, training loops and study tooling.

**Already aligned or worth keeping:**
- radial input saturation (E1);
- non-terminal contact (R1);
- environment steps as the sample unit (X1);
- `metrics.jsonl` logging per run (C4);
- seeding of every RNG (part of C3);
- the obstacle formulation and fallback logic of the tube MPC (AR3b, AR3e);
- the oracle and solved-check machinery, to be redefined rather than discarded.

## 5. Open decisions

Q1–Q9 and Q14–Q16 are closed in step 3; the others at the start of the step they block. The recommendations are proposals for the author, not decisions.

| ID | Question | Pre-alignment code | Recommendation (proposal) | Blocks step | Status |
|---|---|---|---|---|---|
| Q1 | Reference stack (C5) | numpy, scipy, PyTorch (own PPO), Gymnasium, cvxpy + Clarabel, gurobipy, matplotlib; global Python 3.12 interpreter on Windows; nothing pinned | Keep it, since changing libraries needs a reason; declare it and pin versions in a virtual environment. Before upgrading anything, record the versions of the pre-alignment environment (`python -m pip freeze`). Decide whether running every architecture may require a Gurobi licence (today it does, even on the tunnel) | 4 | **ratified** (DL-D5): current stack, Python 3.12 venv with pinned versions; Gurobi only for the MPC Worker |
| Q2 | Definition of a sample (X1) | Environment steps summed over the parallel envs; evaluation, oracle and MPC predictions not counted | Ratify. One sample is one call of the physical step during training. Manager decisions are logged as a secondary counter. MPC model predictions are not samples but are reported as compute time | 4 | **ratified** (DL-D6): environment steps; early stopping off in comparisons |
| Q3 | Numerical parameters (M7) | $T_s = 0.1$ s, $a_{\max} = 2.5$, $v_{\max} = 1.2$, no friction | Choose $\gamma$ with $v_{\max} < a_{\max}/\gamma$ ($\gamma < 2.08\ \mathrm{s^{-1}}$ with these values), choose $\gamma_w$, and choose $\bar d$ satisfying DL-F2 if robust invariance of $V$ is wanted. The current 1× disturbance (0.005 m, 0.05 m/s) is the product of the disks of radii $T_s^2 \bar d$ and $T_s \bar d$ with $\bar d = 0.5\ \mathrm{m/s^2}$ | 5 | **ratified** (DL-D9): $\gamma = 0.5$, $\gamma_w = 1.0\ \mathrm{s^{-1}}$, $\bar d \in \{0, 0.5, 1.0\}\ \mathrm{m/s^2}$; $T_s$, $a_{\max}$, $v_{\max}$ unchanged |
| Q4 | Distribution of $d_k$ in $D$ (M5) | Uniform on two independent disks for $w_p$ and $w_v$ | $d_k$ i.i.d., uniform in area on $D$; worst-case (boundary) disturbances only for stress tests | 5 | **ratified** (DL-D10): i.i.d. uniform on $D$; worst cases only in stress tests |
| Q5 | Geometry and layouts (M6) | Width profile; fixed slalom, tunnel as a sanity check | One geometry module (H-representation of the arena and the obstacles) shared by the env, the MPC, the oracle and the reachability code; the slalom as the fixed main layout | 5 | **ratified** (DL-D11): polytopic geometry module; fixed slalom with 4 rectangular obstacles |
| Q6 | Arena boundary (M6, DL-A2) | $p_x \in [-1, L+1]$ enforced by a position clip without contact; goal at $p_x \ge L$ | A closed arena polygon whose faces are all physical walls, with the goal region strictly inside it | 5 | **ratified** (DL-D11): arena $[-1, 11] \times [-2, 2]$, all faces walls; goal region $p_x \ge 10$ |
| Q7 | MDP elements the spec does not define (R5) | Observation = normalized $(p, v)$; success when $p_x \ge 10$; horizon 200; spawn uniform on $[0,2]\times[-1,1]$ at rest; reward −1 per step, +1000 on success (replacing the step's other terms), progress $10\,\Delta p_x$ cut at $L$, effort $-0.01\lVert u\rVert^2/a_{\max}^2$ on the delivered input | Ratify explicitly (with any changes) in the decision log, including whether the success bonus should replace or add to the other terms of its step | 6 | **ratified** (DL-D13): as now, with the success bonus added and exact potential-based shaping; **revised** (DL-D22): the progress term is undiscounted again, and the learners' discount is set in the agent configuration |
| Q8 | Form of the contact penalty (R2) | Flat −50 per contact step | Price the "free brake" that motivates DL-D1: a term proportional to the cancelled normal velocity, plus an optional small per-step term for sliding. Use the same function for every architecture, inside the hPPO Worker reward too | 6 | **ratified** (DL-D14): $-c_n \lVert \Delta v^{\mathrm{w}} \rVert_2 - c_s \mathbb{1}[\text{contact}]$, $c_n = 50$, $c_s = 1$ to calibrate |
| Q9 | Action map (AM1) | Beta per axis on the box, then radial projection onto $U$ in the env | The spec text assumes a Gaussian policy. Either amend it to the Beta, a bounded-support option it already lists, or switch to a Gaussian. Keep the Beta, but replace the many-to-one box→disk projection with a bijection of the square onto the disk (e.g. $x \mapsto (\lVert x\rVert_\infty/\lVert x\rVert_2)\,x$), so that physical corrections come only from the speed limiter and the walls | 6–7 | **ratified** (DL-D15): per-axis Beta + bijection of the square onto $U$; spec text to update |
| Q10 | Manager/Worker interface (AR5) | $H = 10$; targets: hPPO ±10 m box, PPO_MPC ±1.8 m box, reach arm 4-D; Worker reward = progress + 0.02 × env reward; no Worker termination | One target space (relative position, sized to the $H$-step reach) shared by AR2–AR4 and restricted only in AR4; the common contact penalty in the Worker reward | 8 | open |
| Q11 | MPC formulation (AR3) | Tube MPC, big-M MIQP + Clarabel, $W$ on $(p, v)$, no inter-sample constraints | Keep the tube MPC but have it read the shared model module. Use $W = BD$, or a declared outer bound $W' \supseteq BD$. Exclude corner cutting (one face per segment, or inflated obstacles). Decide whether to tighten $V$ or rely on DL-F3 | 9 | open |
| Q12 | Reachability (AR4) | Disabled and stale | Use the robust $H$-step reachable set of target positions under the tube MPC's tightened constraints. Check the inner approximation against DL-F5/F6 in the obstacle-free case. Define "reached" as being within the tube cross-section around the target | 10 | open |
| Q13 | Experimental protocol (X2–X7) | Unequal budgets and seeds; no tuning protocol; median/IQR; Mann–Whitney with unsolved seeds ranked $+\infty$; solved-check against the undisturbed oracle | Equal budget and seed list per comparison. A declared tuning protocol on held-out seeds. The sample-efficiency metric fixed in advance: first-solve time with censoring handled by survival analysis, or the area under the curve. Means with bootstrap CIs and a correction for multiple comparisons. A disturbed reference for disturbed runs | 11 | open |
| Q14 | Old results tracked under `studies/` | About 2 350 tracked files from the old dynamics | Remove them from `main` in step 4 (they stay in the tag) so they cannot be mixed with new results; keep the index in §8 | 4 | **ratified** (DL-D7); carried out in step 4 |
| Q15 | `docs/stato-ppo-mpc.md` | Untracked, Italian, partly stale | Move what is still valid (tube-MPC deviations, reachability options, questions for the supervisors) into an English doc or the decision log, then drop the file | 3 | **ratified** (DL-D8): content carried over (notes below; supervisor questions in the decision log); the author moves the memo out of the repository |
| Q16 | Tunnel scenario | A full code copy of the slalom | Keep it only as a layout/config of the single environment, or drop it | 5 | **ratified** (DL-D12): kept as a layout of the single environment |

### Notes on open decisions

From the memo `docs/stato-ppo-mpc.md` of 2026-09-29 (DL-D8). Its numbers refer to the pre-alignment dynamics.

**Q11, MPC formulation.** The tube MPC deviates deliberately from the reference note (*Robust Tube-Based Model Predictive Control for Linear Systems with Non-Convex State Constraints*). The deviations are documented in the module docstring of `algorithms/tube_mpc.py`:

| Deviation | Consequence |
|---|---|
| "Safe stop" terminal set (at rest in free space) instead of a terminal set around the target | Recursive feasibility (Thm 4.1 of the note) survives changes of target; convergence (Thm 4.2) is lost |
| Margin $\rho$ also on the walls | — |
| Binaries fixed for obstacles that cannot be reached in the horizon | Exact |
| No inter-sample constraints | To be reversed (row AR3c) |
| Planned speed tightened to $v_{\max}$ minus the tube's speed margin | The env's speed limiter never acts |

A fixed-binary scheme (remark 4.5 of the note) would solve the mixed-integer problem only when the Manager replans and a QP in the other steps. It would keep the guarantees and should cut the compute cost.

**Q12, reachability options.** The memo predates the disk limits and describes the sets per axis. With Euclidean limits the obstacle-free reachable set of positions is a disk (DL-F6). The options:

| Option | Target set | For | Against |
|---|---|---|---|
| A. Reachable at rest | Positions the nominal system can reach and stop at within one segment, under the tube-tightened constraints. Closed form, depends on $v_0$ | Consistent with the safe-stop terminal set; every target is reachable; no extra solver | Targets at rest may slow cruising (seen with hPPO and small goal boxes) |
| B. Reachable in passing | Positions reachable at the end of the segment with any velocity, as in the old reach arm | Continuity with the earlier study; does not brake | The target is not an equilibrium reachable within the horizon |
| C. Reachable and obstacle-free | A or B intersected with the free space: projection onto the nearest reachable free point, with a small mixed-integer problem at each replanning | The Manager can no longer ask for a target inside a wall | Non-convex set; one more solve every $H$ steps |
| D. Backward reachability (liveness) | Targets leading to states from which the goal can no longer be reached (dead ends in front of the gates) are discarded | Addresses the local minima the MPC Worker gets stuck in | More ambitious; needs a careful theoretical definition |

Earlier finding, on the old dynamics (`docs/reachability-regime-study.md`): restricting the targets helped only when the agility ratio $v_{\max}/(a_{\max} H T_s)$ was above about 0.8. The tube raises the nominal system's effective ratio (about 0.87 at the 2× disturbance, in the memo's estimate), so reachability may matter under disturbance.

The memo's plan:
1. Implement A and B as a switch on the target map.
2. Expect no effect without disturbance and a benefit at 2×.
3. Run a regime check at agility ratios 0.8 and 1.2.
4. Use C and D only if A and B fail.

**Q13, reference before the oracle.** Until the oracle is ported (steps 9–11), evaluations measure the arrival time against the bound of DL-F7, the shortest free path covered at the fastest admissible speed. In the tunnel the bound is the minimum time. In the slalom it is only a lower bound (70–88 steps over the grid, against 69–86 for the straight run), so the acceptance criterion of DL-D20 applies to the tunnel only.

## 6. Traceability matrix

Sources: "Instr." is a section of the project instructions; "DL-" is a decision-log entry. The evidence cites the code at tag `pre-alignment`; paths are relative to `scenarios/slalom/envs/`, `algorithms/` or `scenarios/slalom/scripts/` unless given in full. The tunnel copies have the same structure. Step = the workflow step that closes the row.

### Model and physics (M)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| M1 | State $x=(p,v)\in\mathbb{R}^4$; input $u\in\mathbb{R}^2$ = commanded acceleration; unit point mass | Instr. Setup | Legacy: observation $[p_x,p_y,v_x,v_y]$, action $[a_x,a_y]$ (`slalom_env.py:61-75`). New: `physics_step` advances positions, velocities and commanded accelerations given as $(B, 2)$ arrays (`src/hrlmpc/physics.py`), step 5. `NavigationEnv` observes $(p, v)$ mapped to $[-1, 1]$ and takes $x \in [-1, 1]^2$ or commanded accelerations (`src/hrlmpc/env.py`), step 6 | OK | — | keep | `test_step_returns_batches_and_leaves_its_inputs_untouched`, `test_observations_are_normalized_with_fixed_bounds` | 5–6 |
| M2 | Isotropic viscous friction $\gamma>0$ | DL-A1 | Legacy: none; the velocity block of $A$ is the identity (`slalom_env.py:79-84`, `tube_mpc.py:635-638`, `mpc_worker.py:81-92`). New: stage 2 of `physics_step` and `model.py` use the factor $1-\gamma T_s$, step 5 (legacy environments: E7) | OK | — | keep | `test_step_equals_the_linear_model_when_no_correction_acts` | 5 |
| M3 | Semi-implicit Euler: $v^+=(1-\gamma T_s)v+T_s(u+d)$, $p^+=p+T_s v^+$ | DL-D4 | Legacy: exact ZOH of the frictionless double integrator, $B=[\tfrac12 T_s^2 I;\ T_s I]$ (`slalom_env.py:86-91`, `vec_slalom_env.py:57-69`, `tube_mpc.py:639-642`; the oracle reads `env.A`, `env.B` at `optimal_solver.py:305-306`). New: stages 2 and 6 of `physics_step` coincide with `model.free_step` whenever no correction acts, step 5; the MPC Worker, the oracle and the reachability code adopt `model.py` in steps 9–10 | OK | — | keep | `test_step_equals_the_linear_model_when_no_correction_acts`, `tests/test_model_facts.py` | 5 |
| M4 | $U$, $V$, $D$ are Euclidean balls | DL-A5 | Legacy: $U$ and $V$ are disks (`actuation.py:85-108`); the action space is the box $[-a_{\max},a_{\max}]^2$ (`slalom_env.py:75`); $D$ is not an acceleration ball (M5). New: radial saturation onto $U$ and $V$; disturbances outside $D$ are rejected (`physics.py`), step 5. The action map of DL-D15 follows in step 6 (AM1) | OK | — | keep | `test_commanded_action_is_saturated_radially`, `test_speed_limiter_is_radial_and_the_speed_limit_always_holds`, `test_invalid_inputs_are_rejected` | 5 |
| M5 | Disturbance at acceleration level, $w=Bd$, $d\in D$ | DL-D2 | Legacy: $w=(w_p,w_v)$ uniform on two independent disks (`actuation.py:111-126`, `config.py:19-40`); the tube MPC uses the same product set (`tube_mpc.py:664`). New: $d$ enters stage 2 together with the saturated input (`physics.py`), step 5. Still open: the sampling of DL-D10 (step 6) and the tube MPC (step 9). `NavigationEnv` samples $d$ as DL-D10 prescribes, step 6. Still open: the tube MPC (step 9) | PARTIAL | S | refactor the MPC (step 9) | `test_disturbance_enters_like_an_acceleration`, `test_disturbance_is_not_saturated_with_the_input`, `test_disturbances_are_iid_uniform_on_the_disk`; DL-F3 | 5, 6, 9 |
| M6 | Arena polytope with polytopic obstacles; every face is a physical wall | Instr. Setup, DL-A2, DL-D16 | Legacy: piecewise-constant width profile (`width_profile.py`); $p_x$ confined to $[-1,L+1]$ by a position clip without contact (`slalom_env.py:65-66, 189`); the tube MPC turns the profile into 4 rectangles (`tube_mpc.py:534-568`). New: convex polygons in H-representation, layouts built from the YAML configs and validated, free space of DL-D16 (`src/hrlmpc/geometry.py`), step 5; the tube MPC adopts it in step 9 | OK | — | keep | `tests/test_geometry.py`, `test_random_layouts_are_never_penetrated` | 5 |
| M7 | $v_{\max}<a_{\max}/\gamma$ (DL-F1); robust invariance of $V$ (DL-F2) | DL-F1, DL-F2 | Legacy: no $\gamma$; $T_s=0.1$, $a_{\max}=2.5$, $v_{\max}=1.2$ (`config.py:11-18`); these conditions are not validated. New: the DL-D9 values in `configs/env/*.yaml`; `PhysicsParams` rejects violations of F1 and F2, step 4 | OK | — | keep | `tests/test_config.py`, `test_check_physics_rejects_violations` | 4–5 |

### Environment step (E)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| E1 | Commanded action saturated radially onto $U$ | Instr. step 1 | Legacy: `project_disk` (`actuation.py:85-93, 102`). New: stage 1 of `physics_step`, without overflow for huge commands, step 5 | OK | — | keep | `test_commanded_action_is_saturated_radially`, `test_huge_commands_are_saturated_without_overflow` | 5 |
| E2 | Candidate velocity $\tilde v=(1-\gamma T_s)v+T_s(u+d)$ | Instr. step 2, DL-D3 | Legacy: nominal $v+T_s u$ without friction; the disturbance is added after the limiter (`slalom_env.py:172-183`). New: stage 2 of `physics_step`, step 5 | OK | — | keep | `test_step_equals_the_linear_model_when_no_correction_acts` | 5 |
| E3 | Speed limiter $\hat v=\Pi_V(\tilde v)$, before the wall reaction | Instr. step 3, DL-A4, DL-D3 | Legacy: limiter on the nominal velocity, by replacing $u$ (`actuation.py:96-108`); a guard projection after the disturbance (`slalom_env.py:190`). New: stage 3 of `physics_step`, step 5 | OK | — | keep | `test_speed_limiter_is_radial_and_the_speed_limit_always_holds`, `test_speed_limiter_acts_before_the_wall_reaction` | 5 |
| E4 | Wall reaction: Euclidean projection of $\hat v$ onto the half-planes of the faces crossed by $[p_k,p_k+T_s\hat v]$; continuous collision detection, repeated | Instr. step 4, DL-A4, DL-D16, DL-D17 | Legacy: the end position is clamped and the normal velocity zeroed, checked at the end point only; the crossing point only classifies which wall was hit (`slalom_env.py:230-244`, `vec_slalom_env.py:164-181`). A step that cuts a corner between samples is not detected. New: stage 4 of `physics_step`: first impact (DL-D17) on the free space of DL-D16, exact projection, step 5 | OK | — | keep | `test_random_layouts_are_never_penetrated` (independent clipping, KKT check), `test_concave_corners_stop_the_agent`, `test_convex_corner_needs_one_face`, `test_no_phantom_contacts_near_a_vertex`, `test_first_impact_ignores_faces_the_corrected_motion_misses` | 5 |
| E5 | Wall friction $\gamma_w$ on the sliding velocity | Instr. step 5, DL-A3 | Legacy: absent. New: stage 5 of `physics_step`: the agent lies on a face at the start and at the end of the step and moves along it, step 5 | OK | — | keep | `test_sliding_along_a_wall_is_damped_by_wall_friction`, `test_sliding_on_a_slanted_face_goes_on_until_friction_stops_it`, `test_no_wall_friction_when_leaving_a_face_at_its_vertex` | 5 |
| E6 | $p_{k+1}=p_k+T_s v_{k+1}$ | Instr. step 6 | Legacy: $p^+=p+T_s v+\tfrac12 T_s^2 u$ (`slalom_env.py:183`). New: stage 6 of `physics_step`, step 5 | OK | — | keep | as M3 | 5 |
| E7 | One module implements the physical step for all architectures | Instr. Code | Legacy: four implementations (`slalom_env.py`, `vec_slalom_env.py`, `tunnel_env.py`, `vec_tunnel_env.py`) and two `actuation.py`; the plant is re-implemented in `stress_disturbed.py:168`; $A$, $B$ are hard-coded in `tube_mpc.py:635-642` and `mpc_worker.py:81-92`. New: `physics_step` is the only integration of the dynamics in `src/hrlmpc`, and a single agent is a batch of one, bit for bit, step 5. The legacy environments remain until step 6. The MDP wrapper steps through `physics_step` (step 6). The legacy environments remain until their training loops are replaced; the pre-alignment flat PPO stays until step 8 for the comparison of DL-D19 (DL-D21) | PARTIAL | S | remove the legacy environments together with their training loops (steps 8–9) | `test_batch_equals_single_agents_bit_for_bit`, `test_the_environment_steps_through_physics_step` | 5–9 |
| E8 | Tests on non-penetration, speed limit, contact at corners, wall friction | Instr. Code | Legacy: the tests cover radial saturation, the speed limit, gate faces and a corner hit (`envs/tests/test_slalom_env.py:80-704`); none cover friction or penetration between samples. DL-F1–F6 are tested since step 4 (`tests/test_model_facts.py`). New: `tests/test_geometry.py` and `tests/test_physics.py` cover rows M and E, DL-D16 and DL-D17, step 5 | OK | — | keep | — | 5 |

### Contact, reward and metrics (R)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| R1 | Contact allowed and non-terminal | DL-D1 | Legacy: the episode continues after a contact (`slalom_env.py:202-249`). New: `NavigationEnv` never ends an episode on contact (`src/hrlmpc/env.py`), step 6 | OK | — | keep | `test_contact_metrics_count_events_duration_and_impulse` | 6 |
| R2 | Same contact penalty for every architecture, also in the hPPO Worker reward | DL-D1 | Legacy: flat −50 per contact step (`config.py:78`). PPO and the Managers see it through the env reward. The hPPO Worker sees 0.02 × it through the extrinsic mix (`hppo_train.py:1123-1124`), and `--worker-extrinsic-coef 0` removes it. A `--contact-penalty` flag exists in hPPO and PPO_MPC (`hppo_train.py:94`, `ppo_mpc_train.py:106`) but not in flat PPO. New: the penalty of DL-D14, calibrated as DL-D18 (bound updated by DL-D22 and still met), is defined once (`RewardConfig`) and applied by `NavigationEnv` to every agent, step 6. Its weight in the hPPO Worker's reward is set with Q10 | PARTIAL | S | use it in the Worker reward (step 8) | `test_reward_of_a_head_on_impact`, `test_shipped_configs_meet_the_criterion_of_the_contact_penalty` | 6, 8 |
| R3 | Contact metrics: number, duration, cancelled normal velocity | Instr. Metrics | Legacy: `collision_count` counts contact steps, and `collision_impacts` stores the constant penalty (`vec_slalom_env.py:206-208`); no normal velocity. New: per step the contact flag and $\lVert\Delta v^{\mathrm{w}}\rVert_2$; per episode the contact events, steps and cancelled normal velocity (`EpisodeStats`), step 6 | OK | — | keep | `test_contact_metrics_count_events_duration_and_impulse`, `test_episode_statistics_are_the_sums_of_the_step_quantities` | 6 |
| R4 | Frequency and magnitude of the physical corrections of the commanded action | Instr. Metrics | Legacy: not logged; `delivered_action` appears only in the scalar env's info (`slalom_env.py:284`), not in the vector env's (`vec_slalom_env.py:227-234`). New: per step the corrections of the speed limiter, the walls and the saturation onto $U$; per episode their frequency and total (`EpisodeStats`), step 6 | OK | — | keep | `test_correction_metrics_of_the_speed_limiter`, `test_commanded_accelerations_are_saturated_and_the_correction_recorded`, `test_episode_statistics_are_the_sums_of_the_step_quantities` | 6 |
| R5 | MDP elements the spec does not define | — | Legacy: observation = normalized $(p,v)$ (`common.py:61`); success when $p_x\ge L$ (`slalom_env.py:254`); horizon 200 (`config.py:53`); spawn box, uniform, at rest (`spawn_sampler.py:96-101`); reward terms (`config.py:55-118`); fixed layout (`width_profile.py:139-196`). New: DL-D13 with the success bonus added to the step's other terms (`src/hrlmpc/env.py`), step 6. The progress term, potential-based in exact form in step 6, is undiscounted again since DL-D22 (`reward.shaping_discount = 1`; the learners' discount is `update.discount` in `configs/agent/ppo.yaml`), step 7 | OK | — | keep | `test_success_terminates_and_adds_the_bonus_to_the_other_terms`, `test_shaping_telescopes_without_discount`, `test_discounted_shaping_telescopes`, `test_truncation_after_the_horizon`, `test_the_discount_is_the_agents_own` | 6, 7 |

### Action map (AM)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| AM1 | Map from policy output to $U$ identical for PPO and the hPPO Worker; its distortion known and monitored | Instr. RL action space | Legacy: beta per axis on the box (`common.py:191-224`, `ppo/ppo.py:217`, `hppo/hppo.py:654`), then radial projection onto $U$ in the env; the two arms share the map (`hppo/tests/test_hppo.py:592`); the spec text assumes a Gaussian. New: the bijection of DL-D15 inside `NavigationEnv` (`src/hrlmpc/action_map.py`); its Jacobian lies between $a_{\max}^2/2$ and $a_{\max}^2$; corrections are monitored through R4, step 6. The Beta policy follows in step 7 | OK | — | keep | `tests/test_action_map.py`, `test_actions_are_mapped_onto_u_by_the_bijection_of_d15` | 6–7 |

### Architectures (AR)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| AR1 | PPO: the policy outputs the commanded acceleration | Instr. Arch. 1 | Legacy: `ppo/ppo_train.py:394-401`. New: `hrlmpc.train_ppo` trains the in-house PPO of DL-D19 (`src/hrlmpc/ppo.py`) on `NavigationEnv`; the policy's action is mapped onto $U$ by DL-D15, step 7 | PARTIAL | F | acceptance runs of DL-D20 | `tests/test_ppo.py`, `tests/test_train_ppo.py`; tunnel runs of DL-D20 | 7 |
| AR2 | hPPO: the Manager sets a target every $H$ steps; a goal-conditioned PPO Worker tracks it | Instr. Arch. 2 | Re-plans every `manager_freq` = 10 steps or at episode end (`hppo/hppo_train.py:163, 1202`); 2-D relative goal (`:880, 1111`) | OK | — | refactor onto the common runner | — | 8 |
| AR3 | PPO_MPC: PPO Manager; MPC Worker with full knowledge, planning contact-free | Instr. Arch. 3 | `TubeMPCWorker.from_env` (`ppo_mpc/ppo_mpc_train.py:465`); targets = position + goal (`:547`) | PARTIAL | S | refactor (AR3a–f) | — | 9 |
| AR3a | MPC prediction model = spec model; $U$, $V$ as cones or as declared polygons | Instr. Critical points | Frictionless ZOH, hard-coded (`tube_mpc.py:635-642`). $U$ and $V$ are exact second-order cones in the applied plan; a 16-gon relaxation is used only for Gurobi's binary choice (`tube_mpc.py:207, 247-248`) | PARTIAL | S | refactor: read the model module | model consistency env vs MPC | 9 |
| AR3b | Non-convex obstacles with a declared formulation | Instr. Critical points | Big-M MIQP over the horizon; binaries of unreachable obstacles fixed (`tube_mpc.py:534-568, 817-819, 874-901`) | OK | — | keep | — | 9 |
| AR3c | No corner cutting between consecutive samples | Instr. Critical points | "No inter-sample constraints" (`tube_mpc.py:147-149`); the binaries are independent per stage | MISSING | S | implement (Q11) | closed-loop segments never cross an obstacle | 9 |
| AR3d | Robustness to $W$ (tube, tightening), recursive feasibility | Instr. Critical points | Rigid tube with an ε-outer mRPI set (`tube_mpc.py:489-517`); LQR gain (`:647`); tightened disks and enlarged obstacles (`:674-703`); terminal rest set (`:766-767`). $W$ is on $(p,v)$ (`:664`) | PARTIAL | S | refactor: $W=BD$, or a declared $W'\supseteq BD$ | recursive feasibility under worst-case $d$ | 9 |
| AR3e | Behaviour when the target is unreachable | Instr. Critical points | The target enters only the cost (`tube_mpc.py:578-583`). Fallback chain: solver → shifted previous plan → emergency brake (`:1106-1120`). Outcomes are logged in training (`ppo_mpc/ppo_mpc_train.py:655-664`) but discarded at evaluation (`:348`) | OK (logging PARTIAL) | F | keep; log outcomes at evaluation | — | 9 |
| AR3f | Wall reaction never active under the MPC Worker | Instr. Critical points | Zero-contact closed-loop tests with a scripted Manager (`tests/test_tube_mpc.py:347-517`). Untested: a learned Manager, delivered = commanded, contact between samples | PARTIAL | F | add tests | wall-reaction count = 0 in closed loop | 9 |
| AR4 | PPO_MPC_reachability: targets only in the robust $H$-step reachable set | Instr. Arch. 4 | Disabled by `raise SystemExit` (`script_ppo_mpc_reach.py:256-259`). Stale: per-axis limits, old QP worker, no disturbance | MISSING | S | rewrite on the tube MPC and the common runner | — | 10 |
| AR4a | Reachable set computed robustly; approximation declared (inner) | Instr. Critical points | Per-axis zonotope in $(\Delta p,\Delta v)$, inner only with respect to the input bound (`ppo_mpc_reach/ppo_mpc_reach.py:93-96, 160-193`). Not inner under the speed limit; no $W$; gates along the path ignored | CONFLICT | S | rewrite (Q12) | every assigned target reached within $H$ steps for sampled $d$ | 10 |
| AR4b | Manager action restricted to the set without breaking PPO | Instr. Critical points | Affine parametrization of the set with a shrink factor (`ppo_mpc_reach/ppo_mpc_reach.py:160-185, 574-584`) | PARTIAL | S | keep the idea; rewrite | — | 10 |
| AR4c | Meaning of "target reached" under disturbance | not in the spec | None: no tolerance, no reached flag | MISSING | S | decide (Q12) | reached-rate metric | 10 |
| AR5 | Manager/Worker interface: $H$, target space, Worker reward, Manager shaping | Instr. Critical points | $H=10$ in both arms (`hppo/hppo_train.py:163`, `ppo_mpc/ppo_mpc_train.py:149`). Target box ±10 m in hPPO (`hppo/hppo_train.py:166`), ±1.8 m in PPO_MPC (`ppo_mpc/ppo_mpc_train.py:189-197`), 4-D in the reach arm. Manager reward = discounted env reward over the segment (`hppo/hppo_train.py:1107`, `ppo_mpc/ppo_mpc_train.py:574`) | PARTIAL | S | unify (Q10) | — | 8 |

### Experiments and fair comparison (X)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| X1 | One definition of "sample" for all architectures | Instr. Fair comparison | Environment steps everywhere (`ppo/ppo_train.py:391`, `hppo/hppo_train.py:1067`, `ppo_mpc/ppo_mpc_train.py:543`); declared in DL-D6; counters `SampleCounter` and records keyed by `env_steps` in `src/hrlmpc/utils/runlog.py` (step 4). Flat PPO counts with the environment's counter and logs against it; evaluations are not counted (step 7) | PARTIAL | Y | use the counters in the hierarchical loops | `tests/test_runlog.py`, `tests/test_train_ppo.py` | 7–9 |
| X2 | Same interaction budget | Instr. Fair comparison | Defaults are equal (`study.py:111`), but the reported slalom comparison runs PPO and hPPO for 1 024 000 steps and PPO_MPC for 204 800 (`study_ppo_mpc.py:17-19`); early stop on solve is on by default (`ppo/ppo_train.py:111`) | CONFLICT | S | enforce in the comparison tooling | comparison refuses unequal budgets | 11 |
| X3 | Same seeds, at least 5 per configuration | Instr. Fair comparison | Same integer seeds; 20 for PPO and hPPO (`study.py:191`) vs 10 for PPO_MPC (`study_ppo_mpc.py:6`) | PARTIAL | S | one seed list per comparison | — | 11 |
| X4 | Comparable tuning effort | Instr. Fair comparison | No protocol; only hPPO's Worker coefficient was swept (`sweep_extrinsic_coef.py:161`); PPO_MPC "has not been re-swept" (`ppo_mpc/ppo_mpc_train.py:176`) | MISSING | S | define (Q13) | — | 11 |
| X5 | Same physics, contact penalty and action map | Instr. Fair comparison | Legacy: every arm uses the same env classes and map (`script_ppo_mpc.py:29-37` vs `script_ppo.py:20-31`); hPPO and PPO_MPC can override the penalty from the command line, flat PPO cannot. New: physics, contact penalty and action map come from one configuration through `NavigationEnv`, step 6, which flat PPO uses since step 7; the legacy hierarchical arms keep theirs until steps 8–9 | PARTIAL | S | every new training loop uses `NavigationEnv` | `tests/test_env.py`, `tests/test_train_ppo.py` | 6–9 |
| X6 | Metrics: sample efficiency (defined operationally), success rate, contacts, corrections, final return, compute time | Instr. Metrics | First solve = the second consecutive solved-check within 5 of the oracle at the 25 grid starts, checked every 10 240 steps (`ppo/ppo_train.py:118-130`, `solved_check.py:98`). Success rate is loaded but not plotted (`study.py:464`). MPC solve time is logged only in training. New: evaluation of the deterministic policy from a 5×5 grid every 10 240 samples (DL-D20): success rate, unshaped return, arrival gap against the bound of DL-F7, contacts and corrections; training-episode metrics and samples per second in `metrics.jsonl`, step 7 | PARTIAL | S | define the sample-efficiency metric (Q13) | `tests/test_evaluation.py`, `tests/test_paths.py` | 7, 11 |
| X7 | Mean with confidence intervals; appropriate tests | Instr. Code | Median + IQR (`study_plots.py:161`); Mann–Whitney with unsolved seeds = $+\infty$ (`study_plots.py:210-216`); no CIs in the comparison tooling (a bootstrap interval only in the one-off `study_init_sampler.py:330`); no correction for multiple comparisons. New: `hrlmpc.analysis` groups runs by configuration and gives means with Student-t 95% intervals over seeds, step 7 | PARTIAL | S | statistical tests and comparison protocol (Q13) | `tests/test_analysis.py` | 7, 11 |

### Code (C)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| C1 | Modular, typed Python; English docstrings, comments, identifiers | Instr. Code | Legacy: no type hints in the RL code, partial docstrings, monolithic training loops (`hppo/hppo_train.py`: 1 485 lines). New code in `src/hrlmpc` is typed, with English docstrings | PARTIAL | Y | write new code this way; legacy replaced in steps 5–9 | type checker in CI (to add) | all |
| C2 | External configuration files (YAML or similar) | Instr. Code | `configs/env/slalom.yaml`, `tunnel.yaml` with a strict typed loader (`src/hrlmpc/config.py`), step 4. `configs/agent/ppo.yaml` with `src/hrlmpc/ppo_config.py`; both loaders reject repeated keys and non-finite numbers, and `hrlmpc.train_ppo` reads both files, step 7. Legacy entry points still use argparse defaults and dataclasses | PARTIAL | Y | every new entry point reads YAML | `tests/test_config.py`, `tests/test_ppo_config.py`, `tests/test_strict_yaml.py` | 4–11 |
| C3 | Fixed seeds, reproducible experiments | Instr. Code | Legacy: RNGs seeded (`ppo/ppo_train.py:271-274`), `--cuda` on by default (`:89`), no deterministic algorithms. New: `seed_everything` with deterministic PyTorch (`src/hrlmpc/utils/seeding.py`); `meta.json` records seed, git commit and package versions (`runlog.py`), step 4. `hrlmpc.train_ppo` seeds every generator, trains on the CPU with one thread and deterministic algorithms, restores the global settings afterwards, and repeats a run bit for bit, step 7 | PARTIAL | Y | new entry points use them, on CPU by default | `tests/test_seeding.py`, `tests/test_runlog.py`, `tests/test_train_ppo.py` | 4–11 |
| C4 | Learning-curve logging | Instr. Code | Legacy `metrics.jsonl` keyed by `global_step` (`metrics_log.py:40-47`). New `RunLogger` writes `config.yaml`, `meta.json` and `metrics.jsonl` keyed by `env_steps` and `manager_decisions` (`src/hrlmpc/utils/runlog.py`), step 4. Flat PPO adds `evaluations.jsonl` (per-start results) and `final.pt`, step 7 | OK | — | keep | `tests/test_runlog.py`, `tests/test_train_ppo.py` | 4, 7 |
| C5 | Reference stack declared | Instr. Code | `pyproject.toml` (package `hrlmpc`, extras `rl`, `mpc`, `plots`, `test`); `requirements.txt` locks all 56 third-party packages at their pre-alignment versions (DL-D5, `requirements-pre-alignment.txt`), step 4 | OK | — | keep | CI install; `pip check` and the full suite in the locked virtual environment | 4 |
| C6 | Model facts DL-F1–F6 checked automatically | DL | `tests/test_model_facts.py` checks M and F1–F6 on `src/hrlmpc/model.py`, and CI runs it (step 4) | OK | — | keep | CI | 4 |

## 7. Inventory of the pre-alignment code

Planned fates are proposals, to be confirmed in step 3. Deleted or archived files remain available at tag `pre-alignment`.

| Path | Role | Planned fate | Step | Notes |
|---|---|---|---|---|
| `scenarios/{slalom,tunnel}/envs/*_env.py`, `vec_*_env.py`, `actuation.py` | Physics and MDP, 4 copies | rewrite | 5–6 | One physics module and one MDP wrapper; scenarios become configs |
| `scenarios/*/envs/width_profile.py` | Geometry | rewrite | 5 | Polytopic geometry; keep a builder for the slalom layout |
| `scenarios/*/envs/config.py` | Env parameters | rewrite | 4–6 | YAML plus a typed dataclass |
| `scenarios/*/envs/spawn_sampler.py` | Initial states | keep | 6 | The Sobol' option had no measurable effect |
| `scenarios/*/envs/visualize_env.py` | Rendering | refactor | 6 | |
| `scenarios/*/envs/tests/` | Env tests | rewrite | 5 | Port the cases that are still valid |
| `algorithms/common.py` | Shared RL utilities (Beta head, normalization, Manager networks) | refactor | 8 | The flat-PPO parts are ported to `src/hrlmpc/ppo.py` (step 7) |
| `algorithms/ppo/` | Flat PPO | **replaced** (step 7); deleted in step 8 (DL-D21) | 7–8 | Ported to `src/hrlmpc/ppo.py` and `train_ppo.py`; kept for the bit-identity test of DL-D19 |
| `algorithms/hppo/` | hPPO | refactor | 8 | Onto the common runner |
| `algorithms/ppo_mpc/` | PPO_MPC | refactor | 8–9 | Its loop is a fork of hPPO's; move it onto the common runner |
| `algorithms/ppo_mpc_reach/`, `scenarios/*/scripts/script_ppo_mpc_reach.py` | Reach arm, disabled and stale | **deleted** (step 4) | 4 | Rebuilt in step 10 |
| `algorithms/tube_mpc.py` | Tube MPC Worker | refactor | 9 | Model module, $W=BD$, inter-sample constraints |
| `algorithms/mpc_worker.py` and its test | Old QP worker, dead code | **deleted** (step 4) | 4 | |
| `algorithms/optimal_solver.py` | Min-time oracle | refactor | 9–11 | Model module, inter-sample constraints |
| `algorithms/study.py`, `study_plots.py`, `solved_check.py`, `metrics_log.py` | Protocol and analysis | refactor | 11 | `src/hrlmpc/evaluation.py` and `analysis.py` start their replacement (step 7) |
| `algorithms/spawn_coverage.py` | Spawn diagnostic | keep or delete | 6 | Decide together with the spawn sampler |
| `scenarios/slalom/scripts/{stress_disturbed,study_init_sampler,probe_start_variance,sweep_extrinsic_coef}.py` | One-off investigations | **deleted** from `main` (step 4) | 4 | |
| `scenarios/*/scripts/script_optimal.py` | Prints the oracle's metrics | **deleted** (step 4) | 4 | Superseded by the study tooling |
| `scenarios/*/scripts/script_*.py`, `study_*.py`, `compare_disturbed.py` | Entry points | refactor | 4, 11 | One CLI driven by configs |
| `docs/*.md` | Investigation reports on the old dynamics | **deleted** from `main` (step 4, DL-D7) | 4 | Index in §8; still at tag `pre-alignment` |
| `docs/stato-ppo-mpc.md` (untracked) | Memo for the supervisors | **merged and moved out** (step 3, DL-D8) | 3 | |
| `scenarios/*/studies/**` | Old results | **untracked** (step 4, DL-D7) | 4 | Still on the author's disk, ignored by git; tracked copies at tag `pre-alignment` |
| `scenarios/*/ENVIRONMENT.md` | Environment description | rewrite | 6 | |
| `pytest.ini` | Test configuration | keep | — | |

## 8. Pre-alignment results (reference only)

All obtained on the pre-alignment environment, so they are not comparable with post-alignment results. Source: `docs/disk-limits-and-effort.md` §9 unless stated otherwise.

**Slalom, disk limits, no disturbance**
- PPO_MPC solved 8/10 seeds within 204 800 steps, with a median first solve of 118k.
- PPO solved 9/20 seeds within 1 024 000 steps, with a median first solve of 748k.
- hPPO solved 0/20 seeds.

**Slalom, disturbed**
- Two levels: $\lVert w_p\rVert\le 0.005$, $\lVert w_v\rVert\le 0.05$, and twice that.
- Seeds with a clean evaluation, at the two levels:

| Arm | 1× | 2× |
|---|---|---|
| PPO_MPC | 10/10 | 10/10 |
| PPO | 6/10 | 4/10 |
| hPPO | 0/10 | 0/10 |

**Tunnel, 204 800 steps**

Every arm solves every seed: PPO_MPC 10/10, hPPO 20/20, PPO 20/20.

**Slalom, box limits (`box-limits-final`), 204 800 steps**

| Arm | Seeds solved | Median first solve |
|---|---|---|
| PPO | 20/20 | 51k |
| hPPO | 20/20 | 72k |
| PPO_MPC | 10/10 | 72k |

**Earlier investigations**

| Topic | Document |
|---|---|
| Worker reward and termination | `docs/worker-termination-avoidance.md` |
| Goal-box saturation of the old PPO_MPC | `docs/goal-box-saturation.md` |
| Reachability regimes | `docs/reachability-regime-study.md` |
| Sobol' starts (no effect) | `docs/init-sampler.md` |
| Benchmark | `docs/benchmark.md` |
| Progression | `docs/progression.md` |

## 9. Change log

| Date | Step | Change |
|---|---|---|
| 2026-10-06 | 1 | The author creates tag `pre-alignment` on `ff50d7c`. |
| 2026-10-06 | 2 | Audit of the pre-alignment code; this file created. |
| 2026-10-06 | 1 | Baseline test run by the author: 470 passed, 2 harmless warnings (§3). Step 1 closed. |
| 2026-10-06 | 2 | Evidence re-checked; two rows corrected (R2, X5: the `--contact-penalty` flag also exists in PPO_MPC) and X7 made precise. Step 2 closed; this file committed on `align/step-02-audit` and merged into `main` (PR #1). |
| 2026-10-06 | 3 | Q1, Q2, Q14, Q15 ratified (DL-D5–D8). Notes from the memo of 2026-09-29 added under §5 for Q11 and Q12. |
| 2026-10-07 | 3 | Q3–Q9 and Q16 ratified (DL-D9–D15); matrix actions updated (M5, M6, M7, R2, R5, AM1, X1). Step 3 closed. |
| 2026-10-07 | 4 | New package `src/hrlmpc` (model, config, seeding, run logging) with YAML configs and tests (M, F1–F6, config, logging, seeding); `pyproject.toml`, pinned `requirements.txt`, `requirements-pre-alignment.txt`; CI on `tests/`; `.gitattributes`. Old results untracked (kept on disk); old docs, dead code and one-off scripts deleted from `main`. Rows C1–C6, E8, X1 updated. In review. |
| 2026-10-07 | 4 | The author adds the CI workflow (`.github/workflows/tests.yml`). Full suite in the new virtual environment: 443 passed, 1 failed. The legacy test `test_render_rgb_array_returns_frame` (tunnel) failed because matplotlib picked its Tk backend and the virtual environment has no usable Tcl/Tk (`tk.tcl` not found); tests now render with the Agg backend (root `conftest.py`). The comment in `pytest.ini` no longer lists the deleted `ppo_mpc_reach` tests. |
| 2026-10-07 | 4 | Full suite after the Agg fix: 444 passed, 16 warnings (2 in step 1). The 14 new ones come from pyparsing 3.3.3, whose deprecated names matplotlib 3.9.2 still calls. Cause: `requirements.txt` pinned only the direct dependencies, so the virtual environment got newer versions of 32 of the 45 transitive packages than the pre-alignment environment. `requirements.txt` now locks all 56 third-party packages at their pre-alignment versions; C5 updated. |
| 2026-10-07 | 4 | Merged into `main`. The author reports the full suite passing in the locked virtual environment and CI green. Step 4 closed; step 5 opened on `align/step-05-physics`. |
| 2026-10-07 | 5 | Tests first, then `src/hrlmpc/geometry.py` (convex polygons in H-representation, layouts from the YAML configs, free space of DL-D16) and `src/hrlmpc/physics.py` (pure batched step, stages E1–E6, first-impact wall reaction of DL-D17). The tests exposed two gaps in the specification, which the author settled as DL-D16 and DL-D17. An independent adversarial review (more than a million random and constructed steps, KKT check of every projection) found no penetration and no phantom contact, and six defects in edge cases, all fixed: obstacles widening away from a wall, a rounding case at the tolerance boundary, wall friction when leaving a face at its vertex, memory with many obstacles, overflow of huge commands, batch-dependent rounding. Layouts with gaps or obstacle widths below 1e-6 m are now rejected. Rows M1–M7 and E1–E8 updated. In review. |
| 2026-10-07 | 5 | Merged into `main`. The author reports 515 tests passing in the locked virtual environment and CI green, and has updated the project instructions to DL-D16 and DL-D17. Step 5 closed; step 6 opened on `align/step-06-mdp`. |
| 2026-10-07 | 6 | Tests first, then `src/hrlmpc/action_map.py` (bijection of DL-D15), `src/hrlmpc/env.py` (`NavigationEnv`: the MDP of DL-D13 on `physics_step`, disturbances of DL-D10, contact penalty of DL-D14, samples of DL-D6, metrics R3–R4) and `src/hrlmpc/gym_env.py` (single-agent Gymnasium adapter, tested where Gymnasium is installed). The author keeps $c_n = 50$, $c_s = 1$ after the calibration of DL-D18. An independent review (2.3 million samples, rewards recomputed independently) found no error in the MDP and a few defects at the edges, all fixed: the adapter's `info` key collided with Gymnasium's `RecordEpisodeStatistics`, a failed reset consumed random numbers, a non-boolean mask, actions a rounding error outside the square, initial positions in the goal region, a lenient adapter, and the overflow of the norm of a command near the largest float in `physics.py`. Rows M1, M5, E7, R1–R5, AM1 and X5 updated. In review. |
| 2026-10-07 | 6 | Merged into `main`. The author reports 557 tests passing in the locked virtual environment and CI green. Step 6 closed; step 7 opened on `align/step-07-ppo`. |
| 2026-10-07 | 7 | Tests first, then `src/hrlmpc/ppo.py` (in-house PPO of DL-D19, bit-identical to the pre-alignment update), `rollout.py` (rollouts; GAE with one discount per transition), `evaluation.py` and `paths.py` (evaluation protocol of DL-D20; minimum-time bound of DL-F7 with the shortest free path), `train_ppo.py` (training loop and command line), `analysis.py` (curves and summaries over seeds), `configs/agent/ppo.yaml`, and a YAML loader that rejects repeated keys. PyTorch was not available where the code was written: the NumPy parts were tested there and the PyTorch code by two independent reviews; its tests run in the author's environment and, after the workflow change of DL-D21, in CI. Acceptance runs pending. |
| 2026-10-08 | 7 | The author reports 728 tests passing in the locked virtual environment. First acceptance runs of DL-D20, with the exact shaping of DL-D13: the tunnel is accepted (every seed succeeds from all 25 starts, maximum arrival gap 1.4%, no contact); the slalom fails (no training episode reaches the goal, and every seed learns to stand still). Diagnosis and decision in DL-D22: the progress term is undiscounted again (`reward.shaping_discount = 1`) and the learners' discount moves to `update.discount` in `configs/agent/ppo.yaml`; the bound of DL-D18 becomes about 48.4 s/m, which $c_n = 50$ still meets. `analysis.py` now also names the layout from Windows paths. The first runs are archived under `runs/archive/step7-exact-shaping/` in the author's clone (not tracked); both layouts are run again. |
