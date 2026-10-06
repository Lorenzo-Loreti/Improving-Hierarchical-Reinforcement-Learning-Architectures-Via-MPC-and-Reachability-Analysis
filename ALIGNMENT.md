# Alignment tracker

Living record of the alignment of this repository with the thesis specification. Read it before starting a step and update it in the same pull request that changes the code.

- **Target ("spec v1").** Problem setup, architectures and rules in the Claude project's instructions, plus the design decisions in its decision log (`claude/decision-log.md`), as of 2026-10-06.
- **Starting point.** Tag `pre-alignment` (commit `ff50d7c`).
- **Last updated.** 2026-10-06: steps 1 and 2 closed; next is step 3.

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
| 3 | Ratify implicit choices; close the decisions that block steps 4–7 | Q1–Q9 and Q14–Q16 closed in the decision log | Not started | | |
| 4 | Minimal infrastructure | Package layout, pinned dependencies, YAML configs, seeding, logging against the sample counter, pytest and CI, DL-F1–F6 as tests, cleanup of dead code and old results | Not started | | Rows C1–C6, E8 |
| 5 | Physics module, tests first | One pure, batched step function implementing E1–E6 on polytopic geometry; tests of rows M and E | Not started | | Q3–Q6, Q16 |
| 6 | MDP wrapper | Observation, goal, reward with the contact penalty, termination, disturbance sampling, action map, metrics R3–R4 | Not started | | Q7–Q9 |
| 7 | Flat PPO on the new environment | Learns on a trivial layout; becomes the reference for logging and configs | Not started | | |
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
- **Dependencies are not declared** (no `requirements.txt`, no `pyproject.toml`). What the code imports:
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
| Q1 | Reference stack (C5) | numpy, scipy, PyTorch (own PPO), Gymnasium, cvxpy + Clarabel, gurobipy, matplotlib; global Python 3.12 interpreter on Windows; nothing pinned | Keep it, since changing libraries needs a reason; declare it and pin versions in a virtual environment. Before upgrading anything, record the versions of the pre-alignment environment (`python -m pip freeze`). Decide whether running every architecture may require a Gurobi licence (today it does, even on the tunnel) | 4 | open |
| Q2 | Definition of a sample (X1) | Environment steps summed over the parallel envs; evaluation, oracle and MPC predictions not counted | Ratify. One sample is one call of the physical step during training. Manager decisions are logged as a secondary counter. MPC model predictions are not samples but are reported as compute time | 4 | open |
| Q3 | Numerical parameters (M7) | $T_s = 0.1$ s, $a_{\max} = 2.5$, $v_{\max} = 1.2$, no friction | Choose $\gamma$ with $v_{\max} < a_{\max}/\gamma$ ($\gamma < 2.08\ \mathrm{s^{-1}}$ with these values), choose $\gamma_w$, and choose $\bar d$ satisfying DL-F2 if robust invariance of $V$ is wanted. The current 1× disturbance (0.005 m, 0.05 m/s) is the product of the disks of radii $T_s^2 \bar d$ and $T_s \bar d$ with $\bar d = 0.5\ \mathrm{m/s^2}$ | 5 | open |
| Q4 | Distribution of $d_k$ in $D$ (M5) | Uniform on two independent disks for $w_p$ and $w_v$ | $d_k$ i.i.d., uniform in area on $D$; worst-case (boundary) disturbances only for stress tests | 5 | open |
| Q5 | Geometry and layouts (M6) | Width profile; fixed slalom, tunnel as a sanity check | One geometry module (H-representation of the arena and the obstacles) shared by the env, the MPC, the oracle and the reachability code; the slalom as the fixed main layout | 5 | open |
| Q6 | Arena boundary (M6, DL-A2) | $p_x \in [-1, L+1]$ enforced by a position clip without contact; goal at $p_x \ge L$ | A closed arena polygon whose faces are all physical walls, with the goal region strictly inside it | 5 | open |
| Q7 | MDP elements the spec does not define (R5) | Observation = normalized $(p, v)$; success when $p_x \ge 10$; horizon 200; spawn uniform on $[0,2]\times[-1,1]$ at rest; reward −1 per step, +1000 on success (replacing the step's other terms), progress $10\,\Delta p_x$ cut at $L$, effort $-0.01\lVert u\rVert^2/a_{\max}^2$ on the delivered input | Ratify explicitly (with any changes) in the decision log, including whether the success bonus should replace or add to the other terms of its step | 6 | open |
| Q8 | Form of the contact penalty (R2) | Flat −50 per contact step | Price the "free brake" that motivates DL-D1: a term proportional to the cancelled normal velocity, plus an optional small per-step term for sliding. Use the same function for every architecture, inside the hPPO Worker reward too | 6 | open |
| Q9 | Action map (AM1) | Beta per axis on the box, then radial projection onto $U$ in the env | The spec text assumes a Gaussian policy. Either amend it to the Beta, a bounded-support option it already lists, or switch to a Gaussian. Keep the Beta, but replace the many-to-one box→disk projection with a bijection of the square onto the disk (e.g. $x \mapsto (\lVert x\rVert_\infty/\lVert x\rVert_2)\,x$), so that physical corrections come only from the speed limiter and the walls | 6–7 | open |
| Q10 | Manager/Worker interface (AR5) | $H = 10$; targets: hPPO ±10 m box, PPO_MPC ±1.8 m box, reach arm 4-D; Worker reward = progress + 0.02 × env reward; no Worker termination | One target space (relative position, sized to the $H$-step reach) shared by AR2–AR4 and restricted only in AR4; the common contact penalty in the Worker reward | 8 | open |
| Q11 | MPC formulation (AR3) | Tube MPC, big-M MIQP + Clarabel, $W$ on $(p, v)$, no inter-sample constraints | Keep the tube MPC but have it read the shared model module. Use $W = BD$, or a declared outer bound $W' \supseteq BD$. Exclude corner cutting (one face per segment, or inflated obstacles). Decide whether to tighten $V$ or rely on DL-F3 | 9 | open |
| Q12 | Reachability (AR4) | Disabled and stale | Use the robust $H$-step reachable set of target positions under the tube MPC's tightened constraints. Check the inner approximation against DL-F5/F6 in the obstacle-free case. Define "reached" as being within the tube cross-section around the target | 10 | open |
| Q13 | Experimental protocol (X2–X7) | Unequal budgets and seeds; no tuning protocol; median/IQR; Mann–Whitney with unsolved seeds ranked $+\infty$; solved-check against the undisturbed oracle | Equal budget and seed list per comparison. A declared tuning protocol on held-out seeds. The sample-efficiency metric fixed in advance: first-solve time with censoring handled by survival analysis, or the area under the curve. Means with bootstrap CIs and a correction for multiple comparisons. A disturbed reference for disturbed runs | 11 | open |
| Q14 | Old results tracked under `studies/` | About 2 350 tracked files from the old dynamics | Remove them from `main` in step 4 (they stay in the tag) so they cannot be mixed with new results; keep the index in §8 | 4 | open |
| Q15 | `docs/stato-ppo-mpc.md` | Untracked, Italian, partly stale | Move what is still valid (tube-MPC deviations, reachability options, questions for the supervisors) into an English doc or the decision log, then drop the file | 3 | open |
| Q16 | Tunnel scenario | A full code copy of the slalom | Keep it only as a layout/config of the single environment, or drop it | 5 | open |

## 6. Traceability matrix

Sources: "Instr." is a section of the project instructions; "DL-" is a decision-log entry. The evidence cites the code at tag `pre-alignment`; paths are relative to `scenarios/slalom/envs/`, `algorithms/` or `scenarios/slalom/scripts/` unless given in full. The tunnel copies have the same structure. Step = the workflow step that closes the row.

### Model and physics (M)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| M1 | State $x=(p,v)\in\mathbb{R}^4$; input $u\in\mathbb{R}^2$ = commanded acceleration; unit point mass | Instr. Setup | Observation $[p_x,p_y,v_x,v_y]$, action $[a_x,a_y]$ (`slalom_env.py:61-75`) | OK | — | keep | env API test | 5–6 |
| M2 | Isotropic viscous friction $\gamma>0$ | DL-A1 | Absent: the velocity block of $A$ is the identity (`slalom_env.py:79-84`, `tube_mpc.py:635-638`, `mpc_worker.py:81-92`) | CONFLICT | S | rewrite | model test: matrices vs recursion | 5 |
| M3 | Semi-implicit Euler: $v^+=(1-\gamma T_s)v+T_s(u+d)$, $p^+=p+T_s v^+$ | DL-D4 | Exact ZOH of the frictionless double integrator, $B=[\tfrac12 T_s^2 I;\ T_s I]$ (`slalom_env.py:86-91`, `vec_slalom_env.py:57-69`, `tube_mpc.py:639-642`; the oracle reads `env.A`, `env.B` at `optimal_solver.py:305-306`) | CONFLICT | S | rewrite: one model module for env, MPC, oracle and reachability | DL check "M" (compact form) | 5 |
| M4 | $U$, $V$, $D$ are Euclidean balls | DL-A5 | $U$ and $V$ are disks (`actuation.py:85-108`); the action space is the box $[-a_{\max},a_{\max}]^2$ (`slalom_env.py:75`); $D$ is not an acceleration ball (M5) | PARTIAL | S | refactor | — | 5 |
| M5 | Disturbance at acceleration level, $w=Bd$, $d\in D$ | DL-D2 | $w=(w_p,w_v)$ uniform on two independent disks (`actuation.py:111-126`, `config.py:19-40`); the tube MPC uses the same product set (`tube_mpc.py:664`) | CONFLICT | S | rewrite (env), refactor (MPC) | disturbance enters only through $B$; DL-F3 | 5, 9 |
| M6 | Arena polytope with polytopic obstacles; every face is a physical wall | Instr. Setup, DL-A2 | Piecewise-constant width profile (`width_profile.py`); $p_x$ confined to $[-1,L+1]$ by a position clip without contact (`slalom_env.py:65-66, 189`); the tube MPC turns the profile into 4 rectangles (`tube_mpc.py:534-568`) | PARTIAL | S | rewrite the geometry as polytopes | non-penetration on random polygons | 5 |
| M7 | $v_{\max}<a_{\max}/\gamma$ (DL-F1); robust invariance of $V$ (DL-F2) | DL-F1, DL-F2 | No $\gamma$; $T_s=0.1$, $a_{\max}=2.5$, $v_{\max}=1.2$ (`config.py:11-18`); these conditions are not validated | MISSING | S | decide (Q3); validate in the config | config validation test | 3, 5 |

### Environment step (E)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| E1 | Commanded action saturated radially onto $U$ | Instr. step 1 | `project_disk` (`actuation.py:85-93, 102`) | OK | — | keep, move into the physics module | `test_action_is_saturated_radially_before_applied` | 5 |
| E2 | Candidate velocity $\tilde v=(1-\gamma T_s)v+T_s(u+d)$ | Instr. step 2, DL-D3 | Nominal $v+T_s u$ without friction; the disturbance is added after the limiter (`slalom_env.py:172-183`) | CONFLICT | S | rewrite | step equals the model when no correction acts | 5 |
| E3 | Speed limiter $\hat v=\Pi_V(\tilde v)$, before the wall reaction | Instr. step 3, DL-A4, DL-D3 | Limiter on the nominal velocity, by replacing $u$ (`actuation.py:96-108`); a guard projection after the disturbance (`slalom_env.py:190`) | PARTIAL | S | rewrite | $\lVert v\rVert_2\le v_{\max}$ for random states, inputs, disturbances | 5 |
| E4 | Wall reaction: Euclidean projection of $\hat v$ onto the half-planes of the faces crossed by $[p_k,p_k+T_s\hat v]$; continuous collision detection, repeated | Instr. step 4, DL-A4 | The end position is clamped and the normal velocity zeroed, checked at the end point only; the crossing point only classifies which wall was hit (`slalom_env.py:230-244`, `vec_slalom_env.py:164-181`). A step that cuts a corner between samples is not detected | CONFLICT | S | rewrite | $[p_k,p_{k+1}]\subset P_{\mathrm{free}}$; corners; no phantom contacts; projection vs a reference QP | 5 |
| E5 | Wall friction $\gamma_w$ on the sliding velocity | Instr. step 5, DL-A3 | Absent | MISSING | S | implement | sliding decay factor; $\gamma_w=0$ leaves ambient friction only | 5 |
| E6 | $p_{k+1}=p_k+T_s v_{k+1}$ | Instr. step 6 | $p^+=p+T_s v+\tfrac12 T_s^2 u$ (`slalom_env.py:183`) | CONFLICT | S | rewrite | as M3 | 5 |
| E7 | One module implements the physical step for all architectures | Instr. Code | Four implementations (`slalom_env.py`, `vec_slalom_env.py`, `tunnel_env.py`, `vec_tunnel_env.py`) and two `actuation.py`; the plant is re-implemented in `stress_disturbed.py:168`; $A$, $B$ are hard-coded in `tube_mpc.py:635-642` and `mpc_worker.py:81-92` | CONFLICT | S | rewrite: one batched step function, the scalar env being a batch of one | single import site; scalar = batched | 5 |
| E8 | Tests on non-penetration, speed limit, contact at corners, wall friction | Instr. Code | Tests cover radial saturation, the speed limit, gate faces and a corner hit (`envs/tests/test_slalom_env.py:80-704`); none cover friction, penetration between samples or DL-F1–F6 | PARTIAL | F | extend | — | 4–5 |

### Contact, reward and metrics (R)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| R1 | Contact allowed and non-terminal | DL-D1 | The episode continues after a contact (`slalom_env.py:202-249`) | OK | — | keep | — | 6 |
| R2 | Same contact penalty for every architecture, also in the hPPO Worker reward | DL-D1 | Flat −50 per contact step (`config.py:78`). PPO and the Managers see it through the env reward. The hPPO Worker sees 0.02 × it through the extrinsic mix (`hppo_train.py:1123-1124`), and `--worker-extrinsic-coef 0` removes it. A `--contact-penalty` flag exists in hPPO and PPO_MPC (`hppo_train.py:94`, `ppo_mpc_train.py:106`) but not in flat PPO | PARTIAL | S | refactor (Q8): penalty defined once, with a fixed weight in the Worker reward | same penalty across arms (config test) | 6, 8 |
| R3 | Contact metrics: number, duration, cancelled normal velocity | Instr. Metrics | `collision_count` counts contact steps, and `collision_impacts` stores the constant penalty (`vec_slalom_env.py:206-208`); no normal velocity | PARTIAL | F | rewrite (env info: events, duration, $\Delta v_n$) | metric unit tests | 6 |
| R4 | Frequency and magnitude of the physical corrections of the commanded action | Instr. Metrics | Not logged; `delivered_action` appears only in the scalar env's info (`slalom_env.py:284`), not in the vector env's (`vec_slalom_env.py:227-234`) | MISSING | F | implement | metric unit tests | 6 |
| R5 | MDP elements the spec does not define | — | Observation = normalized $(p,v)$ (`common.py:61`); success when $p_x\ge L$ (`slalom_env.py:254`); horizon 200 (`config.py:53`); spawn box, uniform, at rest (`spawn_sampler.py:96-101`); reward terms (`config.py:55-118`); fixed layout (`width_profile.py:139-196`) | IMPLICIT | S | ratify (Q7) | — | 3 |

### Action map (AM)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| AM1 | Map from policy output to $U$ identical for PPO and the hPPO Worker; its distortion known and monitored | Instr. RL action space | Beta per axis on the box (`common.py:191-224`, `ppo/ppo.py:217`, `hppo/hppo.py:654`), then radial projection onto $U$ in the env; the two arms share the map (`hppo/tests/test_hppo.py:592`); the spec text assumes a Gaussian | PARTIAL | S | decide (Q9) and document | correction-rate metric | 3, 6 |

### Architectures (AR)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| AR1 | PPO: the policy outputs the commanded acceleration | Instr. Arch. 1 | `ppo/ppo_train.py:394-401` | OK | — | keep; port to the new env | smoke run on a trivial layout | 7 |
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
| X1 | One definition of "sample" for all architectures | Instr. Fair comparison | Environment steps everywhere (`ppo/ppo_train.py:391`, `hppo/hppo_train.py:1067`, `ppo_mpc/ppo_mpc_train.py:543`), but not declared | PARTIAL | Y | declare (Q2) | — | 3–4 |
| X2 | Same interaction budget | Instr. Fair comparison | Defaults are equal (`study.py:111`), but the reported slalom comparison runs PPO and hPPO for 1 024 000 steps and PPO_MPC for 204 800 (`study_ppo_mpc.py:17-19`); early stop on solve is on by default (`ppo/ppo_train.py:111`) | CONFLICT | S | enforce in the comparison tooling | comparison refuses unequal budgets | 11 |
| X3 | Same seeds, at least 5 per configuration | Instr. Fair comparison | Same integer seeds; 20 for PPO and hPPO (`study.py:191`) vs 10 for PPO_MPC (`study_ppo_mpc.py:6`) | PARTIAL | S | one seed list per comparison | — | 11 |
| X4 | Comparable tuning effort | Instr. Fair comparison | No protocol; only hPPO's Worker coefficient was swept (`sweep_extrinsic_coef.py:161`); PPO_MPC "has not been re-swept" (`ppo_mpc/ppo_mpc_train.py:176`) | MISSING | S | define (Q13) | — | 11 |
| X5 | Same physics, contact penalty and action map | Instr. Fair comparison | Every arm uses the same env classes and map (`script_ppo_mpc.py:29-37` vs `script_ppo.py:20-31`); hPPO and PPO_MPC can override the penalty from the command line, flat PPO cannot | PARTIAL | S | single config source | config test | 6 |
| X6 | Metrics: sample efficiency (defined operationally), success rate, contacts, corrections, final return, compute time | Instr. Metrics | First solve = the second consecutive solved-check within 5 of the oracle at the 25 grid starts, checked every 10 240 steps (`ppo/ppo_train.py:118-130`, `solved_check.py:98`). Success rate is loaded but not plotted (`study.py:464`). MPC solve time is logged only in training | PARTIAL | S | define (Q13) | — | 11 |
| X7 | Mean with confidence intervals; appropriate tests | Instr. Code | Median + IQR (`study_plots.py:161`); Mann–Whitney with unsolved seeds = $+\infty$ (`study_plots.py:210-216`); no CIs in the comparison tooling (a bootstrap interval only in the one-off `study_init_sampler.py:331`); no correction for multiple comparisons | PARTIAL | S | rewrite the analysis | analysis unit tests | 11 |

### Code (C)

| ID | Requirement | Source | Pre-alignment code (evidence) | Status | Class | Action | Verified by (planned) | Step |
|---|---|---|---|---|---|---|---|---|
| C1 | Modular, typed Python; English docstrings, comments, identifiers | Instr. Code | No type hints in the RL code; partial docstrings; English throughout except the untracked Italian memo; monolithic training loops (`hppo/hppo_train.py`: 1 485 lines) | PARTIAL | Y | refactor each file when touched | type checker in CI | all |
| C2 | External configuration files (YAML or similar) | Instr. Code | argparse defaults and dataclasses; `config.json` per run, without the environment parameters (`metrics_log.py:35-37`) | MISSING | Y | YAML configs | config round-trip test | 4 |
| C3 | Fixed seeds, reproducible experiments | Instr. Code | RNGs seeded (`ppo/ppo_train.py:271-274`); `--cuda` on by default (`:89`) with no deterministic algorithms; neither the git hash nor package versions recorded | PARTIAL | Y | refactor | rerun determinism test | 4 |
| C4 | Learning-curve logging | Instr. Code | `metrics.jsonl` keyed by `global_step` (`metrics_log.py:40-47`); W&B optional; the reach script logs only to W&B | OK | — | keep | — | 4 |
| C5 | Reference stack declared | Instr. Code | No requirements file and no `pyproject.toml`; the stack can only be inferred from the imports (§3) | MISSING | Y | declare and pin (Q1) | CI install | 4 |
| C6 | Model facts DL-F1–F6 checked automatically | DL | Only in `claude/verify_setup.py`, in the Claude project | MISSING | F | port into the tests | — | 4 |

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
| `algorithms/common.py` | Shared RL utilities (Beta head, normalization, Manager networks) | refactor | 7–8 | |
| `algorithms/ppo/` | Flat PPO | refactor | 7 | |
| `algorithms/hppo/` | hPPO | refactor | 8 | Onto the common runner |
| `algorithms/ppo_mpc/` | PPO_MPC | refactor | 8–9 | Its loop is a fork of hPPO's; move it onto the common runner |
| `algorithms/ppo_mpc_reach/`, `scenarios/*/scripts/script_ppo_mpc_reach.py` | Reach arm, disabled and stale | delete | 4 | Rebuilt in step 10 |
| `algorithms/tube_mpc.py` | Tube MPC Worker | refactor | 9 | Model module, $W=BD$, inter-sample constraints |
| `algorithms/mpc_worker.py` and its test | Old QP worker, dead code | delete | 4 | |
| `algorithms/optimal_solver.py` | Min-time oracle | refactor | 9–11 | Model module, inter-sample constraints |
| `algorithms/study.py`, `study_plots.py`, `solved_check.py`, `metrics_log.py` | Protocol and analysis | refactor | 11 | |
| `algorithms/spawn_coverage.py` | Spawn diagnostic | keep or delete | 6 | Decide together with the spawn sampler |
| `scenarios/slalom/scripts/{stress_disturbed,study_init_sampler,probe_start_variance,sweep_extrinsic_coef}.py` | One-off investigations | delete from `main` | 4 | |
| `scenarios/*/scripts/script_optimal.py` | Prints the oracle's metrics | delete | 4 | Superseded by the study tooling |
| `scenarios/*/scripts/script_*.py`, `study_*.py`, `compare_disturbed.py` | Entry points | refactor | 4, 11 | One CLI driven by configs |
| `docs/*.md` | Investigation reports on the old dynamics | archive | 4 | Index in §8 |
| `docs/stato-ppo-mpc.md` (untracked) | Memo for the supervisors | merge and drop | 3 | Q15 |
| `scenarios/*/studies/**` | Old results | remove from `main` | 4 | Q14 |
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
| 2026-10-06 | 2 | Evidence re-checked; two rows corrected (R2, X5: the `--contact-penalty` flag also exists in PPO_MPC) and X7 made precise. Step 2 closed; this file committed on `align/step-02-audit`. |
