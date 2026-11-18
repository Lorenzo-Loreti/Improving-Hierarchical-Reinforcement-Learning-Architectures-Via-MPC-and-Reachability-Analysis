# hrlmpc

Code of the master's thesis *Improving Hierarchical Reinforcement Learning Architectures via MPC and
Reachability Analysis* (Lorenzo Loreti, Institute for Systems Theory and Automatic Control, University of
Stuttgart, Master Thesis 163, November 2026). The statement on the use of tools in this work, which covers
the code, is Appendix B.6 of the thesis.

The package `hrlmpc` contains the navigation environment, the four architectures that the thesis compares,
the minimum-time oracle and the analysis that writes the result tables and the result figures of the thesis.
Chapters, sections and tables named below are those of the thesis. Every command below is run from this
directory, which is `code/` in the hand-in archive.

| Number | Architecture | Agent file | Training module |
|---|---|---|---|
| 1 | PPO (flat) | `configs/agent/ppo.yaml` | `hrlmpc.train_ppo` |
| 2 | hPPO (learned Manager and learned Worker) | `configs/agent/hppo.yaml` | `hrlmpc.train_hppo` |
| 3 | PPO_MPC (learned Manager, tube-MPC Worker) | `configs/agent/ppo_mpc.yaml` | `hrlmpc.train_ppo_mpc` |
| 4 | PPO_MPC,reach (as 3, targets restricted to the reachable set) | `configs/agent/ppo_mpc_reachability.yaml` | `hrlmpc.train_ppo_mpc` |

The comments of the code call the architectures by these numbers. For three things the code has names of its
own: the curriculum setting of the thesis is `main`, its sparse setting is `exploration`, and its start box is
the *spawn box*. Section 4 of this file has the table.

## 1. Installation

Python 3.12 is required. The results of the thesis were produced with CPython 3.12 on Windows 11.

    python -m venv .venv
    .venv\Scripts\Activate.ps1                  (PowerShell; Linux and macOS: source .venv/bin/activate)
    python -m pip install -r requirements.txt

`requirements.txt` pins every package to the version used for the thesis and installs the package itself in
editable mode. The lock was taken on Windows; on another platform, if a pinned version is not available,
install the compatible ranges of `pyproject.toml` instead:

    python -m pip install -e ".[rl,mpc,plots,test]"

Main dependencies: NumPy, SciPy, PyYAML, PyTorch, Gymnasium, CVXPY with Clarabel, Gurobi (`gurobipy`),
Matplotlib, pytest.

Gurobi needs a licence (the academic licence is free of charge). It is used only by the mixed-integer
problem of the MPC Worker (architectures 3 and 4) and by the oracle. The environment, PPO and hPPO run
without it.

## 2. Tests

    python -m pytest tests -q

A test module whose packages are not installed is skipped, so the environment, the configurations and the
analysis can be tested without PyTorch and without the solvers. `tests/reference/reference_ppo.py` is a
frozen copy of an earlier stand-alone implementation of flat PPO, which the PPO of the package was written to
reproduce: four tests of `tests/test_ppo.py` compare the two, and the update agrees bit for bit.

## 3. Layout of the repository

| Path | Content |
|---|---|
| `src/hrlmpc/model.py`, `physics.py`, `geometry.py` | the model, the environment step with speed limiter, wall reaction and wall friction, the arena and the obstacles (Chapter 3) |
| `src/hrlmpc/env.py`, `gym_env.py`, `curriculum.py` | the navigation task as a Markov decision process, its two rewards, the parallel agents, the Gymnasium wrapper, the reverse curriculum on the starts (Sections 3.3 and 5.2) |
| `src/hrlmpc/action_map.py`, `ppo.py`, `rollout.py` | the map from the square of actions to the disk of inputs, PPO with Beta policies, rollouts and generalized advantage estimation (Sections 2.1 and 4.1) |
| `src/hrlmpc/hierarchy.py` | the Manager–Worker loop, the target map, the learned Worker's input and reward (Section 4.2) |
| `src/hrlmpc/tube.py`, `mpc_problem.py`, `mpc_worker.py`, `mpc_command.py` | the tube and the tightened constraints, the Worker's problem, its solution with Gurobi and Clarabel, the fallbacks and the statistics of the searches (Section 4.3, Appendix B.3) |
| `src/hrlmpc/reachability.py` | the reachable map of architecture 4: support points, hull, radial map, projection (Section 4.4) |
| `src/hrlmpc/oracle.py`, `paths.py` | the minimum-time oracle and the lower bound on the arrival time (Section 3.4) |
| `src/hrlmpc/evaluation.py` | the evaluation from the fixed grid of starts (Section 5.3) |
| `src/hrlmpc/train_ppo.py`, `train_hppo.py`, `train_ppo_mpc.py` | the training loops and their command lines |
| `src/hrlmpc/analysis.py`, `results.py` | learning curves and summaries of runs; the statistics, tables and figures of Chapter 6 and Appendix B.4 |
| `src/hrlmpc/reevaluate.py`, `trajectories.py` | the escape evaluation of the trap and the recorded episodes of Figure 6.4 |
| `src/hrlmpc/config.py`, `ppo_config.py`, `hppo_config.py`, `ppo_mpc_config.py`, `utils/` | typed configurations loaded from YAML, the run directories and their logs, seeding |
| `configs/` | every configuration of the thesis (next section) |
| `tests/` | the unit tests |

## 4. Configuration files

**Environment**, `configs/env/<layout><setting><level>.yaml`, with the layouts `slalom`, `trap` and `tunnel`:

| File name | Setting of the thesis | Name in the code |
|---|---|---|
| `<layout>.yaml` | dense: dense reward, starts in the start box | `dense` |
| `<layout>_sparse.yaml` | curriculum: sparse reward, starts from the reverse curriculum | `main` |
| `<layout>_sparse_nocur.yaml` | sparse: sparse reward, starts in the start box | `exploration` |

The suffixes `_d05` and `_d10` set the disturbance bound to 0.5 and 1.0 m/s²; without a suffix there is no
disturbance. The code calls the start box of the thesis the *spawn box*. Five files complete the grid of
layouts, settings and levels and are used by the tests, not by the experiments of the thesis: `slalom_d05.yaml`,
`tunnel_d05.yaml`, `tunnel_sparse.yaml`, `tunnel_sparse_d05.yaml` and `tunnel_sparse_d10.yaml`.

**Agent**, `configs/agent/`: the four files of the table above with the hyperparameters selected by the
tuning (Table B.1); `*_h6.yaml` and `*_h4.yaml` with segments of six and four steps (the agility sweep);
`hppo_map.yaml` with the map-aware Worker (the ablation of Section 6.6); `tuning/<architecture>/c1.yaml` to
`c8.yaml`, the eight configurations of the tuning of each architecture (Appendix B.2).

**Oracle**, `configs/oracle/*.json`: the minimum times from the 25 evaluation starts of each layout. They
are shipped; `python -m hrlmpc.oracle --env configs/env/slalom.yaml` computes a table again (Gurobi). A table
is found by its content, not by its name, so one table serves every setting and level of its layout; the
trap's is named `trap_sparse.json` after the file it was computed from.

Symbols that differ between the thesis and the docstrings of the code:

| In the thesis | In the code |
|---|---|
| start box | spawn box, `task.spawn` |
| PPO_MPC,reach | `ppo_mpc_reachability`, architecture 4 |
| goal bonus c_g, wall friction γ_wall | `success_bonus`, `gamma_w` |
| weights q_g, q_e, q_0 of the Worker's cost | `q_g`, `r_e`, `eps` (`target_weight`, `effort_weight`, `start_weight`) |
| margin ρ_wall | `rho` |
| n_stop, the steps needed to stop | `n_b`, `stopping_steps` |
| n_Z, μ, ε_E of the tube | `s` (`num_terms`), `lambda`, `eps` |
| tightened bounds v_max − r_v′ and a_max − r_u | `v_bar`, `u_bar` |

## 5. One training run

    python -m hrlmpc.train_ppo     --env configs/env/slalom_sparse.yaml --agent configs/agent/ppo.yaml     --budget 204800 --seeds 1 --runs-dir runs/final
    python -m hrlmpc.train_hppo    --env configs/env/slalom_sparse.yaml --agent configs/agent/hppo.yaml    --budget 204800 --seeds 1 --runs-dir runs/final
    python -m hrlmpc.train_ppo_mpc --env configs/env/slalom_sparse.yaml --agent configs/agent/ppo_mpc.yaml --budget 204800 --seeds 1 --runs-dir runs/final

`--budget` is the number of samples (environment steps summed over the eight parallel agents) and `--seeds`
takes one or more seeds, one run each. A run writes the directory
`<runs-dir>/<environment file>/<architecture>/seed<k>-<time>/` with `config.yaml` (the resolved configuration),
`meta.json` (versions and machine), `metrics.jsonl` (the learning curves), `evaluations.jsonl` (the result of
every evaluation from every start) and `final.pt` (the final policy).

A run is determined by its seed and its configuration. For runs that repeat to the bit, and for comparable
run times, the numerical libraries are kept on one thread:

    $env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"      (PowerShell)
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1                     (Linux and macOS)

On the laptop named in Section 5.4, a run of PPO or hPPO takes two to three minutes; a run with the MPC
Worker takes one to two hours on the tunnel, about four hours on the slalom and the trap, and about six
hours with a disturbance (Table 6.6). A short check of an installation is a run with `--budget 10240`.

## 6. The experiments of the thesis

All 980 runs of Table 5.3 use the seeds 1 to 10 and the budget 204 800, except the doubled budget.

| Experiments of Table 5.3 | Environment files | Agent files | `--runs-dir` | Runs |
|---|---|---|---|---|
| Slalom and trap, three settings, no disturbance | `slalom.yaml`, `slalom_sparse.yaml`, `slalom_sparse_nocur.yaml`, and the same for `trap` | the four architectures | `runs/final` | 240 |
| Tunnel, sparse and dense settings | `tunnel_sparse_nocur.yaml`, `tunnel.yaml` | the four architectures | `runs/final` | 80 |
| Disturbance 0.5 on the slalom, curriculum and sparse settings | `slalom_sparse_d05.yaml`, `slalom_sparse_nocur_d05.yaml` | the four architectures | `runs/final` | 80 |
| Disturbance 1.0, the eight settings of the first two rows | the eight files of the first two rows with `_d10` | the four architectures | `runs/final` | 320 |
| Agility sweep on the slalom, curriculum and sparse settings | `slalom_sparse.yaml`, `slalom_sparse_nocur.yaml` | `hppo`, `ppo_mpc`, `ppo_mpc_reachability` with `_h6` and `_h4` | `runs/final_h6`, `runs/final_h4` | 120 |
| Doubled budget on the trap (`--budget 409600`) | `trap_sparse.yaml` | the four architectures | `runs/final_budget2` | 40 |
| Ablation of hPPO | the eight files of the first two rows, `slalom_sparse_d10.yaml` and `trap_sparse_d10.yaml` | `hppo_map.yaml` | `runs/final_map` | 100 |

The tuning (Appendix B.2) runs the eight files of `configs/agent/tuning/<architecture>/` on
`configs/env/slalom_sparse.yaml` with `--budget 204800 --seeds 101 102 103` and
`--runs-dir runs/tuning/<architecture>`. `python -m hrlmpc.analysis runs/tuning/<architecture>` then prints the
ranking of the eight configurations by the mean area under the success curve, on which the selection rests.

The escape evaluation of the trap (Section 5.3, Table 6.3) evaluates final policies from two grids of starts
and adds its result to each run directory:

    python -m hrlmpc.reevaluate --name escape --box 4.0 5.0 -0.4 0.4 --grid 3 3 runs/final/trap runs/final/trap_sparse runs/final/trap_sparse_nocur
    python -m hrlmpc.reevaluate --name mouth  --box 3.0 3.7 -0.7 0.7 --grid 3 3 runs/final/trap runs/final/trap_sparse runs/final/trap_sparse_nocur

The episodes of Figure 6.4 are recorded with `python -m hrlmpc.trajectories <run directory>` for one run of
each architecture on `slalom_sparse` and on `trap_sparse`: the run whose final success rate is closest to the
median of its cell, as the caption of the figure says. The command of Section 7 of this file draws the replays
that it finds below the run directories.

## 7. Tables and figures

    python -m hrlmpc.results runs/final runs/final_h6 runs/final_h4 runs/final_map runs/final_budget2 --out results

reads the run directories and writes `results/summary.json` (every number), `results/report.md` (the same,
readable), `results/tables/*.tex` (the bodies of the result tables of the thesis, which are these files unchanged)
and `results/figures/*.pdf`. With `--no-figures` only the tables and the report are written.

    python -m hrlmpc.analysis runs/final/slalom_sparse --inventory
    python -m hrlmpc.analysis runs/final/slalom_sparse/ppo runs/final/slalom_sparse/hppo --plot curves.png --action mode

list and summarize any set of runs and plot their learning curves.

## 8. Data

The logs of the 980 runs (about 800 MB) are not part of this archive. They are available from the author
on request; with them, the command of Section 7 of this file reproduces every result table and result figure
of the thesis without training.
