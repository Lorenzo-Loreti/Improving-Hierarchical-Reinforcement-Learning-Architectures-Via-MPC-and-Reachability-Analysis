"""The MPC Worker's interface to the common loop, and its statistics, without solvers (decision log D34).

:class:`MPCCommand` turns any :class:`StepActor` -- the solver-based
:class:`hrlmpc.mpc_worker.MPCWorker`, or a stand-in in tests -- into the
:class:`~hrlmpc.hierarchy.CommandWorker` of the common loop, and records each
step's outcome, solve time, free binaries, tube ratio, cut rounds, shifted
plan outside its tube and solver failures. The summaries of
:func:`mpc_metrics` are logged in training and evaluation (row AR3e of
``ALIGNMENT.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.env import ObservationMap
from hrlmpc.hierarchy import CommandWorker

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SOLVER, CANDIDATE, EMERGENCY, KEPT = 0, 1, 2, 3
"""Outcomes of a step: the solvers' plan; the shifted previous plan because the solvers found none;
no plan at all (a brake); the shifted previous plan because it was cheaper than the solvers' plan."""


@dataclass(frozen=True)
class WorkerStep:
    """What one step of one agent reports.

    Attributes:
        outcome: :data:`SOLVER`, :data:`CANDIDATE`, :data:`EMERGENCY` or :data:`KEPT`.
        solve_ms: Wall-clock time of the step [ms].
        free_binaries: Binaries the reachable positions left free.
        tube_ratio: ``max(||e_p|| / r_p, ||e_v|| / r_v)`` for the error to the
            last plan's prediction; at most 1 while the disturbance stays in
            ``D`` (a necessary condition for ``e`` in ``Z``). NaN without a plan or a tube.
        cost: The plan's cost, NaN in an emergency.
        cut_rounds: Mixed-integer solves the step took; 0 when Clarabel solved alone.
        plan: The nominal states of the applied plan, or ``None`` in an emergency.
        invalid_candidate: 1 if the shifted plan would have been applied
            (cheaper than the solvers' plan, or the only plan) but the state had
            left its tube, after a disturbance outside ``D``: it was then no
            benchmark, and applied only as the last resort; 0 otherwise.
        solver_errors: Solver failures caught during the step (exceptions, or
            results the step could not use).
    """

    outcome: int
    solve_ms: float
    free_binaries: int
    tube_ratio: float
    cost: float
    cut_rounds: int
    plan: FloatArray | None
    invalid_candidate: int = 0
    solver_errors: int = 0


class StepActor(Protocol):
    """What :class:`MPCCommand` needs of a Worker: one step per agent (slot), and a reset per slot."""

    @property
    def num_slots(self) -> int:
        """The number of agents it can steer."""
        ...

    def act(self, x: ArrayLike, target: ArrayLike, slot: int = 0) -> tuple[FloatArray, WorkerStep]:
        """The input at state ``x`` toward rest at ``target``, and the step's report."""
        ...

    def reset(self, slot: int | None = None) -> None:
        """Forget the memory of ``slot``, or of every slot."""
        ...


STAT_KEYS = ("outcome", "solve_ms", "free_binaries", "tube_ratio", "cut_rounds", "invalid_candidate", "solver_errors")


class MPCCommand:
    """The Worker as a :class:`~hrlmpc.hierarchy.CommandWorker` of the common loop, with step statistics.

    Each call steers every agent to its target with its own slot; the
    statistics of the steps since the last :meth:`pop_stats` are kept, one
    row per call and one column per agent.

    Args:
        worker: The Worker, with at least one slot per agent.
        obs_map: The observation map, to read the velocities.
    """

    def __init__(self, worker: StepActor, obs_map: ObservationMap) -> None:
        self.worker = worker
        self.obs_map = obs_map
        self._stats: dict[str, list[FloatArray]] = {key: [] for key in STAT_KEYS}

    def command(self, obs: FloatArray, positions: FloatArray, targets: FloatArray) -> FloatArray:
        """The commanded accelerations of every agent, ``(B, 2)``."""
        velocities = self.obs_map.velocities(obs)
        n = len(positions)
        if n > self.worker.num_slots:
            raise ValueError(f"{n} agents but only {self.worker.num_slots} slots")
        u = np.empty((n, 2))
        row = {key: np.empty(n) for key in STAT_KEYS}
        for i in range(n):
            x = np.concatenate([positions[i], velocities[i]])
            u[i], info = self.worker.act(x, targets[i], slot=i)
            for key in STAT_KEYS:
                row[key][i] = getattr(info, key)
        for key in STAT_KEYS:
            self._stats[key].append(row[key])
        return u

    def reset(self, mask: BoolArray) -> None:
        """Forget the plans of the agents whose episode just ended."""
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            self.worker.reset(int(i))

    def pop_stats(self) -> dict[str, FloatArray]:
        """The statistics since the last call, each ``(steps, agents)``, and clear them."""
        result = {key: np.array(rows) for key, rows in self._stats.items()}
        self._stats = {key: [] for key in STAT_KEYS}
        return result

    def as_worker(self) -> CommandWorker:
        """This adapter as the Worker of :mod:`hrlmpc.hierarchy`."""
        return CommandWorker(command=self.command, reset=self.reset)


def mpc_metrics(stats: dict[str, FloatArray], prefix: str, mask: BoolArray | None = None) -> dict[str, float]:
    """Summaries of :meth:`MPCCommand.pop_stats` under ``<prefix>/``: solve times, fallbacks, binaries, tube.

    Rates are fractions of the counted steps: ``candidate_rate`` the steps on
    the shifted plan because the solvers found none, ``kept_rate`` those on the
    shifted plan because it was cheaper. Counts: ``emergency_count`` (brakes),
    ``invalid_candidate_count`` (shifted plans that would have been applied
    with the state outside their tube, i.e. after disturbances outside ``D``)
    and ``solver_error_count``. ``tube_ratio_max`` is logged only when some step
    had a tube ratio, so never for a Worker designed for ``d_bar = 0``.

    Args:
        stats: Arrays ``(steps, agents)``.
        prefix: E.g. ``"mpc"`` in training, ``"eval_mpc"`` in an evaluation.
        mask: ``(steps, agents)`` steps to count; all by default.
    """
    outcome = stats["outcome"]
    if outcome.size == 0:
        return {}
    keep = np.ones(outcome.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not keep.any():
        return {}
    solve_ms, ratio, kept = stats["solve_ms"][keep], stats["tube_ratio"][keep], outcome[keep]
    result = {
        f"{prefix}/solve_ms_mean": float(solve_ms.mean()),
        f"{prefix}/solve_ms_max": float(solve_ms.max()),
        f"{prefix}/candidate_rate": float(np.mean(kept == CANDIDATE)),
        f"{prefix}/kept_rate": float(np.mean(kept == KEPT)),
        f"{prefix}/emergency_count": float(np.sum(kept == EMERGENCY)),
        f"{prefix}/invalid_candidate_count": float(np.sum(stats["invalid_candidate"][keep])),
        f"{prefix}/solver_error_count": float(np.sum(stats["solver_errors"][keep])),
        f"{prefix}/free_binaries_mean": float(stats["free_binaries"][keep].mean()),
        f"{prefix}/cut_rounds_mean": float(stats["cut_rounds"][keep].mean()),
        f"{prefix}/steps": float(keep.sum()),
    }
    if np.any(~np.isnan(ratio)):
        result[f"{prefix}/tube_ratio_max"] = float(np.nanmax(ratio))
    return result


def episode_mask(lengths: ArrayLike, steps: int) -> BoolArray:
    """``(steps, agents)``: the steps of each agent's first episode, given its length."""
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    mask: BoolArray = np.arange(steps)[:, None] < lengths_arr[None, :]
    return mask

