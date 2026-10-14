"""Typed configuration of the environment, loaded from YAML (code rule C2).

A file under ``configs/env/`` describes one layout: the physical parameters
(decision log D9), the geometry (D11), the task (D13; with the optional
reverse spawn curriculum of D62) and the reward coefficients (D13, D14).
Loading validates every value. Unknown or missing keys are errors, so a typo
cannot silently fall back to a default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hrlmpc.model import PhysicsParams
from hrlmpc.utils.strict_yaml import load_yaml


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle ``[x[0], x[1]] x [y[0], y[1]]`` in metres.

    Raises:
        ValueError: If an interval is empty or degenerate.
    """

    x: tuple[float, float]
    y: tuple[float, float]

    def __post_init__(self) -> None:
        for name, (lo, hi) in (("x", self.x), ("y", self.y)):
            if not lo < hi:
                raise ValueError(f"box interval {name}=[{lo}, {hi}] must have lo < hi")

    def contains(self, other: Box) -> bool:
        """Whether ``other`` lies in this box (closed sets)."""
        return (
            self.x[0] <= other.x[0]
            and other.x[1] <= self.x[1]
            and self.y[0] <= other.y[0]
            and other.y[1] <= self.y[1]
        )

    def overlaps_interior(self, other: Box) -> bool:
        """Whether the interiors of the two boxes intersect."""
        return (
            min(self.x[1], other.x[1]) > max(self.x[0], other.x[0])
            and min(self.y[1], other.y[1]) > max(self.y[0], other.y[0])
        )


@dataclass(frozen=True)
class GeometryConfig:
    """Arena, obstacles and goal region (decision log D11).

    Attributes:
        arena: The arena ``P``. Every face of ``P`` is a wall.
        obstacles: The obstacles ``O_i``, each inside the arena. Every face is
            a wall.
        goal_x: The goal region is ``{p in P_free : p_x >= goal_x}``.

    Raises:
        ValueError: If an obstacle leaves the arena or the goal line is not
            strictly inside it.
    """

    arena: Box
    obstacles: tuple[Box, ...]
    goal_x: float

    def __post_init__(self) -> None:
        for i, obstacle in enumerate(self.obstacles):
            if not self.arena.contains(obstacle):
                raise ValueError(f"obstacle {i} {obstacle} is not inside the arena {self.arena}")
        if not self.arena.x[0] < self.goal_x < self.arena.x[1]:
            raise ValueError(f"goal_x={self.goal_x} must lie strictly inside the arena's x-range {self.arena.x}")


@dataclass(frozen=True)
class CurriculumConfig:
    """The reverse spawn curriculum of the training episodes (decision log D62).

    Training starts are drawn from a band of the spawn box's shape that begins
    at ``x_start`` and slides back by ``step`` whenever a block of ``window``
    episodes started in the current band has a success rate of at least
    ``threshold``, until it coincides with the spawn box. The evaluation keeps
    its own starts (:mod:`hrlmpc.curriculum`).

    Attributes:
        x_start: Left end [m] of the first band.
        step: By how much [m] the band slides back per stage; at most the spawn box's width.
        window: Episodes of a block, over which the success rate is measured.
        threshold: Success rate of a block that advances the stage, in ``(0, 1]``.
        exclude: Boxes no training start is drawn from (D69): the dead ends of
            a layout, such as the cavity of the trap's U, which the policies
            meet on their way instead of starting in. Inside the arena and
            apart from the spawn box; empty by default.

    Raises:
        ValueError: If the step is not positive, the window is below 1 or the
            threshold is outside ``(0, 1]``.
    """

    x_start: float
    step: float
    window: int
    threshold: float
    exclude: tuple[Box, ...] = ()

    def __post_init__(self) -> None:
        if not self.step > 0.0:
            raise ValueError(f"curriculum step must be positive, got {self.step}")
        if self.window < 1:
            raise ValueError(f"curriculum window must be at least 1, got {self.window}")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"curriculum threshold must lie in (0, 1], got {self.threshold}")


@dataclass(frozen=True)
class TaskConfig:
    """Initial-state distribution and horizon (decision log D13), and the curriculum of the training starts (D62).

    Attributes:
        spawn: Initial positions are uniform on this box, with zero velocity;
            with a curriculum, the evaluation's starts and the last band.
        horizon: Episodes are truncated after this many steps.
        curriculum: The reverse spawn curriculum, or ``None`` for starts uniform on ``spawn``.

    Raises:
        ValueError: If the horizon is not positive.
    """

    spawn: Box
    horizon: int
    curriculum: CurriculumConfig | None = None

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}")


@dataclass(frozen=True)
class RewardConfig:
    """Reward coefficients (decision log D13, D14).

    The reward of step ``k`` is

        r_k = - time_penalty
              - contact_impulse_coef * ||dv_wall_k|| - contact_step_coef * 1[contact at k]
              + success_bonus * 1[p_(k+1) in G]
              + shaping_discount * Phi(p_(k+1)) - Phi(p_k)
              - effort_coef * ||u_k||^2 / a_max^2,

    with ``Phi(p) = -progress_coef * max(goal_x - p_x, 0)``. Every
    coefficient is non-negative; the signs are in the formula.

    With ``shaping_discount`` equal to the learners' discount the progress
    term is exact potential-based shaping (Ng, Harada and Russell, ICML
    1999), which leaves the optimal policy unchanged but gives partial
    progress no lasting value. The configurations use 1 (D22): the
    undiscounted progress reward of the pre-alignment code, which keeps
    rewarding the agent for being closer to the goal.

    Raises:
        ValueError: If a coefficient is negative or the shaping discount is
            outside ``(0, 1]``.
    """

    time_penalty: float
    success_bonus: float
    progress_coef: float
    shaping_discount: float
    effort_coef: float
    contact_impulse_coef: float
    contact_step_coef: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0.0:
                raise ValueError(f"reward coefficient {name} must be non-negative, got {value}")
        if not 0.0 < self.shaping_discount <= 1.0:
            raise ValueError(f"shaping_discount must lie in (0, 1], got {self.shaping_discount}")


@dataclass(frozen=True)
class EnvConfig:
    """Complete configuration of one environment layout.

    Raises:
        ValueError: If the spawn box leaves the arena, overlaps an obstacle or
            reaches the goal region; or, with a curriculum, if its first band
            starts behind the spawn box or crosses the goal line, or its step
            exceeds the spawn box's width (the bands must overlap).
    """

    physics: PhysicsParams
    geometry: GeometryConfig
    task: TaskConfig
    reward: RewardConfig

    def __post_init__(self) -> None:
        spawn = self.task.spawn
        if not self.geometry.arena.contains(spawn):
            raise ValueError(f"spawn box {spawn} is not inside the arena {self.geometry.arena}")
        for i, obstacle in enumerate(self.geometry.obstacles):
            if spawn.overlaps_interior(obstacle):
                raise ValueError(f"spawn box {spawn} overlaps obstacle {i} {obstacle}")
        if not spawn.x[1] < self.geometry.goal_x:
            raise ValueError(f"spawn box {spawn} reaches the goal region p_x >= {self.geometry.goal_x}")
        curriculum = self.task.curriculum
        if curriculum is not None:
            width = spawn.x[1] - spawn.x[0]
            if curriculum.x_start < spawn.x[0]:
                raise ValueError(f"the curriculum's x_start {curriculum.x_start} lies behind the spawn box {spawn}")
            if curriculum.step > width:
                raise ValueError(
                    f"the curriculum's step {curriculum.step} exceeds the spawn box's width {width}: consecutive "
                    "bands would leave a gap"
                )
            band = Box(x=(curriculum.x_start, curriculum.x_start + width), y=spawn.y)
            if not band.x[1] <= self.geometry.goal_x:
                raise ValueError(
                    f"the curriculum's first band {band} crosses the goal line p_x = {self.geometry.goal_x}"
                )
            for i, box in enumerate(curriculum.exclude):
                if not self.geometry.arena.contains(box):
                    raise ValueError(f"excluded box {i} {box} is not inside the arena {self.geometry.arena}")
                if spawn.overlaps_interior(box):
                    raise ValueError(f"excluded box {i} {box} overlaps the spawn box {spawn}, the last band")


_PHYSICS_KEYS = ("dt", "a_max", "v_max", "gamma", "gamma_w", "d_bar")
_REWARD_KEYS = (
    "time_penalty",
    "success_bonus",
    "progress_coef",
    "shaping_discount",
    "effort_coef",
    "contact_impulse_coef",
    "contact_step_coef",
)


def _section(
    data: Mapping[str, Any], keys: tuple[str, ...], where: str, optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Return ``data`` as a dict after checking that its keys are exactly ``keys``, apart from ``optional``."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(data).__name__}")
    missing = [k for k in keys if k not in data and k not in optional]
    unknown = [k for k in data if k not in keys]
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return dict(data)


def _float(value: Any, where: str) -> float:
    """Convert a YAML scalar to a finite float, rejecting booleans, non-numbers, infinities and NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite, got {value!r}")
    return float(value)


def _int(value: Any, where: str) -> int:
    """Convert a YAML scalar to int, rejecting booleans and non-integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    return value


def _interval(value: Any, where: str) -> tuple[float, float]:
    """Convert a two-element YAML list to an interval."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{where} must be a list [lo, hi], got {value!r}")
    return (_float(value[0], where), _float(value[1], where))


def _box(data: Any, where: str) -> Box:
    """Convert ``{x: [lo, hi], y: [lo, hi]}`` to a :class:`Box`."""
    section = _section(data, ("x", "y"), where)
    return Box(x=_interval(section["x"], f"{where}.x"), y=_interval(section["y"], f"{where}.y"))


def _curriculum(data: Any, where: str) -> CurriculumConfig | None:
    """Convert ``{x_start, step, window, threshold[, exclude]}`` to a :class:`CurriculumConfig`; ``None`` stays
    ``None``; ``exclude`` is an optional list of boxes."""
    if data is None:
        return None
    section = _section(data, ("x_start", "step", "window", "threshold", "exclude"), where, optional=("exclude",))
    raw_exclude = section.get("exclude", [])
    if not isinstance(raw_exclude, (list, tuple)):
        raise ValueError(f"{where}.exclude must be a list of boxes, got {raw_exclude!r}")
    exclude = tuple(_box(item, f"{where}.exclude[{i}]") for i, item in enumerate(raw_exclude))
    return CurriculumConfig(
        x_start=_float(section["x_start"], f"{where}.x_start"),
        step=_float(section["step"], f"{where}.step"),
        window=_int(section["window"], f"{where}.window"),
        threshold=_float(section["threshold"], f"{where}.threshold"),
        exclude=exclude,
    )


def env_config_from_dict(data: Mapping[str, Any]) -> EnvConfig:
    """Build and validate an :class:`EnvConfig` from a nested mapping.

    Args:
        data: Mapping with the sections ``physics``, ``geometry``, ``task``
            and ``reward``, as in the files under ``configs/env/``.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If a key is missing or unknown, or a value is invalid.
    """
    top = _section(data, ("physics", "geometry", "task", "reward"), "config")

    physics_raw = _section(top["physics"], _PHYSICS_KEYS, "physics")
    physics = PhysicsParams(**{k: _float(physics_raw[k], f"physics.{k}") for k in _PHYSICS_KEYS})

    geometry_raw = _section(top["geometry"], ("arena", "obstacles", "goal_x"), "geometry")
    obstacles_raw = geometry_raw["obstacles"]
    if not isinstance(obstacles_raw, list):
        raise ValueError(f"geometry.obstacles must be a list, got {obstacles_raw!r}")
    geometry = GeometryConfig(
        arena=_box(geometry_raw["arena"], "geometry.arena"),
        obstacles=tuple(_box(o, f"geometry.obstacles[{i}]") for i, o in enumerate(obstacles_raw)),
        goal_x=_float(geometry_raw["goal_x"], "geometry.goal_x"),
    )

    task_raw = _section(top["task"], ("spawn", "horizon", "curriculum"), "task", optional=("curriculum",))
    task = TaskConfig(
        spawn=_box(task_raw["spawn"], "task.spawn"),
        horizon=_int(task_raw["horizon"], "task.horizon"),
        curriculum=_curriculum(task_raw.get("curriculum"), "task.curriculum"),
    )

    reward_raw = _section(top["reward"], _REWARD_KEYS, "reward")
    reward = RewardConfig(**{k: _float(reward_raw[k], f"reward.{k}") for k in _REWARD_KEYS})

    return EnvConfig(physics=physics, geometry=geometry, task=task, reward=reward)


def _plain(value: Any) -> Any:
    """Turn tuples into lists recursively, so that the result is plain YAML."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def env_config_to_dict(config: EnvConfig) -> dict[str, Any]:
    """Return the configuration as a nested dict of plain YAML values.

    The result round-trips: ``env_config_from_dict(env_config_to_dict(c)) == c``.
    """
    result: dict[str, Any] = _plain(asdict(config))
    return result


def load_env_config(path: str | Path) -> EnvConfig:
    """Load and validate an environment configuration from a YAML file.

    Args:
        path: Path of the YAML file.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If the file repeats a key or does not describe a valid configuration.
    """
    return env_config_from_dict(load_yaml(path))
