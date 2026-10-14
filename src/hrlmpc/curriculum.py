"""The reverse spawn curriculum of the training episodes (decision log D62).

With the sparse reward of D51 no architecture finds the goal of the slalom by
chance (the smoke runs of 2026-10-13: no success in 1024 episodes per seed
for PPO and hPPO, whose policies converge to standing still), so the training
starts are drawn from a band of the spawn box's shape that begins near the
goal and slides back towards the spawn box as the policy succeeds:

- stage ``s`` draws uniformly from the free part of the band
  ``[x_start - s step, x_start - s step + width] x [y_lo, y_hi]``, with the
  width and the ``y``-range of the spawn box, the left end clamped to the
  spawn box's, outside the excluded boxes of the configuration (the dead
  ends of a layout, D69); the last stage is the spawn box itself;
- the finished episodes started in a stage are counted in blocks of
  ``window``: when a block is complete and its success rate reaches
  ``threshold`` the stage advances by one, otherwise a new block starts; the
  stage never goes back. In the last stage nothing advances, and the success
  rate of the last ``window`` episodes is kept as a metric (the training
  success rate from the spawn box);
- the evaluation keeps the starts of D20 (its environment has no curriculum),
  so the arrival times and the time to solve are measured from the spawn box
  as before.

The rule is the same for every architecture and lives in the environment, so
the training loops do not change; the stage is logged with the rollout
metrics. The counting is by the order in which episodes finish, so right
after an advancement the first outcomes lean towards successes, which end
sooner than truncated episodes; the lean is bounded by the number of parallel
agents, the same for every architecture. A fixed-band version of the reverse
curriculum of Florensa, Held, Wulfmeier, Zhang and Abbeel, "Reverse
Curriculum Generation for Reinforcement Learning", CoRL 2017: no replay of
earlier starts is needed, since consecutive bands overlap (the step is at most
the width) and every episode crosses the regions learned before.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.typing import NDArray

from hrlmpc.config import Box, CurriculumConfig
from hrlmpc.geometry import Layout

FloatArray = NDArray[np.float64]

MAX_DRAWS = 10_000
"""Candidate points drawn before a sample is given up (never with :data:`MIN_FREE_FRACTION`)."""

MIN_FREE_FRACTION = 0.01
"""Smallest free fraction of a band, estimated at construction on :data:`CHECK_POINTS` fixed random points; with
it, :data:`MAX_DRAWS` candidates fail to give a free point with a probability below ``exp(-100)``."""

CHECK_POINTS = 4_000
"""Points of the estimate of the free fraction of a band."""


class SpawnCurriculum:
    """The bands, the sampler and the advancement rule of D62.

    Args:
        settings: ``x_start``, ``step``, ``window`` and ``threshold``.
        spawn: The spawn box of the task: the shape of every band and the last band.
        layout: The map, for the free space.

    Raises:
        ValueError: If a band's free fraction, estimated on fixed random points, is below
            :data:`MIN_FREE_FRACTION`.
    """

    def __init__(self, settings: CurriculumConfig, spawn: Box, layout: Layout) -> None:
        self.settings = settings
        self.spawn = spawn
        self.layout = layout
        self.width = spawn.x[1] - spawn.x[0]
        distance = max(0.0, settings.x_start - spawn.x[0])
        self.stages = int(math.ceil(distance / settings.step - 1e-9)) + 1
        """How many bands there are, the last one the spawn box."""
        self.stage = 0
        """The current stage, ``0 .. stages - 1``."""
        self._window: deque[bool] = deque(maxlen=settings.window)
        for stage in range(self.stages):
            band = self._band(stage)
            fraction = self.free_fraction(band)
            if fraction < MIN_FREE_FRACTION:
                raise ValueError(
                    f"the curriculum's band {band} of stage {stage} has a free fraction of {fraction:.4f}, "
                    f"below {MIN_FREE_FRACTION}: too few free points"
                )

    def _band(self, stage: int) -> Box:
        if stage >= self.stages - 1:
            return self.spawn  # the last band is the spawn box, exactly
        x_min = max(self.spawn.x[0], self.settings.x_start - stage * self.settings.step)
        return Box(x=(x_min, x_min + self.width), y=self.spawn.y)

    def _free(self, points: FloatArray) -> NDArray[np.bool_]:
        """Free, short of the goal line and outside the excluded boxes (D69)."""
        result: NDArray[np.bool_] = self.layout.in_free_space(points) & ~self.layout.in_goal(points)
        for box in self.settings.exclude:
            inside = (
                (points[:, 0] >= box.x[0]) & (points[:, 0] <= box.x[1])
                & (points[:, 1] >= box.y[0]) & (points[:, 1] <= box.y[1])
            )
            result &= ~inside
        return result

    def free_fraction(self, band: Box) -> float:
        """The fraction of ``band`` that is free, short of the goal and not excluded, estimated on fixed random
    points."""
        rng = np.random.default_rng(0)
        points = rng.uniform([band.x[0], band.y[0]], [band.x[1], band.y[1]], size=(CHECK_POINTS, 2))
        return float(np.mean(self._free(points)))

    @property
    def band(self) -> Box:
        """The band of the current stage."""
        return self._band(self.stage)

    @property
    def final(self) -> bool:
        """Whether the current band is the spawn box."""
        return self.stage >= self.stages - 1

    @property
    def window_success(self) -> float:
        """The success rate of the current block (of the last ``window`` episodes in the last stage); NaN while
        it is empty."""
        return float(np.mean(self._window)) if self._window else float("nan")

    def advance(self) -> None:
        """Move to the next stage, if there is one, and restart the block."""
        if not self.final:
            self.stage += 1
        self._window.clear()

    def sample(self, rng: np.random.Generator, count: int) -> FloatArray:
        """``(count, 2)`` positions uniform on the free part of the current band, short of the goal line.

        Raises:
            ValueError: If ``count`` is negative.
            RuntimeError: If :data:`MAX_DRAWS` candidates give fewer than ``count`` free points.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        band = self.band
        lo, hi = np.array([band.x[0], band.y[0]]), np.array([band.x[1], band.y[1]])
        accepted: list[FloatArray] = [np.empty((0, 2))]
        found, drawn = 0, 0
        while found < count:
            if drawn >= MAX_DRAWS:
                raise RuntimeError(f"no {count} free points in the band {band} after {drawn} draws")
            batch = max(count - found, 8)
            points = rng.uniform(lo, hi, size=(batch, 2))
            drawn += batch
            keep = self._free(points)
            if keep.any():
                accepted.append(points[keep])
                found += int(keep.sum())
        result: FloatArray = np.concatenate(accepted)[:count]
        return result

    def record(self, stage: int, success: bool) -> None:
        """Count a finished episode that started in ``stage`` (``-1`` for an explicit start: ignored).

        Before the last stage the outcomes form blocks of ``window`` episodes: a complete block at or above the
        threshold advances the stage, any other complete block starts a new one. In the last stage the outcomes
        slide over the last ``window`` episodes, for the metric only.
        """
        if stage != self.stage:
            return
        self._window.append(bool(success))
        if self.final or len(self._window) < self.settings.window:
            return
        if self.window_success >= self.settings.threshold:
            self.advance()
        else:
            self._window.clear()

    def metrics(self) -> dict[str, float]:
        """The stage for the logs: ``curriculum/stage``, ``stages``, ``x_min``, ``final``, ``window_success``."""
        return {
            "curriculum/stage": float(self.stage),
            "curriculum/stages": float(self.stages),
            "curriculum/x_min": float(self.band.x[0]),
            "curriculum/final": float(self.final),
            "curriculum/window_success": self.window_success,
        }
