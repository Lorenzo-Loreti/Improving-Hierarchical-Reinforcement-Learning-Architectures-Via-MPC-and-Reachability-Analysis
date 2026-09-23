import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class WidthSegment:
    """One piece of a piecewise-constant slalom width profile: for
    `x_start <= p_x < x_end`, the corridor's lateral bound is
    `[center_y - half_width, center_y + half_width]`.
    """

    x_start: float
    x_end: float
    half_width: float
    center_y: float = 0.0


class WidthProfile:
    """A piecewise-constant lateral bound on `p_y` as a function of `p_x`.

    Piecewise-*constant* rather than a continuous taper because `MPCWorker`'s
    QP treats `p_x` as a decision variable inside its horizon, so it can only
    look up the segment at the current, measured `p_x` and hold it constant
    across one solve -- no segment may be shorter than one horizon's
    worst-case travel (`slalom_profile` below is where that calibration is
    applied).

    Segments must together cover every real `p_x` exactly once: sorted,
    contiguous (no gap or overlap), the first starting at `-inf` and the
    last ending at `+inf`. That lets every consumer call `bounds_at` for any
    `p_x` without a separate range check.
    """

    def __init__(self, segments):
        segments = tuple(sorted(segments, key=lambda s: s.x_start))

        if not segments:
            raise ValueError("WidthProfile requires at least one segment")
        if segments[0].x_start != -np.inf:
            raise ValueError(
                f"the first segment must start at -inf so every p_x resolves "
                f"to some segment, got x_start={segments[0].x_start}"
            )
        if segments[-1].x_end != np.inf:
            raise ValueError(
                f"the last segment must end at +inf so every p_x resolves to "
                f"some segment, got x_end={segments[-1].x_end}"
            )
        for seg in segments:
            if not seg.x_start < seg.x_end:
                raise ValueError(f"segment x_start must be < x_end, got {seg}")
            if seg.half_width <= 0:
                raise ValueError(f"half_width must be > 0, got {seg}")
        for a, b in zip(segments, segments[1:]):
            if a.x_end != b.x_start:
                raise ValueError(
                    f"segments must be contiguous with no gap or overlap, "
                    f"got x_end={a.x_end} immediately followed by "
                    f"x_start={b.x_start}"
                )

        self.segments = segments
        self._starts = np.array([s.x_start for s in segments], dtype=np.float64)
        self._half_widths = np.array([s.half_width for s in segments], dtype=np.float64)
        self._center_ys = np.array([s.center_y for s in segments], dtype=np.float64)

    def bounds_at(self, p_x):
        """`(y_lo, y_hi)` for the segment containing `p_x`.

        `p_x` may be a scalar (python float/int, or a 0-d array) or an
        ndarray of any shape; the return matches that shape (or is a pair of
        plain floats for scalar input), so both the scalar `SlalomEnv` and
        the batched `SlalomVecEnv`/`MPCWorker` can share this lookup.
        """
        x = np.asarray(p_x, dtype=np.float64)
        scalar_input = x.ndim == 0
        x_flat = np.atleast_1d(x)

        idx = np.searchsorted(self._starts, x_flat, side="right") - 1
        idx = np.clip(idx, 0, len(self._starts) - 1)

        center = self._center_ys[idx]
        half = self._half_widths[idx]
        y_lo, y_hi = center - half, center + half

        if scalar_input:
            return float(y_lo[0]), float(y_hi[0])
        return y_lo.reshape(x.shape), y_hi.reshape(x.shape)

    def envelope(self):
        """`(y_lo, y_hi)` of the union of every segment's bound -- the
        widest lateral extent this profile ever allows, used to build
        `SlalomEnv.observation_space` (which must contain every reachable
        state regardless of which segment produced it)."""
        y_lo = float(np.min(self._center_ys - self._half_widths))
        y_hi = float(np.max(self._center_ys + self._half_widths))
        return y_lo, y_hi


def constant_profile(tunnel_width):
    """A single segment spanning the whole corridor -- the profile every
    caller gets when `SlalomEnvConfig.width_profile` is left unset, and
    byte-equal to the environment's historical constant-width behaviour."""
    return WidthProfile([WidthSegment(-np.inf, np.inf, tunnel_width / 2.0, 0.0)])


def slalom_profile(tunnel_length=10.0, tunnel_width=4.0, gate_half_width=0.75):
    """Two narrow, laterally-offset bottleneck "gates" separated by
    full-width segments, at fixed positions along `p_x`.

    Segment layout, at the defaults (`tunnel_length=10.0, tunnel_width=4.0`):

        (-inf, 4.0]  full width            -- entry; contains the spawn box
        (4.0, 5.0]   narrow, offset +1.0   -- gate 1
        (5.0, 7.0]   full width            -- mid recovery
        (7.0, 8.0]   narrow, offset -1.0   -- gate 2 (mirrored)
        (8.0, +inf)  full width            -- exit; contains the goal line

    These boundaries are fixed constants rather than derived from a
    per-solve travel-distance calibration, so callers relying on `MPCWorker`
    or a manager horizon should independently check that no segment here is
    shorter than that horizon's worst-case per-step travel.
    """
    full_half = tunnel_width / 2.0
    gate_offset = tunnel_width / 4.0

    if gate_half_width >= full_half:
        raise ValueError(
            f"gate_half_width ({gate_half_width}) must be < tunnel_width/2 "
            f"({full_half}) or the gate is not actually narrower than the "
            f"slalom"
        )
    if gate_offset + gate_half_width > full_half:
        raise ValueError(
            f"a gate offset by {gate_offset} with half_width {gate_half_width} "
            f"would extend to {gate_offset + gate_half_width}, past the "
            f"slalom wall at {full_half}"
        )

    spawn_end = 2.0  # matches SlalomEnv.reset()'s hardcoded spawn box p_x in [0, 2]
    entry_end = 4.0
    gate1_end = 5.0
    mid_end = 7.0
    gate2_end = 8.0

    if entry_end <= spawn_end:
        raise ValueError(
            f"gate 1 starts at x={entry_end}, which overlaps the spawn box "
            f"ending at x={spawn_end}"
        )
    if gate2_end >= tunnel_length:
        raise ValueError(
            f"slalom_profile's gates end at x={gate2_end}, which does not "
            f"leave a full-width approach before the goal line at "
            f"tunnel_length={tunnel_length}; increase tunnel_length"
        )

    return WidthProfile([
        WidthSegment(-np.inf, entry_end, full_half, 0.0),
        WidthSegment(entry_end, gate1_end, gate_half_width, gate_offset),
        WidthSegment(gate1_end, mid_end, full_half, 0.0),
        WidthSegment(mid_end, gate2_end, gate_half_width, -gate_offset),
        WidthSegment(gate2_end, np.inf, full_half, 0.0),
    ])
