import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class WidthSegment:
    """One piece of a piecewise-constant tunnel width profile: for
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
    worst-case travel.

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
        plain floats for scalar input), so both the scalar `TunnelEnv` and
        the batched `TunnelVecEnv`/`MPCWorker` can share this lookup.
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

    def boundary_between(self, x_from, x_to):
        """The segment boundary a move from `x_from` to `x_to` crosses: the
        `x_start` of the segment entered when moving forward, its `x_end`
        when moving backward -- the face the vehicle enters it through --
        and NaN where both lie in the same segment. Vectorized like
        `bounds_at`. A step crosses at most one boundary here: no segment is
        shorter than a step's travel (0.12 m against 1 m gates)."""
        a = np.asarray(x_from, dtype=np.float64)
        b = np.asarray(x_to, dtype=np.float64)
        scalar_input = a.ndim == 0 and b.ndim == 0
        a, b = np.broadcast_arrays(np.atleast_1d(a), np.atleast_1d(b))
        last = len(self._starts) - 1
        ia = np.clip(np.searchsorted(self._starts, a, side="right") - 1, 0, last)
        ib = np.clip(np.searchsorted(self._starts, b, side="right") - 1, 0, last)
        edges = np.append(self._starts, np.inf)
        face = np.where(ib > ia, edges[ib], np.where(ib < ia, edges[np.minimum(ib + 1, last + 1)], np.nan))
        return float(face[0]) if scalar_input else face

    def envelope(self):
        """`(y_lo, y_hi)` of the union of every segment's bound -- the
        widest lateral extent this profile ever allows, used to build
        `TunnelEnv.observation_space` (which must contain every reachable
        state regardless of which segment produced it)."""
        y_lo = float(np.min(self._center_ys - self._half_widths))
        y_hi = float(np.max(self._center_ys + self._half_widths))
        return y_lo, y_hi


def outside_face(face, forward):
    """The float32 position closest to `face` on the side a vehicle moving
    `forward` (or backward) comes from: below it moving forward, at it
    moving backward, since a segment covers x_start <= p_x < x_end. Where a
    vehicle that ran into a segment's face is put back (the environments'
    step). Vectorized over `face` and `forward`."""
    face = np.asarray(face, dtype=np.float64)
    f = face.astype(np.float32)
    below = np.where(f < face, f, np.nextafter(f, np.float32(-np.inf)))
    at_or_above = np.where(f >= face, f, np.nextafter(f, np.float32(np.inf)))
    out = np.where(forward, below, at_or_above).astype(np.float32)
    return out[()] if out.ndim == 0 else out


def constant_profile(tunnel_width):
    """A single segment spanning the whole corridor -- the profile every
    caller gets when `TunnelEnvConfig.width_profile` is left unset, and
    byte-equal to the environment's historical constant-width behaviour."""
    return WidthProfile([WidthSegment(-np.inf, np.inf, tunnel_width / 2.0, 0.0)])
