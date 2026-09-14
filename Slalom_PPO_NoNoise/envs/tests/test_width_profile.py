import numpy as np
import pytest

from envs.width_profile import WidthSegment, WidthProfile, constant_profile, slalom_profile


def test_constant_profile_reduces_to_symmetric_box():
    profile = constant_profile(tunnel_width=4.0)
    assert profile.envelope() == (-2.0, 2.0)
    for p_x in [-1.0, 0.0, 5.0, 10.0, 11.0, 1e6]:
        assert profile.bounds_at(p_x) == (-2.0, 2.0)


def test_bounds_at_scalar_and_vectorized_agree():
    profile = slalom_profile()
    xs = np.array([-1.0, 1.0, 3.0, 6.0, 8.5, 11.0])
    vec_lo, vec_hi = profile.bounds_at(xs)
    for i, x in enumerate(xs):
        scalar_lo, scalar_hi = profile.bounds_at(float(x))
        assert scalar_lo == pytest.approx(vec_lo[i])
        assert scalar_hi == pytest.approx(vec_hi[i])


def test_bounds_at_preserves_input_shape():
    profile = slalom_profile()
    xs = np.array([[1.0, 3.0], [6.0, 8.5]])
    lo, hi = profile.bounds_at(xs)
    assert lo.shape == xs.shape
    assert hi.shape == xs.shape


def test_slalom_covers_full_extent_with_no_gaps():
    profile = slalom_profile()
    segments = profile.segments
    assert segments[0].x_start == -np.inf
    assert segments[-1].x_end == np.inf
    for a, b in zip(segments, segments[1:]):
        assert a.x_end == b.x_start


def test_slalom_envelope_matches_full_tunnel_width():
    profile = slalom_profile(tunnel_width=4.0)
    assert profile.envelope() == (-2.0, 2.0)


def test_slalom_spawn_box_lands_in_a_full_width_segment():
    profile = slalom_profile()
    for p_x in [0.0, 1.0, 2.0]:
        y_lo, y_hi = profile.bounds_at(p_x)
        assert (y_lo, y_hi) == (-2.0, 2.0)


def test_slalom_gates_are_narrower_and_on_opposite_sides():
    profile = slalom_profile()
    gate1_lo, gate1_hi = profile.bounds_at(4.5)
    gate2_lo, gate2_hi = profile.bounds_at(7.5)
    full_lo, full_hi = profile.bounds_at(0.0)

    assert (gate1_hi - gate1_lo) < (full_hi - full_lo)
    assert (gate2_hi - gate2_lo) < (full_hi - full_lo)
    assert gate1_lo > 0.0  # offset toward +y
    assert gate2_hi < 0.0  # offset toward -y
    # strictly inside the walls
    assert gate1_hi < full_hi and gate1_lo > full_lo
    assert gate2_hi < full_hi and gate2_lo > full_lo


def test_slalom_goal_line_is_in_a_full_width_segment():
    profile = slalom_profile(tunnel_length=10.0)
    assert profile.bounds_at(10.0) == (-2.0, 2.0)


def test_slalom_gate_positions_are_fixed():
    profile = slalom_profile()
    starts_ends = [(seg.x_start, seg.x_end) for seg in profile.segments]
    assert starts_ends == [
        (-np.inf, 4.0),
        (4.0, 5.0),
        (5.0, 7.0),
        (7.0, 8.0),
        (8.0, np.inf),
    ]


def test_slalom_gate_wider_than_slalom_raises():
    with pytest.raises(ValueError):
        slalom_profile(tunnel_width=1.0, gate_half_width=0.75)


def test_slalom_too_short_slalom_raises():
    with pytest.raises(ValueError):
        slalom_profile(tunnel_length=3.0)


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [WidthSegment(0.0, np.inf, 1.0)],  # doesn't start at -inf
        [WidthSegment(-np.inf, 0.0, 1.0)],  # doesn't end at +inf
        [WidthSegment(-np.inf, 5.0, 1.0), WidthSegment(6.0, np.inf, 1.0)],  # gap
        [WidthSegment(-np.inf, 5.0, 1.0), WidthSegment(4.0, np.inf, 1.0)],  # overlap
        [WidthSegment(-np.inf, np.inf, 0.0)],  # non-positive half_width
        [WidthSegment(-np.inf, np.inf, -1.0)],
    ],
)
def test_invalid_profile_construction_raises(segments):
    with pytest.raises(ValueError):
        WidthProfile(segments)


def test_segments_supplied_out_of_order_are_sorted():
    profile = WidthProfile([
        WidthSegment(0.0, np.inf, 1.0, center_y=5.0),
        WidthSegment(-np.inf, 0.0, 2.0, center_y=0.0),
    ])
    assert profile.segments[0].x_start == -np.inf
    assert profile.bounds_at(-1.0) == (-2.0, 2.0)
    assert profile.bounds_at(1.0) == (4.0, 6.0)
