import numpy as np
import pytest

from envs.width_profile import WidthSegment, WidthProfile, constant_profile


def test_constant_profile_reduces_to_symmetric_box():
    profile = constant_profile(tunnel_width=4.0)
    assert profile.envelope() == (-2.0, 2.0)
    for p_x in [-1.0, 0.0, 5.0, 10.0, 11.0, 1e6]:
        assert profile.bounds_at(p_x) == (-2.0, 2.0)


def _multi_segment_profile():
    return WidthProfile([
        WidthSegment(-np.inf, 2.0, 2.0, 0.0),
        WidthSegment(2.0, 7.0, 1.0, 1.0),
        WidthSegment(7.0, np.inf, 2.0, 0.0),
    ])


def test_bounds_at_scalar_and_vectorized_agree():
    profile = _multi_segment_profile()
    xs = np.array([-1.0, 1.0, 3.0, 6.0, 8.5, 11.0])
    vec_lo, vec_hi = profile.bounds_at(xs)
    for i, x in enumerate(xs):
        scalar_lo, scalar_hi = profile.bounds_at(float(x))
        assert scalar_lo == pytest.approx(vec_lo[i])
        assert scalar_hi == pytest.approx(vec_hi[i])


def test_bounds_at_preserves_input_shape():
    profile = _multi_segment_profile()
    xs = np.array([[1.0, 3.0], [6.0, 8.5]])
    lo, hi = profile.bounds_at(xs)
    assert lo.shape == xs.shape
    assert hi.shape == xs.shape


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
