"""Unit tests for the CVI cubic-spline basis (src.spline)."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import make_knot_grid
from src.spline import CVISplineBasis


@pytest.fixture
def basis10() -> CVISplineBasis:
    return CVISplineBasis(make_knot_grid(10))


@pytest.mark.parametrize("n_knots", [5, 10, 15, 20, 30, 40])
def test_knot_grid_contains_atm(n_knots: int) -> None:
    knots = make_knot_grid(n_knots)
    assert len(knots) == n_knots
    assert np.all(np.diff(knots) > 0)
    assert np.any(np.abs(knots) < 1e-12)
    assert knots[0] == pytest.approx(-8.0)
    assert knots[-1] == pytest.approx(8.0)


def test_constant_variance_is_reachable(basis10: CVISplineBasis) -> None:
    """v(z) = v0 is realized by (v0, 0, 0, ..., 0)."""
    p = np.zeros(basis10.n_params)
    p[0] = 0.04
    z = np.linspace(-6, 6, 25)
    v, vp, vpp = basis10.evaluate(p, z)
    assert np.allclose(v, 0.04)
    assert np.allclose(vp, 0.0)
    assert np.allclose(vpp, 0.0)


def test_linear_variance_is_reachable(basis10: CVISplineBasis) -> None:
    """v(z) = a + b z is reachable because second derivatives are zero."""
    p = np.zeros(basis10.n_params)
    p[0], p[1] = 0.04, 0.01
    z = np.linspace(-6, 6, 25)
    v, vp, vpp = basis10.evaluate(p, z)
    assert np.allclose(v, 0.04 + 0.01 * z)
    assert np.allclose(vp, 0.01)
    assert np.allclose(vpp, 0.0)


def test_operators_are_linear_in_params(basis10: CVISplineBasis) -> None:
    """B_v @ p == v(z); same for the derivative operators."""
    rng = np.random.default_rng(0)
    z = np.linspace(-6, 6, 15)
    B_v, B_vp, B_vpp = basis10.operators(z)
    assert B_v.shape == (15, basis10.n_params)
    for _ in range(3):
        p = rng.normal(size=basis10.n_params)
        v_direct, vp_direct, vpp_direct = basis10.evaluate(p, z)
        assert np.allclose(B_v @ p, v_direct)
        assert np.allclose(B_vp @ p, vp_direct)
        assert np.allclose(B_vpp @ p, vpp_direct)


def test_tails_are_linearly_extrapolated(basis10: CVISplineBasis) -> None:
    """Outside [z_0, z_{n-1}] the spline is linear in z: v'' = 0."""
    rng = np.random.default_rng(1)
    p = rng.normal(size=basis10.n_params)
    p[2] = 0.0   # edge convexity c_0
    p[-1] = 0.0  # edge convexity c_{n-1}
    z_out_right = np.array([9.0, 12.0, 20.0])
    z_out_left = np.array([-9.0, -15.0, -25.0])
    for z in (z_out_right, z_out_left):
        _, _, vpp = basis10.evaluate(p, z)
        assert np.allclose(vpp, 0.0)


def test_invalid_knot_grid_raises() -> None:
    with pytest.raises(ValueError):
        CVISplineBasis(np.array([-1.0, 0.0, 1.0]))          # too short
    with pytest.raises(ValueError):
        CVISplineBasis(np.array([0.0, -1.0, 1.0, 2.0]))     # not increasing
    with pytest.raises(ValueError):
        CVISplineBasis(np.array([-2.0, -1.0, 1.0, 2.0]))    # 0 missing
