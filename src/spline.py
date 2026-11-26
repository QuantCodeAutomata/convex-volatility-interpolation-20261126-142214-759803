"""CVI cubic spline basis.

Implements the parameterization of Section "The CVI Parameterization"
(Deschâtres, 2026, p. 2): a cubic spline whose second derivative is piecewise
linear between the knots ``z_0, ..., z_{n-1}``, parameterized by

    p = [v(0), (dv/dz)(0), c_0, c_1, ..., c_{n-1}]      (n+2 entries)

where ``c_i`` are the second derivatives of ``v`` at the knots. Edge
convexities ``c_0`` and ``c_{n-1}`` are enforced to zero by the calibration
(linear extrapolation in log-strike in the tails, paper p. 2); they are kept
as formal parameters so that calibration constraints can set them.

Below ``z_0`` and above ``z_{n-1}`` the function is extrapolated linearly in
log-strike (so ``v`` is a linear function of ``z`` outside the knot range).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _propagate(v0: float, vp0: float, c: np.ndarray, knots: np.ndarray,
               l: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(v_i, v'_i)`` at every knot from ``(v(z_l), v'(z_l), c)``.

    Uses the piecewise-linear-``v''`` recurrences:

        v'_{i+1} = v'_i + 0.5 (c_i + c_{i+1}) h_i
        v_{i+1}  = v_i  + v'_i h_i + (1/6)(2 c_i + c_{i+1}) h_i^2

    with ``h_i = z_{i+1} - z_i``. Propagation is done both forward and
    backward from the anchor knot ``l``.
    """
    n = len(knots)
    v = np.zeros(n)
    vp = np.zeros(n)
    v[l] = v0
    vp[l] = vp0
    # Forward propagation i -> i+1.
    for i in range(l, n - 1):
        h = knots[i + 1] - knots[i]
        vp[i + 1] = vp[i] + 0.5 * (c[i] + c[i + 1]) * h
        v[i + 1] = v[i] + vp[i] * h + (2.0 * c[i] + c[i + 1]) * h * h / 6.0
    # Backward propagation i -> i-1 using negative step.
    for i in range(l, 0, -1):
        h = knots[i] - knots[i - 1]
        # Running v'_i = v'_{i-1} + 0.5 (c_{i-1} + c_i) h  =>
        #   v'_{i-1} = v'_i - 0.5 (c_{i-1} + c_i) h
        vp[i - 1] = vp[i] - 0.5 * (c[i - 1] + c[i]) * h
        # v_i = v_{i-1} + v'_{i-1} h + (1/6)(2 c_{i-1} + c_i) h^2  =>
        #   v_{i-1} = v_i - v'_{i-1} h - (1/6)(2 c_{i-1} + c_i) h^2
        v[i - 1] = v[i] - vp[i - 1] * h - (2.0 * c[i - 1] + c[i]) * h * h / 6.0
    return v, vp


def _eval_point(z: float, v: np.ndarray, vp: np.ndarray, c: np.ndarray,
                knots: np.ndarray) -> tuple[float, float, float]:
    """Evaluate ``(v(z), v'(z), v''(z))`` from the knot-value arrays."""
    n = len(knots)
    if z <= knots[0]:
        # Linear extrapolation beyond z_0: v''=0, v' constant at v'_0.
        return v[0] + vp[0] * (z - knots[0]), vp[0], 0.0
    if z >= knots[-1]:
        return v[n - 1] + vp[n - 1] * (z - knots[-1]), vp[n - 1], 0.0
    # Locate the interval [z_i, z_{i+1}].
    i = int(np.searchsorted(knots, z, side="right") - 1)
    h = knots[i + 1] - knots[i]
    dz = z - knots[i]
    # v''(z) = c_i + (c_{i+1} - c_i) dz/h (piecewise linear).
    vpp = c[i] + (c[i + 1] - c[i]) * dz / h
    # Integrated twice from (v_i, v'_i, c_i, c_{i+1}).
    vp_z = vp[i] + c[i] * dz + 0.5 * (c[i + 1] - c[i]) * dz * dz / h
    v_z = v[i] + vp[i] * dz + 0.5 * c[i] * dz * dz + (c[i + 1] - c[i]) * dz ** 3 / (6.0 * h)
    return v_z, vp_z, vpp


class CVISplineBasis:
    """Linear operators mapping the (n+2) cubic-spline parameters to v/v'/v''.

    Parameters
    ----------
    knots : np.ndarray
        Strictly increasing array of ``n`` knot positions in normalized
        log-moneyness. Must contain ``0`` (ATM knot); the constructor locates
        the ATM index and raises if it is missing.
    """

    def __init__(self, knots: np.ndarray) -> None:
        knots = np.asarray(knots, dtype=float)
        if knots.ndim != 1 or len(knots) < 4:
            raise ValueError("knots must be a 1-D array of length >= 4")
        if not np.all(np.diff(knots) > 0):
            raise ValueError("knots must be strictly increasing")
        atm_idx = int(np.argmin(np.abs(knots)))
        if abs(knots[atm_idx]) > 1e-10:
            raise ValueError("knot grid must contain z = 0 (ATM knot)")
        self.knots = knots
        self.n = len(knots)
        self.atm_idx = atm_idx
        self.n_params = self.n + 2  # v(0), v'(0), c_0, ..., c_{n-1}

    # ------------------------------------------------------------------
    # Core: evaluate the k-th basis function (one-hot parameter) at points.
    # ------------------------------------------------------------------
    def _basis_column(self, k: int, z_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(v, v', v'')`` arrays at ``z_eval`` when parameter ``k`` is 1."""
        if k == 0:
            v0, vp0 = 1.0, 0.0
            c = np.zeros(self.n)
        elif k == 1:
            v0, vp0 = 0.0, 1.0
            c = np.zeros(self.n)
        else:
            v0, vp0 = 0.0, 0.0
            c = np.zeros(self.n)
            c[k - 2] = 1.0
        v_knots, vp_knots = _propagate(v0, vp0, c, self.knots, self.atm_idx)
        out_v = np.empty_like(z_eval)
        out_vp = np.empty_like(z_eval)
        out_vpp = np.empty_like(z_eval)
        for j, z in enumerate(z_eval):
            out_v[j], out_vp[j], out_vpp[j] = _eval_point(
                float(z), v_knots, vp_knots, c, self.knots
            )
        return out_v, out_vp, out_vpp

    def operators(self, z_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return dense operator triplets ``(B_v, B_vp, B_vpp)``.

        Shapes are ``(len(z_eval), n+2)``. ``B_v @ p`` gives ``v(z_eval)``;
        ``B_vp @ p`` gives ``(dv/dz)(z_eval)``; ``B_vpp @ p`` gives
        ``(d^2v/dz^2)(z_eval)``.
        """
        z_eval = np.asarray(z_eval, dtype=float)
        m = len(z_eval)
        B_v = np.zeros((m, self.n_params))
        B_vp = np.zeros((m, self.n_params))
        B_vpp = np.zeros((m, self.n_params))
        for k in range(self.n_params):
            vv, vpp_col, vppp_col = self._basis_column(k, z_eval)
            B_v[:, k] = vv
            B_vp[:, k] = vpp_col
            B_vpp[:, k] = vppp_col
        return B_v, B_vp, B_vpp

    def operators_sparse(self, z_eval: np.ndarray) -> tuple[sparse.csc_matrix, sparse.csc_matrix, sparse.csc_matrix]:
        """Return sparse CSC versions of the ``operators`` triplet."""
        B_v, B_vp, B_vpp = self.operators(z_eval)
        return (
            sparse.csc_matrix(B_v),
            sparse.csc_matrix(B_vp),
            sparse.csc_matrix(B_vpp),
        )

    # Convenience: direct evaluation given a parameter vector.
    def evaluate(self, params: np.ndarray, z_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(v(z), v'(z), v''(z))`` at ``z_eval`` for a given parameter vector."""
        params = np.asarray(params, dtype=float)
        if params.shape[-1] != self.n_params:
            raise ValueError(f"params must have last dim {self.n_params}")
        v0, vp0 = float(params[0]), float(params[1])
        c = np.asarray(params[2:], dtype=float)
        v_knots, vp_knots = _propagate(v0, vp0, c, self.knots, self.atm_idx)
        z_eval = np.asarray(z_eval, dtype=float)
        out_v = np.empty_like(z_eval)
        out_vp = np.empty_like(z_eval)
        out_vpp = np.empty_like(z_eval)
        for j, z in enumerate(z_eval):
            out_v[j], out_vp[j], out_vpp[j] = _eval_point(
                float(z), v_knots, vp_knots, c, self.knots
            )
        return out_v, out_vp, out_vpp
