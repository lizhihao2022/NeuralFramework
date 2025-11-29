# models/m2no/utils.py
# ---------------------------------------------------------------
# This file is based on the implementation in:
#   https://github.com/gaurav71531/mwt-operator
# ---------------------------------------------------------------
from __future__ import annotations

from functools import partial
from typing import Callable, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.special import eval_legendre
from sympy import Poly, Symbol, chebyshevt, legendre

Array = npt.NDArray[np.float64]
BasisFn = Callable[[Array], Array]


def legendreDer(k: int, x: Array | float) -> Array:
    """
    Derivative-related helper for Legendre polynomials.

    This follows the original formula:
        sum_{i = k-1, k-3, ... >= 0} (2*i+1) * P_i(x)

    Args:
        k: polynomial order
        x: evaluation points

    Returns:
        Array of same shape as x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x_arr, dtype=np.float64)

    def _legendre(idx: int, xx: Array) -> Array:
        return (2 * idx + 1) * eval_legendre(idx, xx)

    for i in range(k - 1, -1, -2):
        out = out + _legendre(i, x_arr)
    return out


def phi_(phi_c: Array, x: Array | float, lb: float = 0.0, ub: float = 1.0) -> Array:
    """
    Evaluate a polynomial on [lb, ub], zero outside.

    Args:
        phi_c: polynomial coefficients in ascending order (np.polynomial style)
        x: points to evaluate
        lb, ub: support interval

    Returns:
        Values of the polynomial on x, zero outside [lb, ub].
    """
    x_arr = np.asarray(x, dtype=np.float64)
    mask = np.logical_or(x_arr < lb, x_arr > ub)
    poly_val = np.polynomial.polynomial.Polynomial(phi_c)(x_arr)
    # ensure we return a float64 ndarray (mask may promote types)
    return poly_val.astype(np.float64) * (~mask).astype(np.float64)


def get_phi_psi(
    k: int,
    base: str,
) -> Tuple[list[BasisFn], list[BasisFn], list[BasisFn]]:
    """
    Construct scaling functions (phi) and wavelets (psi1, psi2) for a
    multiwavelet basis based on either Legendre or Chebyshev polynomials.

    Args:
        k: number of basis functions
        base: 'legendre' or 'chebyshev'

    Returns:
        (phi, psi1, psi2):
            phi  : list of k scaling basis functions
            psi1 : list of k wavelet functions on [0, 0.5]
            psi2 : list of k wavelet functions on [0.5, 1]
    """
    if base not in ("legendre", "chebyshev"):
        raise ValueError(f"Base '{base}' not supported. Use 'legendre' or 'chebyshev'.")

    x_sym = Symbol("x")

    # coefficient matrices (ascending power, Polynomial-style)
    phi_coeff: Array = np.zeros((k, k), dtype=np.float64)
    phi_2x_coeff: Array = np.zeros((k, k), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Legendre-based basis
    # ---------------------------------------------------------------------
    if base == "legendre":
        # build phi and phi(2x)
        for ki in range(k):
            # legendre(ki, 2x-1) expanded in x
            coeff_ = Poly(legendre(ki, 2 * x_sym - 1), x_sym).all_coeffs()  # type: ignore
            coeff_arr = np.array([float(c) for c in coeff_], dtype=np.float64)
            phi_coeff[ki, : ki + 1] = np.flip(np.sqrt(2 * ki + 1.0) * coeff_arr)

            # legendre(ki, 4x-1) expanded in x
            coeff_2 = Poly(legendre(ki, 4 * x_sym - 1), x_sym).all_coeffs()  # type: ignore
            coeff_2_arr = np.array([float(c) for c in coeff_2], dtype=np.float64)
            phi_2x_coeff[ki, : ki + 1] = np.flip(
                np.sqrt(2.0) * np.sqrt(2 * ki + 1.0) * coeff_2_arr
            )

        psi1_coeff: Array = np.zeros((k, k), dtype=np.float64)
        psi2_coeff: Array = np.zeros((k, k), dtype=np.float64)

        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            psi2_coeff[ki, :] = phi_2x_coeff[ki, :]

            # project out phi_i
            for i in range(k):
                a = phi_2x_coeff[ki, : ki + 1]
                b = phi_coeff[i, : i + 1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0.0
                idx = np.arange(len(prod_), dtype=np.float64)
                proj_ = (prod_ * 1.0 / (idx + 1.0) * np.power(0.5, 1.0 + idx)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]

            # orthogonalize against previous psi_j
            for j in range(ki):
                a = phi_2x_coeff[ki, : ki + 1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0.0
                idx = np.arange(len(prod_), dtype=np.float64)
                proj_ = (prod_ * 1.0 / (idx + 1.0) * np.power(0.5, 1.0 + idx)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            # normalize
            def _norm_from_coeff(c: Array, use_left_half: bool) -> float:
                prod = np.convolve(c, c)
                prod[np.abs(prod) < 1e-8] = 0.0
                idx2 = np.arange(len(prod), dtype=np.float64)
                if use_left_half:
                    # integral over [0, 0.5]
                    w = 1.0 / (idx2 + 1.0) * np.power(0.5, 1.0 + idx2)
                else:
                    # integral over [0.5, 1] (complement)
                    w = 1.0 / (idx2 + 1.0) * (1.0 - np.power(0.5, 1.0 + idx2))
                return float((prod * w).sum())

            norm1 = _norm_from_coeff(psi1_coeff[ki, :], use_left_half=True)
            norm2 = _norm_from_coeff(psi2_coeff[ki, :], use_left_half=False)
            norm_ = np.sqrt(norm1 + norm2)

            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_

            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0.0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0.0

        # convert coeffs to callable polynomials (np.poly1d uses descending powers)
        phi: list[BasisFn] = [
            np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)
        ]
        psi1: list[BasisFn] = [
            np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)
        ]
        psi2: list[BasisFn] = [
            np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)
        ]

    # ---------------------------------------------------------------------
    # Chebyshev-based basis
    # ---------------------------------------------------------------------
    else:  # base == 'chebyshev'
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, : ki + 1] = np.array([np.sqrt(2.0 / np.pi)], dtype=np.float64)
                phi_2x_coeff[ki, : ki + 1] = np.array(
                    [np.sqrt(2.0 / np.pi) * np.sqrt(2.0)], dtype=np.float64
                )
            else:
                coeff_ = Poly(chebyshevt(ki, 2 * x_sym - 1), x_sym).all_coeffs()  # type: ignore
                coeff_arr = np.array([float(c) for c in coeff_], dtype=np.float64)
                phi_coeff[ki, : ki + 1] = np.flip(2.0 / np.sqrt(np.pi) * coeff_arr)

                coeff_2 = Poly(chebyshevt(ki, 4 * x_sym - 1), x_sym).all_coeffs()  # type: ignore
                coeff_2_arr = np.array([float(c) for c in coeff_2], dtype=np.float64)
                phi_2x_coeff[ki, : ki + 1] = np.flip(
                    np.sqrt(2.0) * 2.0 / np.sqrt(np.pi) * coeff_2_arr
                )

        # scaling functions on [0,1]
        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]

        k_use = 2 * k
        roots = Poly(chebyshevt(k_use, 2 * x_sym - 1), x_sym).all_roots()  # type: ignore
        x_m = np.array([float(rt.evalf(20)) for rt in roots], dtype=np.float64)
        w_m = np.pi / k_use / 2.0

        psi1_coeff: Array = np.zeros((k, k), dtype=np.float64)
        psi2_coeff: Array = np.zeros((k, k), dtype=np.float64)

        psi1: list[BasisFn] = [lambda _x: np.zeros_like(x_m) for _ in range(k)]
        psi2: list[BasisFn] = [lambda _x: np.zeros_like(x_m) for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            psi2_coeff[ki, :] = phi_2x_coeff[ki, :]

            # project out phi_i
            for i in range(k):
                proj = (w_m * phi[i](x_m) * np.sqrt(2.0) * phi[ki](2.0 * x_m)).sum()
                psi1_coeff[ki, :] -= proj * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj * phi_coeff[i, :]

            # orthogonalize against previous psi_j
            for j in range(ki):
                proj = (w_m * psi1[j](x_m) * np.sqrt(2.0) * phi[ki](2.0 * x_m)).sum()
                psi1_coeff[ki, :] -= proj * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj * psi2_coeff[j, :]

            # build piecewise functions on [0, 0.5] and (0.5, 1]
            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0.0, ub=0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1.0)

            norm1 = float((w_m * psi1[ki](x_m) * psi1[ki](x_m)).sum())
            norm2 = float((w_m * psi2[ki](x_m) * psi2[ki](x_m)).sum())
            norm_ = np.sqrt(norm1 + norm2)

            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_

            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0.0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0.0

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0.0, ub=0.5 + 1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1.0)

    return phi, psi1, psi2


def get_filter(
    base: str,
    k: int,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Build multiresolution filter matrices (H0, H1, G0, G1, PHI0, PHI1).

    Args:
        base: 'legendre' or 'chebyshev'
        k: number of basis functions

    Returns:
        (H0, H1, G0, G1, PHI0, PHI1) each of shape (k, k)
    """
    if base not in ("legendre", "chebyshev"):
        raise ValueError(f"Base '{base}' not supported. Use 'legendre' or 'chebyshev'.")

    x_sym = Symbol("x")

    def psi_fn(
        psi1: Sequence[BasisFn],
        psi2: Sequence[BasisFn],
        i: int,
        inp: Array,
    ) -> Array:
        """Piecewise wavelet on [0, 0.5] and (0.5, 1]."""
        mask = (inp <= 0.5)
        return psi1[i](inp) * mask.astype(np.float64) + psi2[i](inp) * (~mask).astype(
            np.float64
        )

    H0: Array = np.zeros((k, k), dtype=np.float64)
    H1: Array = np.zeros((k, k), dtype=np.float64)
    G0: Array = np.zeros((k, k), dtype=np.float64)
    G1: Array = np.zeros((k, k), dtype=np.float64)
    PHI0: Array = np.zeros((k, k), dtype=np.float64)
    PHI1: Array = np.zeros((k, k), dtype=np.float64)

    phi, psi1, psi2 = get_phi_psi(k, base)

    # ------------------------------------------------------------------
    # Legendre
    # ------------------------------------------------------------------
    if base == "legendre":
        roots = Poly(legendre(k, 2 * x_sym - 1), x_sym).all_roots()  # type: ignore
        x_m = np.array([float(rt.evalf(20)) for rt in roots], dtype=np.float64)
        wm = 1.0 / k / legendreDer(k, 2.0 * x_m - 1.0) / eval_legendre(k - 1, 2.0 * x_m - 1.0)

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1.0 / np.sqrt(2.0) * (
                    wm * phi[ki](x_m / 2.0) * phi[kpi](x_m)
                ).sum()
                G0[ki, kpi] = 1.0 / np.sqrt(2.0) * (
                    wm * psi_fn(psi1, psi2, ki, x_m / 2.0) * phi[kpi](x_m)
                ).sum()
                H1[ki, kpi] = 1.0 / np.sqrt(2.0) * (
                    wm * phi[ki]((x_m + 1.0) / 2.0) * phi[kpi](x_m)
                ).sum()
                G1[ki, kpi] = 1.0 / np.sqrt(2.0) * (
                    wm * psi_fn(psi1, psi2, ki, (x_m + 1.0) / 2.0) * phi[kpi](x_m)
                ).sum()

        PHI0 = np.eye(k, dtype=np.float64)
        PHI1 = np.eye(k, dtype=np.float64)

    # ------------------------------------------------------------------
    # Chebyshev
    # ------------------------------------------------------------------
    else:  # base == 'chebyshev'
        k_use = 2 * k
        roots = Poly(chebyshevt(k_use, 2 * x_sym - 1), x_sym).all_roots()  # type: ignore
        x_m = np.array([float(rt.evalf(20)) for rt in roots], dtype=np.float64)
        w_m = np.pi / k_use / 2.0

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1.0 / np.sqrt(2.0) * (
                    w_m * phi[ki](x_m / 2.0) * phi[kpi](x_m)
                ).sum()
                G0[ki, kpi] = 1.0 / np.sqrt(2.0) * (
                    w_m * psi_fn(psi1, psi2, ki, x_m / 2.0) * phi[kpi](x_m)
                ).sum()
                H1[ki, kpi] = 1.0 / np.sqrt(2.0) * (
                    w_m * phi[ki]((x_m + 1.0) / 2.0) * phi[kpi](x_m)
                ).sum()
                G1[ki, kpi] = 1.0 / np.sqrt(2.0) * (
                    w_m * psi_fn(psi1, psi2, ki, (x_m + 1.0) / 2.0) * phi[kpi](x_m)
                ).sum()

                PHI0[ki, kpi] = float(
                    (w_m * phi[ki](2.0 * x_m) * phi[kpi](2.0 * x_m)).sum() * 2.0
                )
                PHI1[ki, kpi] = float(
                    (w_m * phi[ki](2.0 * x_m - 1.0) * phi[kpi](2.0 * x_m - 1.0)).sum()
                    * 2.0
                )

        PHI0[np.abs(PHI0) < 1e-8] = 0.0
        PHI1[np.abs(PHI1) < 1e-8] = 0.0

    # threshold tiny entries for numerical cleanliness
    for mat in (H0, H1, G0, G1):
        mat[np.abs(mat) < 1e-8] = 0.0

    return H0, H1, G0, G1, PHI0, PHI1
