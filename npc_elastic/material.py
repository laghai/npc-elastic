"""Stiffness in Voigt form (xx yy zz yz xz xy, engineering shear) and its rotation."""

from __future__ import annotations

import numpy as np


def isotropic_C(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    return C


def cubic_C(C11: float, C12: float, C44: float) -> np.ndarray:
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    C[np.arange(3), np.arange(3)] = C11
    C[np.arange(3, 6), np.arange(3, 6)] = C44
    return C


def bond_matrix(R: np.ndarray) -> np.ndarray:
    """6x6 M with C_lab = M C M^T, stress-like Voigt order (Auld, Acoustic Fields, 3.34)."""
    R = np.asarray(R, float)
    M = np.zeros((6, 6))
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for row, (i, j) in enumerate(idx):
        for col, (k, m) in enumerate(idx):
            if col < 3:
                M[row, col] = R[i, k] * R[j, m]
            else:
                M[row, col] = R[i, k] * R[j, m] + R[i, m] * R[j, k]
    return M


def rotate_C(C: np.ndarray, R: np.ndarray | None) -> np.ndarray:
    if R is None:
        return np.asarray(C, float)
    R = np.asarray(R, float)
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-9) or np.linalg.det(R) < 0:
        raise ValueError("orientation must be a proper rotation matrix")
    M = bond_matrix(R)
    return M @ np.asarray(C, float) @ M.T


def stiffness(material) -> tuple[np.ndarray, float, bool]:
    """(C_voigt in Pa, rho, is_isotropic)."""
    if material.model == "isotropic":
        return isotropic_C(material.E, material.nu), material.rho, True
    if material.model == "cubic":
        C = cubic_C(material.C11, material.C12, material.C44)
    else:
        C = np.asarray(material.C, float)
        if C.shape != (6, 6) or not np.allclose(C, C.T, rtol=1e-9):
            raise ValueError("C must be a symmetric 6x6 Voigt matrix")
    C = rotate_C(C, material.orientation)
    if np.linalg.eigvalsh(C).min() <= 0:
        raise ValueError("C is not positive definite")
    return C, material.rho, False
