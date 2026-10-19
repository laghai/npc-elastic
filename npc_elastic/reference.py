"""Exact free-plate dispersion (Rayleigh-Lamb + SH), zone-folded into a periodic cell (check E1).

Plate of thickness t = 2h, isotropic, free faces. With p^2 = w^2/cL^2 - k^2, q^2 = w^2/cT^2 - k^2
(Graff, Wave Motion in Elastic Solids, 8.1; Auld vol. II, 10.C), written analytic in p^2, q^2
so that real roots need no branch handling:
    S (symmetric, u_z odd in z):  (q^2-k^2)^2 cos(ph) sin(qh)/q + 4 k^2 p sin(ph) cos(qh) = 0
    A (antisymmetric):            (q^2-k^2)^2 sin(ph)/p cos(qh) + 4 k^2 q cos(ph) sin(qh) = 0
    SH_n:                         w^2 = cT^2 (k^2 + (n pi / t)^2); n even symmetric, odd anti
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _sinc_like(x2: complex, h: float):
    """cos(xh), sin(xh)/x, x sin(xh) for x = sqrt(x2), all real for real x2."""
    x = np.sqrt(complex(x2))
    if abs(x) < 1e-14:
        return 1.0, h, 0.0
    return (np.cos(x * h).real, (np.sin(x * h) / x).real, (x * np.sin(x * h)).real)


def _F(w: float, k: float, h: float, cL: float, cT: float, sym: bool) -> float:
    p2 = (w / cL) ** 2 - k**2
    q2 = (w / cT) ** 2 - k**2
    cp, sp_p, psp = _sinc_like(p2, h)
    cq, sq_q, qsq = _sinc_like(q2, h)
    if sym:
        return (q2 - k**2) ** 2 * cp * sq_q + 4 * k**2 * psp * cq
    return (q2 - k**2) ** 2 * sp_p * cq + 4 * k**2 * cp * qsq


def lamb_omegas(k: float, t: float, cL: float, cT: float, wmax: float, sym: bool, n_grid=4000):
    """Real roots w in (0, wmax] of the S (sym=True) or A Rayleigh-Lamb equation at wavenumber k."""
    h = t / 2
    if k == 0:
        # k = 0: thickness resonances only (rigid modes at 0 Hz are added by the caller)
        out = []
        n = 1
        while True:
            added = False
            for c in (cL, cT):
                # S: cos(ph) sin(qh)=0 -> L: p h = (n-1/2) pi  [u_z odd]; T: q h = n pi
                # A: sin(ph) cos(qh)=0 -> L: p h = n pi; T: q h = (n-1/2) pi
                if (c is cL) == sym:
                    w = (n - 0.5) * np.pi * c / h
                else:
                    w = n * np.pi * c / h
                if w <= wmax:
                    out.append(w)
                    added = True
            if not added:
                break
            n += 1
        return sorted(out)
    ws = np.linspace(wmax * 1e-6, wmax, n_grid)
    F = np.array([_F(w, k, h, cL, cT, sym) for w in ws])
    roots = []
    for i in np.flatnonzero(np.sign(F[:-1]) * np.sign(F[1:]) < 0):
        roots.append(brentq(_F, ws[i], ws[i + 1], args=(k, h, cL, cT, sym), xtol=1e-14 * wmax))
    return roots


def plate_frequencies(k: float, t: float, E: float, nu: float, rho: float, fmax: float):
    """{'sym': [...], 'antisym': [...]} exact frequencies (Hz) <= fmax at wavenumber |k| (rad/m)."""
    G = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    cL, cT = np.sqrt((lam + 2 * G) / rho), np.sqrt(G / rho)
    wmax = 2 * np.pi * fmax
    out = {}
    for sym in (True, False):
        w = list(lamb_omegas(k, t, cL, cT, wmax, sym))
        n = 0 if sym else 1
        while True:
            wsh = cT * np.sqrt(k**2 + (n * np.pi / t) ** 2)
            if wsh > wmax:
                break
            w.append(wsh)
            n += 2
        out["sym" if sym else "antisym"] = sorted(x / (2 * np.pi) for x in w)
    return out


def folded_plate_frequencies(kvec, a1, a2, t, E, nu, rho, fmax, nG=4):
    """All exact plate frequencies <= fmax of a periodic cell (lattice a1, a2, m) at Bloch k:
    the plate modes at k + G for every reciprocal vector G. Rigid modes at Gamma included (0 Hz)."""
    A = np.array([a1, a2], float)
    B = 2 * np.pi * np.linalg.inv(A).T  # rows b1, b2 with a_i . b_j = 2 pi delta_ij
    out = {"sym": [], "antisym": []}
    for m1 in range(-nG, nG + 1):
        for m2 in range(-nG, nG + 1):
            kk = np.asarray(kvec, float) + m1 * B[0] + m2 * B[1]
            kn = float(np.linalg.norm(kk))
            fr = plate_frequencies(kn, t, E, nu, rho, fmax)
            if kn < 1e-9 * np.linalg.norm(B[0]):
                fr["sym"] = [0.0, 0.0] + fr["sym"]  # u_x, u_y translations
                fr["antisym"] = [0.0] + fr["antisym"]  # u_z translation
            for s in out:
                out[s] += fr[s]
    return {s: sorted(v) for s, v in out.items()}
