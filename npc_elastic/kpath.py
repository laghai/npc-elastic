"""k-paths through high-symmetry points, in rad/m."""

from __future__ import annotations

import numpy as np

from .spec import KPath, Lattice


def high_symmetry(lattice: Lattice) -> dict[str, np.ndarray]:
    a = lattice.a * 1e-9
    if lattice.type == "hex":
        # her sweep.py / schematic.py: K a vertex, M an edge midpoint of the hexagonal BZ
        return {
            "G": np.array([0.0, 0.0]),
            "M": np.array([np.pi / a, np.pi / (a * np.sqrt(3))]),
            "K": np.array([4 * np.pi / (3 * a), 0.0]),
        }
    return {
        "G": np.array([0.0, 0.0]),
        "X": np.array([np.pi / a, 0.0]),
        "M": np.array([np.pi / a, np.pi / a]),
    }


def kpoints(kp: KPath, lattice: Lattice) -> list[tuple[float, float]]:
    """Her construction: n+1 points per segment, then repeated points dropped (rounded to 1 rad/m).
    G-M-K-G with n=9 gives 27 points; the closing G is dropped because it repeats the first."""
    if kp.k is not None:
        raw = [np.asarray(k, float) for k in kp.k]
    else:
        hs = high_symmetry(lattice)
        try:
            pts = [hs[p.upper().replace("GAMMA", "G")] for p in kp.points]
        except KeyError as e:
            raise ValueError(f"unknown point {e}; {lattice.type} has {sorted(hs)}") from None
        raw = [pts[0]]
        for p0, p1 in zip(pts[:-1], pts[1:], strict=True):
            raw += [p0 + (p1 - p0) * i / kp.n_per_segment for i in range(1, kp.n_per_segment + 1)]
    if not kp.dedupe:
        return [(float(k[0]), float(k[1])) for k in raw]
    seen, out = set(), []
    for k in raw:
        key = (round(k[0]), round(k[1]))
        if key not in seen:
            seen.add(key)
            out.append((float(k[0]), float(k[1])))
    return out
