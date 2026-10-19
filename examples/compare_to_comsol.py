"""Compare npc_elastic's band CSVs with COMSOL's, mode by mode.

    python compare_to_comsol.py OURS_DIR COMSOL_DIR [--prefix Python_diamond_Gamma] [--tol 1e-3]
                                [--scale S]

Both dirs hold <prefix>_{sym,antisym}.csv (columns kx, ky, f, f). For each symmetry and k the
frequencies above 1 MHz (the rigid modes dropped, as plot_bands.py does) are sorted and compared by
index. Prints median and max |ours/comsol - 1| and exits 1 if max > tol or the k-points differ.

--scale multiplies ours first. It is the negative control: --scale 0.8158 (= sqrt(2329/3500), the
same cell at diamond's density instead of the 2329 her COMSOL used) must FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from npc_elastic.io import read_csv


def compare(ours: Path, comsol: Path, prefix: str, scale: float = 1.0) -> dict:
    out = {}
    for sym in ("sym", "antisym"):
        a = read_csv(ours / f"{prefix}_{sym}.csv")
        b = read_csv(comsol / f"{prefix}_{sym}.csv")
        if [k[:2] for k in a] != [k[:2] for k in b] and not np.allclose(
            [k[:2] for k in a], [k[:2] for k in b], rtol=1e-9, atol=1e-3
        ):
            raise SystemExit(f"{sym}: k-points differ ({len(a)} vs {len(b)})")
        rel = []
        for (_, _, fa), (_, _, fb) in zip(a, b, strict=True):
            fa = np.array([f for f in fa if f > 1e6]) * scale
            fb = np.array([f for f in fb if f > 1e6])
            n = min(len(fa), len(fb))
            rel.extend(np.abs(fa[:n] / fb[:n] - 1))
        rel = np.array(rel)
        out[sym] = {"n_k": len(a), "n_modes": len(rel), "median": float(np.median(rel)),
                    "max": float(rel.max())}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ours", type=Path)
    p.add_argument("comsol", type=Path)
    p.add_argument("--prefix", default="Python_diamond_Gamma")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--scale", type=float, default=1.0)
    a = p.parse_args(argv)
    res = compare(a.ours, a.comsol, a.prefix, a.scale)
    worst = 0.0
    for sym, r in res.items():
        print(f"{sym:8s} {r['n_k']} k, {r['n_modes']} modes: median {r['median']:.2e}, "
              f"max {r['max']:.2e}")
        worst = max(worst, r["max"])
    ok = worst <= a.tol
    print(f"{'PASS' if ok else 'FAIL'}: max {worst:.2e} vs tol {a.tol:.0e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
