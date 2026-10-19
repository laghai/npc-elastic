"""Outputs: result JSON, and Abby's COMSOL CSV so her plot_bands.py reads ours unchanged.

Her rows (runs/diamond_mech_v1/sweep.py): `kx, ky, f, f,` -- k in rad/m, f the real
eigenfrequency in Hz written twice, a trailing empty field, n_modes rows per k, no header.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .spec import ElasticResult


def write_csv(result: ElasticResult, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for run in result.runs:
        p = out / f"{result.spec.outputs.csv_prefix}_{run.symmetry}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            for k in run.k:
                for f in k.freqs_hz:
                    w.writerow([f"{k.kx}", f"{k.ky}", f"{f}", f"{f}", ""])
        paths.append(p)
    return paths


def read_csv(
    path: str | Path, n_modes: int | None = None
) -> list[tuple[float, float, list[float]]]:
    """[(kx, ky, sorted freqs)] in file order. Consecutive rows with the same k form one k-point."""
    out: list[tuple[float, float, list[float]]] = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue
            kx, ky, f = float(row[0]), float(row[1]), abs(float(row[2]))
            if out and (kx, ky) == out[-1][:2] and (n_modes is None or len(out[-1][2]) < n_modes):
                out[-1][2].append(f)
            else:
                out.append((kx, ky, [f]))
    return [(kx, ky, sorted(fs)) for kx, ky, fs in out]


def write_json(result: ElasticResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.model_dump(by_alias=True), indent=1) + "\n")
    return p


def read_json(path: str | Path) -> ElasticResult:
    return ElasticResult.model_validate_json(Path(path).read_text())
