"""npc_elastic: Bloch-periodic elastic eigenmodes of a 3D unit cell (open COMSOL Solid Mechanics ->
Eigenfrequency, for phononic crystal band structures).

    from npc_elastic import run
    result = run("spec.json", out_dir="results/")   # result JSON + her CSVs

Spec and result schema: elastic/schema/ (v0). Engine: NGSolve + SciPy ARPACK.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .spec import ElasticResult, ElasticSpec  # noqa: E402


def engine_versions() -> dict[str, str]:
    import ngsolve
    import scipy

    return {"npc_elastic": __version__, "ngsolve": ngsolve.__version__, "scipy": scipy.__version__}


def run(spec, out_dir=None, progress=None, kpoints=None, workers=1) -> ElasticResult:
    """Solve a spec (ElasticSpec, dict, or JSON path). Writes result.json and her CSVs to out_dir.
    kpoints overrides the spec's k-path (the cloud lane uses it to split k across tasks);
    workers > 1 solves k-points in that many processes."""
    from pathlib import Path

    from .io import write_csv, write_json
    from .kpath import kpoints as make_kpoints
    from .solver import run_spec

    if isinstance(spec, (str, Path)):
        spec = ElasticSpec.model_validate_json(Path(spec).read_text())
    elif isinstance(spec, dict):
        spec = ElasticSpec.model_validate(spec)
    ks = kpoints if kpoints is not None else make_kpoints(spec.kpath, spec.geometry.lattice)
    fields_dir = out_dir if spec.outputs.fields else None
    runs, mesh = run_spec(spec, ks, progress=progress, workers=workers, fields_dir=fields_dir)
    res = ElasticResult(spec=spec, runs=runs, mesh=mesh, engine=engine_versions())
    if out_dir is not None:
        write_json(res, Path(out_dir) / "result.json")
        write_csv(res, out_dir)
    return res


__all__ = ["run", "ElasticSpec", "ElasticResult", "engine_versions", "__version__"]
