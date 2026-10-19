"""Checks that need no FEM engine: spec, schema, k-path, material algebra, CSV format."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from npc_elastic.kpath import high_symmetry, kpoints
from npc_elastic.material import cubic_C, isotropic_C, rotate_C
from npc_elastic.spec import ElasticSpec

HERE = Path(__file__).parent
EX = HERE.parent / "examples" / "diamond_mech_v1.json"  # design value, rho 3500
EX_REPRO = HERE.parent / "examples" / "diamond_mech_v1_repro_rho2329.json"  # her COMSOL CSVs


def spec():
    return ElasticSpec.model_validate_json(EX.read_text())


def test_schema_is_current():
    from npc_elastic.schema import SCHEMAS, schema_text

    for name in SCHEMAS:
        assert (HERE.parent / "schema" / name).read_text() == schema_text(name), name


def test_example_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((HERE.parent / "schema" / "elastic-spec.v0.schema.json").read_text())
    jsonschema.validate(json.loads(EX.read_text()), schema)
    bad = json.loads(EX.read_text())
    bad["geometry"]["hole"]["family"] = "hexagram"  # negative control
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_density_examples_differ_only_in_rho():
    design, repro = json.loads(EX.read_text()), json.loads(EX_REPRO.read_text())
    assert (design["material"]["rho"], repro["material"]["rho"]) == (3500, 2329)
    design["material"]["rho"] = repro["material"]["rho"]
    assert design == repro
    ElasticSpec.model_validate(repro)


def test_her_kpath_27_points():
    s = spec()
    k = np.array(kpoints(s.kpath, s.geometry.lattice))
    assert len(k) == 27
    a = 300e-9
    hs = high_symmetry(s.geometry.lattice)
    assert np.allclose(k[0], 0) and np.allclose(k[9], hs["M"]) and np.allclose(k[18], hs["K"])
    assert np.allclose(hs["M"], [np.pi / a, np.pi / (np.sqrt(3) * a)])
    # M is the midpoint of b1 for a1=(a,0), a2=(-a/2, a sqrt3/2)
    a1, a2 = np.array([a, 0]), np.array([-a / 2, a * np.sqrt(3) / 2])
    assert np.allclose(np.exp(1j * (2 * hs["M"]) @ np.array([a1, a2]).T), 1)
    assert not np.allclose(np.exp(1j * hs["M"] @ np.array([a1, a2]).T), 1)


def test_isotropic_C_is_rotation_invariant():
    C = isotropic_C(1050e9, 0.07)
    th = 0.37
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert np.allclose(rotate_C(C, R), C, rtol=1e-12, atol=1)


def test_cubic_rotated_45_about_z():
    # textbook: C11' = (C11 + C12)/2 + C44, C66' = (C11 - C12)/2 for a 45 deg rotation about [001]
    C11, C12, C44 = 1075e9, 149e9, 567e9
    t = np.pi / 4
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    Cr = rotate_C(cubic_C(C11, C12, C44), R)
    assert np.isclose(Cr[0, 0], (C11 + C12) / 2 + C44)
    assert np.isclose(Cr[5, 5], (C11 - C12) / 2)
    assert np.isclose(Cr[2, 2], C11)
    with pytest.raises(ValueError):  # negative control: a reflection is refused
        rotate_C(cubic_C(C11, C12, C44), np.diag([1.0, 1.0, -1.0]))


def test_csv_roundtrip_her_format(tmp_path):
    from npc_elastic.io import read_csv, write_csv
    from npc_elastic.spec import ElasticResult, KResult, MeshInfo, RunResult

    s = spec()
    runs = [
        RunResult(
            symmetry=sym,
            ndof=1,
            seconds=0.0,
            k=[
                KResult(kx=0.0, ky=0.0, freqs_hz=[0.0, 2.0e10]),
                KResult(kx=1.0e6, ky=0.0, freqs_hz=[1.0e9, 2.1e10]),
            ],
        )
        for sym in ("sym", "antisym")
    ]
    res = ElasticResult(
        spec=s, runs=runs, mesh=MeshInfo(hmax_nm=1, order=2, n_elements=1, n_vertices=1), engine={}
    )
    paths = write_csv(res, tmp_path)
    assert [p.name for p in paths] == [
        "Python_diamond_Gamma_sym.csv",
        "Python_diamond_Gamma_antisym.csv",
    ]
    rows = list(csv.reader(open(paths[0])))
    assert rows[1] == ["0.0", "0.0", "20000000000.0", "20000000000.0", ""]  # her exact row shape
    assert read_csv(paths[0], n_modes=2)[1] == (1.0e6, 0.0, [1.0e9, 2.1e10])
