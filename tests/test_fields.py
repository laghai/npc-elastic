"""outputs.fields: what the npz holds, in what units, and that the exported displacement is the
solved Bloch mode. Controls: the conjugate Bloch phase, the other mid-plane condition and a solve
with the two Bloch phases swapped must fail."""

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from npc_elastic.solver import CellSolver, solve_k  # noqa: E402
from npc_elastic.spec import ElasticSpec  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

A, T = 1000.0, 200.0  # nm, the E1 plate
K = (0.23 * np.pi / (A * 1e-9), 0.11 * np.pi / (A * 1e-9))  # generic k, rad/m
N = 6
A1, A2 = np.array([A, 0.0]), np.array([-A / 2, A * np.sqrt(3) / 2])  # nm


def plate_spec(sym):
    return ElasticSpec.model_validate(
        dict(
            geometry=dict(lattice=dict(type="hex", a=A), t=T, hole=dict(family="none")),
            material=dict(model="isotropic", E=1050e9, nu=0.2, rho=3515.0),
            symmetries=[sym],
            kpath=dict(k=[K]),
            mesh=dict(hmax=T, order=2),
            solver=dict(n_modes=N, shift_hz=0.0),
            outputs=dict(parity=False, fields=True),
        )
    )


@pytest.fixture(scope="module", params=["sym", "antisym", "full"])
def exported(request, tmp_path_factory):
    run = request.param
    spec = plate_spec(run)
    out = tmp_path_factory.mktemp(run)
    kr, _ = solve_k(CellSolver(spec, run), *K, spec.outputs, fields_dir=out, index=0)
    return run, kr, out / "fields" / f"{run}_k000"


def test_npz_contents_and_units(exported):
    run, kr, stem = exported
    d = np.load(f"{stem}.npz")
    xyz, u = d["xyz_nm"], d["u"]
    assert stem.with_suffix(".vtu").exists()
    assert str(d["run"]) == run
    assert u.shape == (N, len(xyz), 3) and np.iscomplexobj(u)
    assert np.allclose(np.abs(u).reshape(N, -1).max(axis=1), 1.0)  # max |u| = 1 per mode
    assert np.allclose(d["freqs_hz"], kr.freqs_hz) and np.allclose(d["k"], K)
    z0 = 0.0 if run != "full" else -T / 2  # nm: half slab is z in [0, t/2]
    assert np.isclose(xyz[:, 2].min(), z0) and np.isclose(xyz[:, 2].max(), T / 2)
    assert d["tets"].shape[1] == 4 and d["tets"].max() < len(xyz)


def bloch_residual(xyz, u, shift_nm, phase):
    """max |u(r + a) - phase u(r)| over vertex pairs on the two identified faces."""
    tree = cKDTree(xyz)
    dist, j = tree.query(xyz + np.r_[shift_nm, 0.0])
    i = np.flatnonzero(dist < 1e-6 * A)
    assert len(i) > 10
    return np.abs(u[:, j[i]] - phase * u[:, i]).max()


def test_bloch_condition_on_exported_field(exported):
    _, _, stem = exported
    d = np.load(f"{stem}.npz")
    k = np.array(K) * 1e-9  # rad/nm
    for a in (A1, A2):
        ph = np.exp(1j * k @ a)
        assert bloch_residual(d["xyz_nm"], d["u"], a, ph) < 1e-8
        # negative control: the conjugate phase is a different k and must not fit
        assert bloch_residual(d["xyz_nm"], d["u"], a, np.conj(ph)) > 1e-2


def test_mid_plane_condition_on_exported_field(exported):
    """sym: u_z = 0 on z = 0 (roller); antisym: u_x = u_y = 0. The other components are not zero
    there (negative control), so the check sees which condition was applied."""
    run, _, stem = exported
    if run == "full":
        pytest.skip("no mid-plane condition on the full slab")
    d = np.load(f"{stem}.npz")
    mid = np.isclose(d["xyz_nm"][:, 2], 0.0)
    u = d["u"][:, mid]
    held, free = ([2], [0, 1]) if run == "sym" else ([0, 1], [2])
    assert np.abs(u[..., held]).max() < 1e-10
    assert np.abs(u[..., free]).max() > 1e-1


def test_swapped_phases_fail_the_field_check(tmp_path):
    """The real bug fixed on this branch: phases in the wrong face order solve the mirror-image k.
    Every plate frequency is unchanged (to mesh error), so only the field check sees it."""
    spec = plate_spec("sym")
    cs = CellSolver(spec, "sym")
    good = cs.phases
    f_good = solve_k(cs, *K, spec.outputs, fields_dir=tmp_path / "good")[0].freqs_hz
    cs.phases = lambda kx, ky: good(kx, ky)[::-1]
    f_bad = solve_k(cs, *K, spec.outputs, fields_dir=tmp_path / "bad")[0].freqs_hz
    # same frequencies to discretization error (the mesh is not mirror-symmetric: 2e-5 here)
    assert np.allclose(f_good, f_bad, rtol=1e-4)
    d = np.load(tmp_path / "bad" / "fields" / "sym_k000.npz")
    ph = np.exp(1j * (np.array(K) * 1e-9) @ A1)
    assert bloch_residual(d["xyz_nm"], d["u"], A1, ph) > 1e-1
