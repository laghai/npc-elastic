"""E1 (free plate vs exact Rayleigh-Lamb, zone-folded) and E6 (sym + antisym = full), with the
negative controls that must fail. Small meshes: a screening tolerance here; E1's 1e-4 is the
converged-mesh number reported with the qualified run."""

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from npc_elastic.reference import folded_plate_frequencies, plate_frequencies  # noqa: E402
from npc_elastic.solver import CellSolver  # noqa: E402
from npc_elastic.spec import ElasticSpec  # noqa: E402

A, T = 1000.0, 200.0  # nm (the plan's Figure 1 plate)
E, NU, RHO = 1050e9, 0.2, 3515.0
# generic k, no mirror of the hex lattice maps it to itself: a swapped or sign-flipped phase
# lands on a different |k + G| set
K = (0.23 * np.pi / (A * 1e-9), 0.11 * np.pi / (A * 1e-9))
N = 8
TOL = 2e-3  # P3, hmax = t: screening mesh


def plate_spec(sym):
    return ElasticSpec.model_validate(
        dict(
            geometry=dict(lattice=dict(type="hex", a=A), t=T, hole=dict(family="none")),
            material=dict(model="isotropic", E=E, nu=NU, rho=RHO),
            symmetries=[sym],
            kpath=dict(k=[K]),
            mesh=dict(hmax=T, order=3),
            solver=dict(n_modes=N, shift_hz=0.0),
        )
    )


@pytest.fixture(scope="module")
def solvers():
    return {s: CellSolver(plate_spec(s), s) for s in ("sym", "antisym", "full")}


def exact(fmax=200e9):
    a1, a2 = np.array([A, 0]) * 1e-9, np.array([-A / 2, A * np.sqrt(3) / 2]) * 1e-9
    return folded_plate_frequencies(K, a1, a2, T * 1e-9, E, NU, RHO, fmax)


def test_reference_limits():
    """The reference itself: S0 -> plate velocity, A0 -> Kirchhoff flexural, at small k."""
    t = T * 1e-9
    k = 0.01 / t
    f = plate_frequencies(k, t, E, NU, RHO, fmax=5e9)
    cS0 = np.sqrt(E / (RHO * (1 - NU**2)))
    cSH = np.sqrt(E / (2 * (1 + NU)) / RHO)
    D = E * t**3 / (12 * (1 - NU**2))
    fA0 = np.sqrt(D / (RHO * t)) * k**2 / (2 * np.pi)
    assert abs(f["antisym"][0] / fA0 - 1) < 1e-3
    assert abs(f["sym"][0] / (cSH * k / 2 / np.pi) - 1) < 1e-9
    assert abs(f["sym"][1] / (cS0 * k / 2 / np.pi) - 1) < 1e-3


@pytest.mark.parametrize("run", ["sym", "antisym"])
def test_e1_lamb_half_slab(solvers, run):
    f = solvers[run].solve(*K).freqs_hz
    ex = np.array(exact()[run][:N])
    err = np.abs(f / ex - 1)
    assert err.max() < TOL, (run, f / 1e9, ex / 1e9)


def test_e1_negative_control_conjugated_phase(solvers):
    """Conjugate one Bloch phase (k.a2 -> -k.a2): the 60-degree basis has a different metric, so
    |k + G| changes and E1 must fail. (Swapping the two phases is a lattice mirror, invisible on a
    homogeneous plate and on any C3v cell under time reversal; the Bloch test below pins order.)"""
    cs = solvers["sym"]
    good = cs.phases

    def flipped(kx, ky):
        p = good(kx, ky)
        return [p[0], np.conj(p[1])]

    cs.phases = flipped
    try:
        f = cs.solve(*K).freqs_hz
    finally:
        cs.phases = good
    ex = np.array(exact()["sym"][:N])
    assert np.abs(f / ex - 1).max() > 10 * TOL


def test_bloch_condition_on_the_mode(solvers):
    """u(r + a_i) = exp(i k.a_i) u(r) on the identified faces, for both lattice vectors."""
    assert_bloch(solvers["full"])


def assert_bloch(cs):
    md = cs.solve(*K, n=4, vectors=True)
    gf = cs.gridfunctions(md)[3]
    rng = np.random.default_rng(0)
    c = cs.cell
    kvec = np.array(K) * 1e-6  # rad per model unit (um)
    for vec, along in ((c.a1, c.a2), (c.a2, c.a1)):
        ph = np.exp(1j * kvec @ vec)
        s = rng.uniform(0.1, 0.9, 20)
        z = rng.uniform(c.z_bottom, c.z_top, 20) * 0.9
        src = np.c_[np.outer(s, along), z]
        dst = src + np.r_[vec, 0]
        us, ud = cs._eval(gf, src), cs._eval(gf, dst)
        assert np.allclose(ud, ph * us, atol=1e-8 * np.abs(us).max()), (ud / us)[:2]


def test_e6_union_equals_full(solvers):
    fs = solvers["sym"].solve(*K, n=N).freqs_hz
    fa = solvers["antisym"].solve(*K, n=N).freqs_hz
    ff = solvers["full"].solve(*K, n=N).freqs_hz
    union = np.sort(np.r_[fs, fa])[:N]
    assert np.abs(ff / union - 1).max() < TOL
    # negative control: one symmetry alone is not the full spectrum
    assert np.abs(ff / np.sort(fs)[:N] - 1).max() > 10 * TOL


def test_full_run_parity_matches_union(solvers):
    cs = solvers["full"]
    md = cs.solve(*K, n=N, vectors=True)
    pz = np.array(cs.parity(cs.gridfunctions(md), *K, "z", cs.sample_points(1500)))
    assert np.all(np.abs(np.abs(pz) - 1) < 1e-2), pz
    ex = exact()
    labels = sorted([(f, 1) for f in ex["sym"][:N]] + [(f, -1) for f in ex["antisym"][:N]])[:N]
    assert [int(np.sign(p)) for p in pz] == [s for _, s in labels]


def test_mirror_selection(solvers):
    """sigma_y on Gamma-K (ky = 0) and at Gamma; never at a generic k (the unit bug this caught
    reported mirrors everywhere)."""
    cs = solvers["full"]
    a = A * 1e-9
    assert cs.mirror_ok("y", 0.7 * np.pi / a, 0.0)
    assert cs.mirror_ok("x", 0.0, 0.0) and cs.mirror_ok("y", 0.0, 0.0)
    assert not cs.mirror_ok("y", *K)
    assert not cs.mirror_ok("x", *K)
    assert cs.mirror_ok("z", *K) and not solvers["sym"].mirror_ok("z", *K)


def test_uniform_refine_converges_and_keeps_periodicity():
    """mesh.refine: one uniform refinement of a P2 plate must cut the E1 error (positive control:
    P2 converges at h^4 on this plate), and the refined mesh must keep the periodic identification
    (the Bloch condition on the mode, as above; a lost identification fails it)."""

    def p2(refine, run="sym"):
        s = plate_spec(run).model_dump(by_alias=True)
        s["mesh"] = dict(hmax=T, order=2, refine=refine)
        return CellSolver(ElasticSpec.model_validate(s), run)

    ref = np.array(exact()["sym"][:N])
    coarse, fine = p2(0), p2(1)
    err = [np.abs(cs.solve(*K).freqs_hz / ref - 1).max() for cs in (coarse, fine)]
    assert err[1] < err[0] / 4, err
    assert fine.mesh.ne == 8 * coarse.mesh.ne
    assert_bloch(p2(1, "full"))
