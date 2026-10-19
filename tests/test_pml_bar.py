"""Stage 3 groundwork: a complex-stretching PML gives the exact complex frequency of a leaky elastic
resonator. nu = 0 bar (longitudinal waves exactly 1D; roller sides), clamped at x = 0, impedance Z1
on [0, L], Z2 = Z1 / r beyond, then a PML. Outgoing wave (e^{-i w t}): tan(w L / c1) = -i r, so
w_n = (c1 / L) (n pi - i artanh r), Q_n = n pi / (2 artanh r).

Controls that must fail: no PML (clamped far end) gives real w; the PML on NGSolve's VectorH1
Grad (6.2.2608 does not transform it) gives the wrong w. Probe and numbers:
reports/phononics_q/probe_pml_bar.py."""

import numpy as np
import pytest

pytest.importorskip("ngsolve")

import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sla  # noqa: E402
from netgen.occ import Box, Glue, OCCGeometry, Pnt, X  # noqa: E402
from ngsolve import H1, BilinearForm, FESpace, Grad, Mesh, VectorH1, dx, grad, pml  # noqa: E402

L, W, L2, LP = 1.0, 0.1, 1.0, 1.0  # um
R = 0.5  # Z1 / Z2; same wave speed c = 1 on both sides


def exact(n):
    return n * np.pi - 1j * np.arctanh(R)


@pytest.fixture(scope="module")
def mesh():
    b1 = Box(Pnt(0, 0, 0), Pnt(L, W, W)).mat("m1")
    b2 = Box(Pnt(L, 0, 0), Pnt(L + L2, W, W)).mat("m2")
    b3 = Box(Pnt(L + L2, 0, 0), Pnt(L + L2 + LP, W, W)).mat("pml")
    shape = Glue([b1, b2, b3])
    shape.faces.Min(X).name = "clamp"
    shape.faces.Max(X).name = "end"
    for f in shape.faces:
        if f.name not in ("clamp", "end"):
            f.name = "side"
    return Mesh(OCCGeometry(shape).GenerateMesh(maxh=0.1))


def resonances(mesh, use_pml=True, space="h1x3", targets=(1, 2)):
    mesh.UnSetPML("pml")
    if use_pml:
        mesh.SetPML(pml.Cartesian(mins=(-1, -1, -1), maxs=(L + L2, 1, 1), alpha=2j), "pml")
    E = mesh.MaterialCF({"m1": 1.0, "m2": 1 / R, "pml": 1 / R})
    rho = mesh.MaterialCF({"m1": 1.0, "m2": 1 / R, "pml": 1 / R})
    x_bc = "clamp" + ("" if use_pml else "|end")
    kw = dict(order=3, complex=True)
    if space == "h1x3":
        fes = FESpace(
            [H1(mesh, dirichlet=x_bc, **kw)] + [H1(mesh, dirichlet="clamp|side|end", **kw)] * 2
        )
        u, v = fes.TnT()
        gu, gv = [grad(c) for c in u], [grad(c) for c in v]
    else:
        fes = VectorH1(
            mesh, dirichletx=x_bc, dirichlety="clamp|side|end", dirichletz="clamp|side|end", **kw
        )
        u, v = fes.TnT()
        Gu, Gv = Grad(u), Grad(v)
        gu = [[Gu[i, j] for j in range(3)] for i in range(3)]
        gv = [[Gv[i, j] for j in range(3)] for i in range(3)]

    def eps(g, i, j):
        return 0.5 * (g[i][j] + g[j][i])

    K = BilinearForm(fes)
    K += E * sum(eps(gu, i, j) * eps(gv, i, j) for i in range(3) for j in range(3)) * dx  # 2 mu
    M = BilinearForm(fes)
    M += rho * sum(u[i] * v[i] for i in range(3)) * dx
    K.Assemble()
    M.Assemble()
    free = np.flatnonzero(np.array(list(fes.FreeDofs())))

    def csr(m):
        i, j, s = m.COO()
        A = sp.csr_matrix((np.array(s), (np.array(i), np.array(j))), shape=(m.height, m.width))
        return A[free][:, free]

    Ks, Ms = csr(K.mat), csr(M.mat)
    out = []
    for n in targets:
        # standard problem on (K - s M)^-1 M: scipy's generalized eigs assumes M Hermitian positive
        s0 = exact(n).real ** 2
        lu = sla.splu((Ks - s0 * Ms).tocsc())
        op = sla.LinearOperator(Ks.shape, matvec=lambda b, lu=lu: lu.solve(Ms @ b), dtype=complex)
        w = np.sqrt(s0 + 1 / sla.eigs(op, k=12, which="LM", return_eigenvectors=False))
        w = np.where(w.real < 0, -w, w)
        out.append(w[np.argmin(np.abs(w - exact(n)))])
    return out


def q(w):
    return w.real / (2 * abs(w.imag)) if w.imag != 0 else np.inf


def test_pml_gives_exact_complex_frequency(mesh):
    for n, w in zip((1, 2), resonances(mesh), strict=True):
        assert abs(w / exact(n) - 1) < 1e-4, (n, w)
        assert abs(q(w) / q(exact(n)) - 1) < 1e-3, (n, q(w))  # measured 5e-5, 9e-5


def test_no_pml_is_lossless(mesh):
    for w in resonances(mesh, use_pml=False):
        assert q(w) > 1e6


def test_vectorh1_grad_is_not_pml_transformed(mesh):
    """Pins the NGSolve trap: if this starts passing exact, VectorH1 may be used for the PML."""
    w = resonances(mesh, space="vectorh1", targets=(1,))[0]
    assert abs(w / exact(1) - 1) > 1e-2, w
