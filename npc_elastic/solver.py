"""Bloch-periodic linear-elastic eigenmodes of one unit cell (NGSolve + SciPy shift-invert).

Weak form: find u in the quasi-periodic space, u(r + a_i) = exp(i k.a_i) u(r), with
    int eps(u) : C : eps(v)* dV = w^2 int rho u . v* dV.
Top face free. Half slab (z in [0, t/2]): 'sym' clamps u_z on z=0 (COMSOL Symmetry, in-plane
modes); 'antisym' clamps u_x, u_y (COMSOL Antisymmetry, flexural modes). 'full': z in [-t/2, t/2],
both faces free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .geometry import NM, UM, build_cell, hole_shapely, mesh_cell
from .material import stiffness
from .spec import ElasticSpec, KResult

_VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _voigt(e):
    return [e[i, j] if i == j else 2 * e[i, j] for i, j in _VOIGT]


def _csr(mat) -> sp.csr_matrix:
    d, i, p = mat.CSR()
    return sp.csr_matrix((np.array(d), np.array(i), np.array(p)))


@dataclass
class Modes:
    freqs_hz: np.ndarray
    vectors: np.ndarray | None  # free-dof eigenvectors, columns
    complete: bool
    fes: object = None
    free: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


class CellSolver:
    """Mesh once, then solve any k. One instance per (geometry, symmetry run)."""

    def __init__(self, spec: ElasticSpec, run: str, hmax_nm: float | None = None):
        from ngsolve import CF, Grad, InnerProduct, Sym

        self.spec, self.run = spec, run
        g = spec.geometry
        self.hmax_nm = hmax_nm or spec.mesh.hmax or g.t / spec.mesh.hmax_over_t
        self.order = spec.mesh.order
        self.cell = build_cell(g, half=(run != "full"))
        self.mesh = mesh_cell(self.cell, self.hmax_nm, self.order, spec.mesh.refine)
        C, rho, iso = stiffness(spec.material)
        self.C, self.rho, self.iso = C, rho, iso

        def strain_energy(u, v):
            eu, ev = Sym(Grad(u)), Sym(Grad(v))
            if iso:
                lam, mu = C[0, 1], C[3, 3]
                return 2 * mu * InnerProduct(eu, ev) + lam * (eu[0, 0] + eu[1, 1] + eu[2, 2]) * (
                    ev[0, 0] + ev[1, 1] + ev[2, 2]
                )
            Vu, Vv = _voigt(eu), _voigt(ev)
            terms = [
                float(C[i, j]) * Vu[j] * Vv[i] for i in range(6) for j in range(6) if C[i, j] != 0
            ]
            return sum(terms[1:], terms[0])

        self._strain_energy = strain_energy
        self._CF = CF

    # ------------------------------------------------------------------------------------
    def ident_vectors(self) -> list[np.ndarray]:
        """Translation of each periodic identification, in netgen's numbering (which is NOT the
        build order: per_a1 is built first and numbered 2). Read from the mesh's vertex pairs."""
        if not hasattr(self, "_ident"):
            ng = self.mesh.ngmesh
            P = np.array(ng.Coordinates())
            ids = np.array([[r[0].nr, r[1].nr, int(r[2])] for r in ng.GetIdentifications()])
            self._ident = []
            for n in range(1, ng.GetNrIdentifications() + 1):
                d = P[ids[ids[:, 2] == n, 1] - 1] - P[ids[ids[:, 2] == n, 0] - 1]
                if np.ptp(d, axis=0).max() > 1e-9:
                    raise RuntimeError(f"identification {n} is not a single translation")
                self._ident.append(d[0, :2])
        return self._ident

    def phases(self, kx: float, ky: float) -> list[complex]:
        """Bloch factor exp(i k.d) for each identification, in netgen's order."""
        k = np.array([kx, ky]) * UM  # rad per model unit
        return [np.exp(1j * float(k @ d)) for d in self.ident_vectors()]

    def space(self, kx: float, ky: float):
        from ngsolve import Periodic, VectorH1

        kw = {}
        if self.run == "sym":
            kw["dirichletz"] = "mid"
        elif self.run == "antisym":
            kw["dirichletx"] = "mid"
            kw["dirichlety"] = "mid"
        base = VectorH1(self.mesh, order=self.order, complex=True, **kw)
        fes = Periodic(base, phase=self.phases(kx, ky))
        # Periodic() drops the base space's component-wise Dirichlet flags (NGSolve 6.2.2608):
        # 327 free dofs with or without dirichletz on a test cell. Combine the masks ourselves.
        fes.free_mask = np.array(list(fes.FreeDofs())) & np.array(list(base.FreeDofs()))
        return fes

    def assemble(self, fes):
        from ngsolve import BilinearForm, InnerProduct, dx

        u, v = fes.TnT()
        K = BilinearForm(fes)
        K += self._strain_energy(u, v) * dx
        M = BilinearForm(fes)
        M += self.rho * InnerProduct(u, v) * dx
        K.Assemble()
        M.Assemble()
        free = np.flatnonzero(fes.free_mask)
        Ks = _csr(K.mat)[free][:, free]
        Ms = _csr(M.mat)[free][:, free]
        return Ks, Ms, free

    def _shift_invert(self, fes, free, sigma):
        """(K - sigma M)^-1 on free dofs via NGSolve UMFPACK. SuperLU's ordering fills ~40x more
        (150 s vs ~1 s); NGSolve 'sparsecholesky' is complex-symmetric, so wrong for Bloch."""
        from ngsolve import BilinearForm, InnerProduct, dx

        u, v = fes.TnT()
        A = BilinearForm(fes)
        A += (self._strain_energy(u, v) - sigma * self.rho * InnerProduct(u, v)) * dx
        A.Assemble()
        from ngsolve import BitArray

        mask = BitArray(len(fes.free_mask))
        mask.Clear()
        for i in free:
            mask.Set(int(i))
        inv = A.mat.Inverse(mask, inverse="umfpack")
        x, y = A.mat.CreateColVector(), A.mat.CreateColVector()
        xv, yv = x.FV().NumPy(), y.FV().NumPy()

        def mv(b):
            xv[:] = 0
            xv[free] = np.ravel(b)
            y.data = inv * x
            return yv[free].copy()

        return sla.LinearOperator((len(free), len(free)), matvec=mv, dtype=complex)

    def solve(
        self,
        kx: float,
        ky: float,
        n: int | None = None,
        shift_hz: float | None = None,
        vectors: bool = False,
    ) -> Modes:
        n = n or self.spec.solver.n_modes
        fs = self.spec.solver.shift_hz if shift_hz is None else shift_hz
        fes = self.space(kx, ky)
        Ks, Ms, free = self.assemble(fes)
        herm = sp.linalg.norm(Ks - Ks.conj().T) / sp.linalg.norm(Ks)
        if herm > 1e-10:
            raise RuntimeError(
                f"stiffness matrix not Hermitian ({herm:.1e}); check the Bloch phases"
            )
        # model-unit omega^2 = (2 pi f)^2 * UM^2
        to_w2 = lambda f: (2 * np.pi * f * UM) ** 2  # noqa: E731
        m = min(n + max(8, n // 2), Ks.shape[0] - 2)
        sigma = to_w2(fs)
        op = self._shift_invert(fes, free, sigma)
        while True:
            w2, vec = sla.eigsh(Ks, k=m, M=Ms, sigma=sigma, which="LM", OPinv=op, tol=1e-12)
            f = np.sign(w2.real) * np.sqrt(np.abs(w2.real)) / (2 * np.pi * UM)
            # the m eigenvalues nearest sigma fill [sigma - r, sigma + r] in omega^2, no gaps
            r = np.max(np.abs(w2.real - sigma))
            lo = np.sqrt(max(sigma - r, 0.0)) / (2 * np.pi * UM) if sigma - r > 0 else -np.inf
            hi = np.sqrt(sigma + r) / (2 * np.pi * UM)
            dist = np.abs(f - fs)
            order = np.argsort(dist)[:n]
            dmax = dist[order].max()
            complete = (fs - dmax >= lo or lo == -np.inf) and fs + dmax <= hi
            if complete or m >= Ks.shape[0] - 2 or m >= 4 * n:
                break
            m = min(2 * m, Ks.shape[0] - 2)
        idx = order[np.argsort(f[order])]
        return Modes(np.abs(f[idx]), vec[:, idx] if vectors else None, bool(complete), fes, free)

    # ------------------------------------------------------------------------------------
    def gridfunctions(self, modes: Modes):
        from ngsolve import GridFunction

        out = []
        for j in range(modes.vectors.shape[1]):
            gf = GridFunction(modes.fes)
            x = np.zeros(modes.fes.ndof, complex)
            x[modes.free] = modes.vectors[:, j]
            gf.vec.FV().NumPy()[:] = x
            out.append(gf)
        return out

    def sample_points(self, max_points: int = 3000) -> np.ndarray:
        """Element centroids (model units), a deterministic subset."""
        ng = self.mesh.ngmesh
        P = np.array(ng.Coordinates())
        cents = np.array(
            [P[[v.nr - 1 for v in el.vertices]].mean(axis=0) for el in ng.Elements3D()]
        )
        step = max(1, len(cents) // max_points)
        return cents[::step]

    def _fold(self, pts: np.ndarray, kx: float, ky: float):
        """Map in-plane points into the cell; return folded points and Bloch factors exp(i k.R)."""
        B = np.column_stack([self.cell.a1, self.cell.a2])
        s = np.linalg.solve(B, pts[:, :2].T).T
        nshift = np.floor(s + 1e-12)
        R = nshift @ B.T
        out = pts.copy()
        out[:, :2] -= R
        k = np.array([kx, ky]) * UM
        return out, np.exp(1j * (R @ k))

    def _eval(self, gf, pts):
        mp = self.mesh(pts[:, 0], pts[:, 1], pts[:, 2])
        return np.asarray(gf(mp)).reshape(len(pts), 3)

    def parity(self, gfs, kx, ky, op: str, pts: np.ndarray) -> list[float]:
        """<u, P u> / <u, u> on sample points, P the mirror op ('z', 'x' or 'y')."""
        c = self.cell.center
        q = pts.copy()
        S = np.ones(3)
        if op == "z":
            q[:, 2] = -q[:, 2]
            S[2] = -1
        elif op == "x":
            q[:, 0] = 2 * c[0] - q[:, 0]
            S[0] = -1
        else:
            q[:, 1] = 2 * c[1] - q[:, 1]
            S[1] = -1
        qf, bloch = self._fold(q, kx, ky)
        out = []
        for gf in gfs:
            u = self._eval(gf, pts)
            um = self._eval(gf, qf) * bloch[:, None] * S[None, :]
            out.append(float(np.real(np.vdot(u, um)) / np.real(np.vdot(u, u))))
        return out

    def mirror_ok(self, op: str, kx: float, ky: float) -> bool:
        """The mirror maps the crystal and this k onto themselves (mod G)."""
        from shapely import affinity

        if op == "z":
            return self.run == "full"
        hole = self.spec.geometry.hole
        if hole.family != "none":
            h = hole_shapely(hole)
            hm = affinity.scale(h, -1 if op == "x" else 1, -1 if op == "y" else 1, origin=(0, 0))
            if h.symmetric_difference(hm).area > 1e-6 * max(h.area, 1.0):
                return False
        a1, a2 = self.cell.a1 * UM, self.cell.a2 * UM  # m
        S = np.diag([-1.0, 1.0]) if op == "x" else np.diag([1.0, -1.0])
        B = np.column_stack([a1, a2])
        if not np.allclose(
            np.linalg.solve(B, S @ B), np.round(np.linalg.solve(B, S @ B)), atol=1e-9
        ):
            return False
        dk = np.array([kx, ky]) - S @ np.array([kx, ky])
        m = (B.T @ dk) / (2 * np.pi)
        return bool(np.allclose(m, np.round(m), atol=1e-6))

    def energy_fraction(self, gfs) -> list[float]:
        """Strain energy in material domains / total. One material domain today, so exactly 1
        unless another domain (air, PML) is added; kept so the field exists in the schema."""
        from ngsolve import Integrate, dx  # noqa: F401

        mats = self.mesh.GetMaterials()
        return [1.0 if len(mats) == 1 else float("nan") for _ in gfs]

    def export_fields(self, gfs, md: Modes, kx, ky, out_dir, stem: str):
        """npz: vertex coordinates (nm) and complex displacement per mode at the vertices (arbitrary
        normalisation, max |u| = 1), Bloch k and frequencies. vtu: Re/Im of each mode (ParaView)."""
        from pathlib import Path

        from ngsolve import VTKOutput

        out = Path(out_dir) / "fields"
        out.mkdir(parents=True, exist_ok=True)
        P = np.array(self.mesh.ngmesh.Coordinates())
        U = np.stack([self._eval(gf, P) for gf in gfs])
        U /= np.abs(U).reshape(len(gfs), -1).max(axis=1)[:, None, None]
        tets = np.array([[v.nr - 1 for v in el.vertices] for el in self.mesh.ngmesh.Elements3D()])
        np.savez_compressed(
            out / f"{stem}.npz",
            xyz_nm=P / NM,
            tets=tets,
            u=U,
            freqs_hz=md.freqs_hz,
            k=np.array([kx, ky]),
            run=self.run,
        )
        coefs, names = [], []
        for j, gf in enumerate(gfs):
            coefs += [gf.real, gf.imag]
            names += [f"re_u{j}", f"im_u{j}"]
        VTKOutput(self.mesh, coefs=coefs, names=names, filename=str(out / stem), subdivision=0).Do()

    def ndof(self) -> int:
        fes = self.space(0.0, 0.0)
        return int(fes.free_mask.sum())


def solve_k(
    cs: CellSolver, kx: float, ky: float, outputs, pts=None, fields_dir=None, index: int = 0
) -> tuple[KResult, Modes]:
    """One k-point of one run, with the requested parities: the unit the cloud lane splits.
    outputs.fields and fields_dir: fields/<run>_k<index>.npz (+ .vtu) with the displacement."""
    if outputs.fields and fields_dir is None:
        raise ValueError("outputs.fields needs an output directory")
    want_vec = outputs.parity or outputs.fields
    md = cs.solve(kx, ky, vectors=want_vec)
    kr = KResult(kx=kx, ky=ky, freqs_hz=md.freqs_hz.tolist(), complete=md.complete)
    if cs.run != "full":
        kr.parity_z = [1.0 if cs.run == "sym" else -1.0] * len(md.freqs_hz)
    if want_vec:
        gfs = cs.gridfunctions(md)
        pts = cs.sample_points() if pts is None else pts
        if cs.run == "full":
            kr.parity_z = cs.parity(gfs, kx, ky, "z", pts)
        for op in ("x", "y"):
            if cs.mirror_ok(op, kx, ky):
                setattr(kr, f"parity_{op}", cs.parity(gfs, kx, ky, op, pts))
        kr.energy_fraction = cs.energy_fraction(gfs)
        if outputs.fields:
            cs.export_fields(gfs, md, kx, ky, fields_dir, f"{cs.run}_k{index:03d}")
    md.vectors = None
    # The UMFPACK factor sits in a reference cycle (the shift-invert closure); without a collect
    # each k leaks it: +0.36 GB per k at 21k dof (P2 t/5), the likely cause of the dmv1 timeouts.
    import gc

    gc.collect()
    return kr, md


_WORKER: dict = {}


def _worker_init(spec_json: str, threads: int, fields_dir=None):
    import os

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    _WORKER["spec"] = ElasticSpec.model_validate_json(spec_json)
    _WORKER["solvers"] = {}
    _WORKER["fields_dir"] = fields_dir


def _worker_task(item):
    run, i, kx, ky = item
    spec = _WORKER["spec"]
    if run not in _WORKER["solvers"]:
        cs = CellSolver(spec, run)
        _WORKER["solvers"][run] = (cs, cs.sample_points() if spec.outputs.parity else None)
    cs, pts = _WORKER["solvers"][run]
    t0 = time.time()
    kr, md = solve_k(cs, kx, ky, spec.outputs, pts, _WORKER["fields_dir"], i)
    return run, i, kr.model_dump(), time.time() - t0, cs.ndof(), _peak_rss_gb()


def _peak_rss_gb() -> float:
    import resource
    import sys

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6  # bytes on macOS, KiB on Linux


def run_spec(
    spec: ElasticSpec, kpoints: list[tuple[float, float]], progress=None, workers=1, fields_dir=None
):
    """Solve every requested symmetry run at every k (workers > 1: k-points across processes,
    each process meshes once per run). Returns (runs, mesh_info)."""
    from .spec import MeshInfo, RunResult

    first = CellSolver(spec, spec.symmetries[0])
    mesh_info = MeshInfo(
        hmax_nm=first.hmax_nm,
        order=first.order,
        refine=spec.mesh.refine,
        n_elements=first.mesh.ne,
        n_vertices=first.mesh.nv,
    )
    items = [(run, i, kx, ky) for run in spec.symmetries for i, (kx, ky) in enumerate(kpoints)]
    got: dict[str, dict[int, KResult]] = {run: {} for run in spec.symmetries}
    ndof, secs = {}, {run: 0.0 for run in spec.symmetries}
    if workers <= 1:
        solvers = {spec.symmetries[0]: first}
        for run, i, kx, ky in items:
            if run not in solvers:
                solvers[run] = CellSolver(spec, run)
            cs = solvers[run]
            tk = time.time()
            kr, md = solve_k(cs, kx, ky, spec.outputs, None, fields_dir, i)
            got[run][i], ndof[run] = kr, cs.ndof()
            secs[run] += time.time() - tk
            if progress:
                progress(run, i, len(kpoints), md)
    else:
        import multiprocessing as mp
        import sys
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # ProcessPoolExecutor, not mp.Pool: a worker killed by the OOM killer raises
        # BrokenProcessPool here; mp.Pool waits for it forever (dmv1-20261016224829, tasks 4/6/7).
        ctx = mp.get_context("spawn")
        init = (spec.model_dump_json(by_alias=True), 1, fields_dir)
        with ProcessPoolExecutor(workers, ctx, _worker_init, init) as pool:
            futs = [pool.submit(_worker_task, it) for it in items]
            for fut in as_completed(futs):
                run, i, krd, dt, nd, rss = fut.result()
                got[run][i], ndof[run] = KResult.model_validate(krd), nd
                secs[run] += dt
                print(
                    f"  {run} k {i + 1}: {dt:.0f} s, ndof {nd}, worker peak RSS {rss:.2f} GB",
                    file=sys.stderr,
                    flush=True,
                )
                if progress:
                    progress(run, i, len(kpoints), got[run][i])
    runs = [
        RunResult(
            symmetry=run,
            k=[got[run][i] for i in range(len(kpoints))],
            ndof=ndof[run],
            seconds=secs[run],
        )
        for run in spec.symmetries
    ]
    return runs, mesh_info


__all__ = ["CellSolver", "run_spec", "solve_k", "NM"]
