# npc-elastic

Elastic eigenmodes of a 3D phononic-crystal unit cell: the open counterpart of COMSOL Solid
Mechanics → Eigenfrequency. One unit cell (slab minus hole), Bloch-periodic sides, free top face,
COMSOL's symmetric and antisymmetric half-slab runs, a k-path, and the n eigenfrequencies nearest a
shift. NGSolve does the finite elements, SciPy ARPACK the eigensolve.

## Install (one command)

Needs Python 3.12 or newer, on macOS (Intel or Apple Silicon) or Linux x86_64. NGSolve and Netgen
come from pip as wheels with this command. On macOS there is nothing else to install. On Linux, the
OpenBLAS that NGSolve bundles needs one system library, `libquadmath` (without it `import ngsolve`
fails with `libquadmath.so.0: cannot open shared object file`). Most desktops with gcc already have it:

```
sudo apt-get install libquadmath0      # Debian, Ubuntu
sudo dnf install libquadmath           # Fedora, RHEL
```

```
python3 -m venv npc-env && source npc-env/bin/activate
pip install "npc-elastic[plot] @ https://github.com/laghai/npc-elastic/archive/refs/tags/v0.1.0.zip"
```

If `python3 --version` prints below 3.12 (macOS's own python3 is 3.9), use
[uv](https://docs.astral.sh/uv/): `uv venv --python 3.12 npc-env && source npc-env/bin/activate`,
then the same line as `uv pip install ...`.

## Reproduce diamond_mech_v1

From a checkout of [optomechanics_copy](https://github.com/ajmstein124/optomechanics_copy). The
results go into a separate folder, `npc_check/`, so your COMSOL CSVs are not touched.

```
cd optomechanics_copy
curl -LO https://raw.githubusercontent.com/laghai/npc-elastic/v0.1.0/examples/diamond_mech_v1_repro_rho2329.json
curl -LO https://raw.githubusercontent.com/laghai/npc-elastic/v0.1.0/examples/compare_to_comsol.py
python -m npc_elastic diamond_mech_v1_repro_rho2329.json npc_check/runs/diamond_mech_v1/data --workers 2
(cd npc_check && python ../runs/diamond_mech_v1/plot_bands.py)
python compare_to_comsol.py npc_check/runs/diamond_mech_v1/data runs/diamond_mech_v1/data
```

- The figure is `npc_check/runs/diamond_mech_v1/figures/diamond_mech_bands.png`, drawn by your
  `plot_bands.py` from `npc_check/runs/diamond_mech_v1/data/Python_diamond_Gamma_{sym,antisym}.csv`.
- `compare_to_comsol.py` compares every mode with your COMSOL CSVs and prints PASS or FAIL (tolerance
  0.1 %). We measured median 0.01 %, max 0.03 % (table below).
- `--workers N` solves N k-points at once. Each worker needs about 5 GB of memory at this mesh, so
  use about (memory in GB) / 6: 2 on a 16 GB laptop, 4 on 32 GB.

**Parameters.** All from your repo (`CLAUDE.md`, `runs/diamond_mech_v1/sweep.py`): hex lattice
a = 300 nm, t = 140 nm, 3-arm snowflake d = 104 nm, w = 107 nm, fillet 16 nm; isotropic E = 1050 GPa,
ν = 0.07; Γ→M→K→Γ with 9 steps per segment (27 k-points); 20 modes nearest 10 GHz; sym and antisym
runs; tetrahedra at t/7.5, cubic elements (P3).

**Density.** Your CSVs match ρ = 2329 kg/m³, not the 3500 your scripts set: only 2329 reproduces them,
and 3500 gives every mode at 0.816× yours (√(2329/3500)). This is the COMSOL trap your `CLAUDE.md`
describes: setting `rho` in the `def` group leaves the Output Property at silicon's 2329. So the
reproduction file uses 2329, and `diamond_mech_v1.json` is the same cell at the design value 3500.
Run that one for diamond:

```
curl -LO https://raw.githubusercontent.com/laghai/npc-elastic/v0.1.0/examples/diamond_mech_v1.json
python -m npc_elastic diamond_mech_v1.json npc_diamond/runs/diamond_mech_v1/data --workers 2
```

**A quick look first.** `diamond_mech_v1_repro_rho2329_p2.json` is the same run with quadratic
elements (P2): about 4× faster and 5× less memory per worker, and within 0.2 % of the P3 answer.

**Your own cell.** Copy a spec file and edit it: the fields are in
[`schema/elastic-spec.v0.schema.json`](schema/elastic-spec.v0.schema.json). Lengths are nm,
frequencies Hz, k rad/m, stiffness Pa, density kg/m³. Hole families: `snowflake` (d, w, fillet),
`circle` (r), `cross` (h, w, fillet), `polygon` (vertices, fillet), `none`. Materials: `isotropic`
(E, nu, rho), `cubic` (C11, C12, C44, rho, orientation), `anisotropic` (6×6 Voigt C). From Python:

```python
from npc_elastic import run
res = run("diamond_mech_v1.json", out_dir="out/", workers=2)   # out/result.json + the two CSVs
```

`out/result.json` also carries each mode's z-parity, the mesh, the engine versions and whether the
mode set is provably the n nearest the shift. The same spec runs unchanged on our cloud with
`npc submit` / `npc sweep` when a sweep is too big for the laptop.
