"""The run spec and result schema, v0 (frozen 2026-10-16; later changes are versioned, README).

Units: lengths in nm, frequencies in Hz, wavevectors in rad/m (as in Abby's CSVs), stiffness in Pa,
density in kg/m^3. One spec describes one unit cell; `symmetries` lists the runs made on it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION = "npc.elastic.spec/0"
RESULT_VERSION = "npc.elastic.result/0"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- geometry -------------------------------------------------------------------------------


class Lattice(_M):
    """hex: a1=(a,0), a2=(-a/2, a*sqrt(3)/2), her unit_cell_3.mph rhombus. square: a2=(0,a)."""

    type: Literal["hex", "square"]
    a: float = Field(gt=0, description="lattice constant, nm")


class Snowflake(_M):
    """Three arms of length d (from the hole centre) and width w at 90/210/330 deg, every corner
    (convex and concave) rounded with radius fillet. Her scripts/schematic.py hole_poly."""

    family: Literal["snowflake"]
    d: float = Field(gt=0, description="arm length from the hole centre, nm")
    w: float = Field(gt=0, description="arm width, nm")
    fillet: float = Field(0.0, ge=0, description="fillet radius f_r, nm")


class Circle(_M):
    family: Literal["circle"]
    r: float = Field(gt=0, description="radius, nm")


class Cross(_M):
    """Two orthogonal bars of length h and width w, centred (Safavi-Naeini & Painter 2010)."""

    family: Literal["cross"]
    h: float = Field(gt=0, description="bar length tip to tip, nm")
    w: float = Field(gt=0, description="bar width, nm")
    fillet: float = Field(0.0, ge=0, description="fillet radius, nm")


class PolygonHole(_M):
    family: Literal["polygon"]
    vertices: list[tuple[float, float]] = Field(
        min_length=3, description="hole outline relative to the hole centre, nm, CCW"
    )
    fillet: float = Field(0.0, ge=0, description="fillet radius applied to every corner, nm")


class NoHole(_M):
    family: Literal["none"]


Hole = Annotated[Snowflake | Circle | Cross | PolygonHole | NoHole, Field(discriminator="family")]


class Geometry(_M):
    lattice: Lattice
    t: float = Field(gt=0, description="slab thickness, nm")
    hole: Hole
    hole_center: tuple[float, float] | None = Field(
        None,
        description="hole centre from the cell origin corner, nm. Default: the cell centroid. "
        "Physics does not depend on it; the hole must lie inside the cell.",
    )


# --- material -------------------------------------------------------------------------------


class Isotropic(_M):
    model: Literal["isotropic"]
    E: float = Field(gt=0, description="Young's modulus, Pa")
    nu: float = Field(gt=-1, lt=0.5, description="Poisson's ratio")
    rho: float = Field(gt=0, description="density, kg/m^3")


class Cubic(_M):
    """Cubic crystal, e.g. diamond C11/C12/C44 = 1075/149/567 GPa."""

    model: Literal["cubic"]
    C11: float = Field(gt=0, description="Pa")
    C12: float = Field(description="Pa")
    C44: float = Field(gt=0, description="Pa")
    rho: float = Field(gt=0, description="kg/m^3")
    orientation: list[list[float]] | None = Field(
        None, description="3x3 rotation R, crystal axes -> lab axes (columns = crystal axes in lab)"
    )


class Anisotropic(_M):
    model: Literal["anisotropic"]
    C: list[list[float]] = Field(
        description="6x6 Voigt stiffness, Pa, order xx yy zz yz xz xy (engineering shear)"
    )
    rho: float = Field(gt=0, description="kg/m^3")
    orientation: list[list[float]] | None = Field(None, description="3x3 rotation, as in Cubic")


Material = Annotated[Isotropic | Cubic | Anisotropic, Field(discriminator="model")]


# --- k-path, mesh, solver -------------------------------------------------------------------


class KPath(_M):
    """Named path through high-symmetry points (hex: G, M, K; square: G, X, M), n_per_segment steps
    each, repeated points dropped (her sweep.py: G-M-K-G, 9 steps -> 27 points), or a list."""

    points: list[str] | None = Field(None, examples=[["G", "M", "K", "G"]])
    n_per_segment: int = Field(9, ge=1)
    k: list[tuple[float, float]] | None = Field(None, description="explicit (kx, ky), rad/m")
    dedupe: bool = True

    @model_validator(mode="after")
    def _one(self):
        if (self.points is None) == (self.k is None):
            raise ValueError("give exactly one of points or k")
        return self


class Mesh(_M):
    hmax: float | None = Field(None, gt=0, description="max element size, nm")
    hmax_over_t: float | None = Field(
        7.5, gt=0, description="hmax = t / this (her COMSOL tets: t/7.5); used if hmax unset"
    )
    order: int = Field(2, ge=1, le=5, description="Lagrange element order (COMSOL default: 2)")
    refine: int = Field(
        0, ge=0, le=3, description="uniform refinements after meshing (each splits a tet into 8)"
    )


class Solver(_M):
    n_modes: int = Field(20, ge=1)
    shift_hz: float = Field(
        10e9, ge=0, description="return the n_modes eigenfrequencies nearest this"
    )


SymmetryRun = Literal["sym", "antisym", "full"]


class Outputs(_M):
    parity: bool = True
    fields: bool = Field(False, description="write displacement fields (npz) per k and mode")
    csv_prefix: str = Field("Python_diamond_Gamma", description="her CSV name: <prefix>_<sym>.csv")


class ElasticSpec(_M):
    """One unit cell, one material, a k-path, and the runs to make (sym/antisym on the half slab
    with a roller/antiroller on z=0, as her COMSOL does, or the full thickness)."""

    schema_version: Literal["npc.elastic.spec/0"] = Field(SPEC_VERSION, alias="schema")
    geometry: Geometry
    material: Material
    symmetries: list[SymmetryRun] = Field(["sym", "antisym"], min_length=1)
    kpath: KPath
    mesh: Mesh = Mesh()
    solver: Solver = Solver()
    outputs: Outputs = Outputs()

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --- result ---------------------------------------------------------------------------------


class KResult(_M):
    kx: float
    ky: float
    freqs_hz: list[float]
    parity_z: list[float] | None = Field(
        None, description="<u, sigma_z u>/<u,u> per mode, +1 symmetric (in-plane), -1 antisymmetric"
    )
    parity_x: list[float] | None = Field(None, description="mirror x->-x through the hole centre")
    parity_y: list[float] | None = Field(None, description="mirror y->-y through the hole centre")
    energy_fraction: list[float] | None = Field(
        None, description="strain energy in material / total, per mode"
    )
    complete: bool = Field(
        True, description="the returned set is provably the n_modes nearest the shift"
    )


class RunResult(_M):
    symmetry: SymmetryRun
    k: list[KResult]
    ndof: int
    seconds: float


class MeshInfo(_M):
    hmax_nm: float
    order: int
    refine: int = 0
    n_elements: int
    n_vertices: int


class ElasticResult(_M):
    schema_version: Literal["npc.elastic.result/0"] = Field(RESULT_VERSION, alias="schema")
    spec: ElasticSpec
    runs: list[RunResult]
    mesh: MeshInfo
    engine: dict[str, str] = Field(description="package versions: npc_elastic, ngsolve, scipy")
    status: Literal["screening", "qualified"] = Field(
        "screening",
        description="qualified only with a mesh-convergence plateau (docs/reporting.md)",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
