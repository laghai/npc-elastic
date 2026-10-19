"""Unit-cell geometry from parameters: an OCC prism (cell minus hole) with periodic side faces.

The model is built in micrometres (OCC's default tolerance, 1e-7, is too coarse for SI metres).
Lattice and hole definitions follow Abby's unit_cell_3.mph as read by her scripts/schematic.py:
a1 = (a, 0), a2 = (-a/2, a*sqrt(3)/2); snowflake = union of three w x d rectangles from the hole
centre at 90/210/330 deg, every corner rounded with f_r.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import Geometry

UM = 1e-6  # model length unit, m
NM = 1e-3  # one nm in model units


def lattice_vectors(lattice) -> tuple[np.ndarray, np.ndarray]:
    """Primitive vectors in nm."""
    a = lattice.a
    if lattice.type == "hex":
        return np.array([a, 0.0]), np.array([-a / 2, a * np.sqrt(3) / 2])
    return np.array([a, 0.0]), np.array([0.0, a])


def cell_polygon(lattice) -> np.ndarray:
    a1, a2 = lattice_vectors(lattice)
    return np.array([[0.0, 0.0], a1, a1 + a2, a2])


def hole_center(geom: Geometry) -> np.ndarray:
    if geom.hole_center is not None:
        return np.asarray(geom.hole_center, float)
    a1, a2 = lattice_vectors(geom.lattice)
    return (a1 + a2) / 2


def sharp_outline(hole) -> np.ndarray | None:
    """Hole outline before filleting, nm, about the hole centre, CCW. None for circle/none."""
    from shapely import affinity
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if hole.family == "snowflake":
        base = Polygon(
            [(-hole.w / 2, 0), (hole.w / 2, 0), (hole.w / 2, hole.d), (-hole.w / 2, hole.d)]
        )
        u = unary_union([affinity.rotate(base, ang, origin=(0, 0)) for ang in (0, 120, -120)])
    elif hole.family == "cross":
        h, w = hole.h, hole.w
        u = unary_union(
            [
                Polygon([(-h / 2, -w / 2), (h / 2, -w / 2), (h / 2, w / 2), (-h / 2, w / 2)]),
                Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]),
            ]
        )
    elif hole.family == "polygon":
        u = Polygon(hole.vertices)
    else:
        return None
    u = u.simplify(1e-9 * max(1.0, u.length))
    xy = np.asarray(u.exterior.coords)[:-1]
    if Polygon(xy).exterior.is_ccw is False:
        xy = xy[::-1]
    return xy


def hole_shapely(hole):
    """Exact filleted hole as a shapely polygon (nm), her hole_poly construction. For checks."""
    from shapely.geometry import Point, Polygon

    if hole.family == "circle":
        return Point(0, 0).buffer(hole.r, 512)
    if hole.family == "none":
        return Polygon()
    p = Polygon(sharp_outline(hole))
    r = hole.fillet
    if r > 0:
        p = p.buffer(-r, 256).buffer(r, 256).buffer(r, 256).buffer(-r, 256)
    return p


def _turning(xy: np.ndarray) -> np.ndarray:
    n = len(xy)
    th = np.empty(n)
    for i in range(n):
        e_in = xy[i] - xy[i - 1]
        e_out = xy[(i + 1) % n] - xy[i]
        th[i] = np.arctan2(e_in[0] * e_out[1] - e_in[1] * e_out[0], np.dot(e_in, e_out))
    return th


def _filleted_face(xy_nm: np.ndarray, r_nm: float, center_nm: np.ndarray, z0: float):
    """Turtle-draw a polygon with every corner replaced by a tangent arc of radius r."""
    from netgen.occ import Axes, WorkPlane, X, Z

    xy = (xy_nm + center_nm) * NM
    r = r_nm * NM
    wp = WorkPlane(Axes((0, 0, z0), n=Z, h=X))
    n = len(xy)
    if r <= 0:
        wp.MoveTo(*xy[0])
        for p in xy[1:]:
            wp.LineTo(*p)
        return wp.Close().Face()
    th = _turning(xy)
    tl = r * np.tan(np.abs(th) / 2)
    edges = [xy[(i + 1) % n] - xy[i] for i in range(n)]
    lens = [np.linalg.norm(e) for e in edges]
    for i in range(n):
        if lens[i] - tl[i] - tl[(i + 1) % n] < -1e-12:
            raise ValueError(f"fillet {r_nm} nm too large for edge {i} ({lens[i] / NM:.2f} nm)")
    d0 = edges[0] / lens[0]
    wp.MoveTo(*(xy[0] + tl[0] * d0)).Direction(*d0)
    for i in range(n):
        j = (i + 1) % n
        seg = lens[i] - tl[i] - tl[j]
        if seg > 1e-12:
            wp.Line(seg)
        if abs(th[j]) > 1e-12:
            wp.Arc(r, float(np.degrees(th[j])))
    return wp.Face()


@dataclass
class Cell:
    shape: object  # netgen.occ solid
    z_bottom: float  # model units
    z_top: float
    a1: np.ndarray  # model units
    a2: np.ndarray
    center: np.ndarray  # hole centre, model units
    half: bool


def build_cell(geom: Geometry, half: bool) -> Cell:
    """half=True: z in [0, t/2] with the mid-plane face named 'mid'. Else z in [-t/2, t/2]."""
    from netgen.occ import Axes, Glue, IdentificationType, Pnt, WorkPlane, X, Z, gp_Trsf, gp_Vec  # noqa: F401

    t = geom.t * NM
    z0, z1 = (0.0, t / 2) if half else (-t / 2, t / 2)
    cell_xy = cell_polygon(geom.lattice) * NM
    wp = WorkPlane(Axes((0, 0, z0), n=Z, h=X)).MoveTo(*cell_xy[0])
    for p in cell_xy[1:]:
        wp.LineTo(*p)
    face = wp.Close().Face()

    c = hole_center(geom)
    hole = geom.hole
    if hole.family != "none":
        cell_sh = _shapely_cell(geom)
        hs = _translate(hole_shapely(hole), c)
        if not cell_sh.buffer(-1e-6).contains(hs):
            raise ValueError(
                "the hole crosses the cell boundary; move it with geometry.hole_center "
                f"(clearance {cell_sh.exterior.distance(hs):.2f} nm)"
            )
        if hole.family == "circle":
            hf = (
                WorkPlane(Axes((0, 0, z0), n=Z, h=X))
                .Circle(c[0] * NM, c[1] * NM, hole.r * NM)
                .Face()
            )
        else:
            hf = _filleted_face(sharp_outline(hole), hole.fillet, c, z0)
        face = face - hf

    solid = face.Extrude(z1 - z0)
    solid.faces.Min(Z).name = "mid" if half else "bottom"
    solid.faces.Max(Z).name = "top"

    a1 = cell_xy[1] - cell_xy[0]
    a2 = cell_xy[3] - cell_xy[0]
    zm = (z0 + z1) / 2
    for vec, (p, q) in ((a1, (cell_xy[0], cell_xy[3])), (a2, (cell_xy[0], cell_xy[1]))):
        m = (p + q) / 2
        f_src = solid.faces.Nearest(Pnt(m[0], m[1], zm))
        f_dst = solid.faces.Nearest(Pnt(m[0] + vec[0], m[1] + vec[1], zm))
        f_src.Identify(
            f_dst,
            "per_a1" if vec is a1 else "per_a2",
            IdentificationType.PERIODIC,
            gp_Trsf.Translation(gp_Vec(vec[0], vec[1], 0)),
        )
    return Cell(solid, z0, z1, a1, a2, c * NM, half)


def _shapely_cell(geom: Geometry):
    from shapely.geometry import Polygon

    return Polygon(cell_polygon(geom.lattice))


def _translate(p, c):
    from shapely import affinity

    return affinity.translate(p, c[0], c[1])


def material_area_nm2(geom: Geometry) -> float:
    """Exact in-plane material area (shapely), for checking the OCC model."""
    return _shapely_cell(geom).area - hole_shapely(geom.hole).area


def mesh_cell(cell: Cell, hmax_nm: float, order: int, refine: int = 0):
    from netgen.occ import OCCGeometry
    from ngsolve import Mesh

    ngmesh = OCCGeometry(cell.shape).GenerateMesh(maxh=hmax_nm * NM)
    for _ in range(refine):
        ngmesh.Refine()  # new boundary points are projected onto the OCC surfaces
    mesh = Mesh(ngmesh)
    if order > 1:
        mesh.Curve(order)
    return mesh
