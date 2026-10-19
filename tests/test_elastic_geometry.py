"""OCC unit cell vs exact areas, and the refusal of a hole that crosses the cell."""

import numpy as np
import pytest

pytest.importorskip("netgen")

from npc_elastic.geometry import NM, _turning, build_cell, cell_polygon, sharp_outline  # noqa: E402
from npc_elastic.spec import Geometry  # noqa: E402


def geom(hole, a=300, center=None):
    return Geometry.model_validate(
        dict(lattice=dict(type="hex", a=a), t=140, hole=hole, hole_center=center)
    )


def occ_hole_area(g):
    from shapely.geometry import Polygon

    cell = build_cell(g, half=True)
    return Polygon(cell_polygon(g.lattice)).area - cell.shape.mass / NM**3 / (g.t / 2)


@pytest.mark.parametrize(
    "hole",
    [
        dict(family="snowflake", d=104, w=107, fillet=16),  # her diamond_mech_v1
        dict(family="snowflake", d=124, w=85, fillet=16),
        dict(family="cross", h=200, w=60, fillet=10),
    ],
)
def test_filleted_area_is_exact(hole):
    """Every corner of turning angle th loses or gains r^2 (tan(|th|/2) - |th|/2)."""
    from shapely.geometry import Polygon

    g = geom(hole)
    xy = sharp_outline(g.hole)
    th = _turning(xy)
    r = g.hole.fillet
    corner = r**2 * (np.tan(np.abs(th) / 2) - np.abs(th) / 2)
    exact = Polygon(xy).area - corner[th > 0].sum() + corner[th < 0].sum()
    assert abs(occ_hole_area(g) - exact) < 1e-6 * exact


def test_her_snowflake_outline():
    """Three arms at 90/210/330 deg: 9 corners, 6 convex (+90) and 3 concave (-60)."""
    th = np.degrees(_turning(sharp_outline(geom(dict(family="snowflake", d=104, w=107)).hole)))
    assert sorted(np.round(th).astype(int).tolist()) == [-60] * 3 + [90] * 6


def test_circle_area():
    g = geom(dict(family="circle", r=80))
    assert abs(occ_hole_area(g) - np.pi * 80**2) < 1e-6 * np.pi * 80**2


def test_hole_across_the_boundary_is_refused():
    with pytest.raises(ValueError, match="crosses the cell boundary"):
        build_cell(geom(dict(family="circle", r=80), center=(10, 10)), half=True)
