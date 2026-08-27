"""Slippy-map tile helpers for the Street View coverage sweep.

streetlevel's coverage endpoint works on zoom-17 XYZ tiles, so the fetch unit
is a tile rather than a grid cell. At Indonesian latitudes a z17 tile is about
304 m square, which means a campus of a few square kilometres is covered by
tens of tiles rather than hundreds of per-cell queries.

Decoupling the two resolutions is the point: tiles are the *fetch* unit, grid
cells are the *analysis* unit. Change the cell size and you re-run the
spatial join, not the download.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box
from streetlevel.geo import tile_coord_to_wgs84, wgs84_to_tile_coord

from campus_svi import boundaries

ZOOM = 17


def tile_bounds(x: int, y: int, zoom: int = ZOOM) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a tile in WGS84 degrees."""
    north, west = tile_coord_to_wgs84(x, y, zoom)
    south, east = tile_coord_to_wgs84(x + 1, y + 1, zoom)
    return west, south, east, north


def tiles_for_boundary(
    campus_id: str,
    zoom: int = ZOOM,
    pad: int = 1,
    boundary_dir=None,
) -> gpd.GeoDataFrame:
    """Every zoom-17 tile intersecting a campus boundary.

    ``pad`` extends the tile range outward by that many tiles on each side.
    One ring of padding is the default because panoramas just outside the
    boundary still matter: they are what step 4's reclip trims away, and
    their absence would otherwise look like a boundary effect rather than a
    real coverage edge.
    """
    bnd = boundaries.load(campus_id, boundary_dir)
    poly = boundaries.dissolve(bnd)
    west, south, east, north = poly.bounds

    x0, y0 = wgs84_to_tile_coord(north, west, zoom)   # NW corner
    x1, y1 = wgs84_to_tile_coord(south, east, zoom)   # SE corner
    x0, x1 = min(x0, x1) - pad, max(x0, x1) + pad
    y0, y1 = min(y0, y1) - pad, max(y0, y1) + pad

    rows, geoms = [], []
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            w, s, e, n = tile_bounds(tx, ty, zoom)
            geoms.append(box(w, s, e, n))
            rows.append({
                "tile_id": f"{zoom}_{tx}_{ty}",
                "campus_id": campus_id,
                "tile_x": tx, "tile_y": ty, "zoom": zoom,
                "west": w, "south": s, "east": e, "north": n,
            })

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    # Padding is applied to the tile *range*, so trim tiles that miss the
    # campus entirely — a padded corner tile can sit well outside it.
    keep = gdf.intersects(poly.buffer(0.0035))          # roughly one tile of slack
    return gdf[keep].reset_index(drop=True)
