"""Analysis grid and Mapillary seed boxes.

The **grid** is the analysis unit: a regular square lattice in the local UTM
CRS, filtered to cells that meaningfully overlap the campus. Points are
assigned to cells by spatial join at the end, never at fetch time.

**Seed boxes** are a fetch unit, not an analysis unit. A whole-campus bbox is
refused outright by the Mapillary API, so the AOI is cut into coarse boxes that
each become an atomic, resumable fetch. They deliberately do not align with the
grid — the quadtree inside each seed subdivides on its own terms.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from campus_svi import boundaries, config


def _tile_polygons(poly, size_m: float, crs_m):
    """Square tiles of ``size_m`` covering a polygon's bounds, snapped to a
    multiple of the size so runs are reproducible."""
    minx, miny, maxx, maxy = poly.bounds
    x0 = np.floor(minx / size_m) * size_m
    y0 = np.floor(miny / size_m) * size_m
    out, rows, cols = [], [], []
    for j, y in enumerate(np.arange(y0, maxy, size_m)):
        for i, x in enumerate(np.arange(x0, maxx, size_m)):
            out.append(box(x, y, x + size_m, y + size_m))
            rows.append(j)
            cols.append(i)
    return gpd.GeoDataFrame({"row": rows, "col": cols}, geometry=out, crs=crs_m)


# --------------------------------------------------------------------------
# Analysis grid
# --------------------------------------------------------------------------

def build_grid(campus_id: str, cell_size_m: float | None = None,
               min_overlap: float | None = None) -> gpd.GeoDataFrame:
    cell_size_m = cell_size_m or config.CELL_SIZE_M
    min_overlap = config.MIN_CELL_OVERLAP if min_overlap is None else min_overlap

    bnd = boundaries.load(campus_id)
    crs_m = boundaries.utm_crs(bnd)
    poly = bnd.to_crs(crs_m).geometry.iloc[0]

    grid = _tile_polygons(poly, cell_size_m, crs_m)
    grid = grid[grid.intersects(poly)].copy()
    if grid.empty:
        raise ValueError(f"Grid for '{campus_id}' is empty — check the boundary CRS.")

    grid["area_m2"] = grid.geometry.area
    grid["area_inside_m2"] = grid.geometry.intersection(poly).area
    grid["frac_inside"] = grid["area_inside_m2"] / grid["area_m2"]
    grid = grid[grid["frac_inside"] >= min_overlap].copy()
    grid = grid.sort_values(["row", "col"]).reset_index(drop=True)

    grid["campus_id"] = campus_id
    grid["grid_id"] = [f"{campus_id}_{r:04d}_{c:04d}"
                       for r, c in zip(grid["row"], grid["col"])]
    grid["cell_size_m"] = cell_size_m

    out = grid.to_crs("EPSG:4326")
    b = out.geometry.bounds
    out["bbox_west"], out["bbox_south"] = b["minx"].values, b["miny"].values
    out["bbox_east"], out["bbox_north"] = b["maxx"].values, b["maxy"].values
    cent = grid.geometry.centroid.to_crs("EPSG:4326")
    out["cx"], out["cy"] = cent.x.values, cent.y.values

    return out[["grid_id", "campus_id", "row", "col", "cell_size_m",
                "area_m2", "area_inside_m2", "frac_inside",
                "bbox_west", "bbox_south", "bbox_east", "bbox_north",
                "cx", "cy", "geometry"]]


def grid_path(campus_id: str) -> Path:
    return config.GRID_DIR / f"{campus_id}_grid.gpkg"


def save_grid(grid: gpd.GeoDataFrame, campus_id: str) -> Path:
    config.ensure_dirs()
    p = grid_path(campus_id)
    grid.to_file(p, layer="grid", driver="GPKG")
    return p


def load_grid(campus_id: str) -> gpd.GeoDataFrame:
    p = grid_path(campus_id)
    if not p.exists():
        raise FileNotFoundError(f"No grid for '{campus_id}'. Run the grid step first.")
    return gpd.read_file(p, layer="grid")


# --------------------------------------------------------------------------
# Mapillary seed boxes
# --------------------------------------------------------------------------

def seed_boxes(campus_id: str, size_m: float | None = None) -> gpd.GeoDataFrame:
    """Coarse boxes covering the campus, each an atomic Mapillary fetch unit."""
    size_m = size_m or config.MLY_SEED_SIZE_M

    bnd = boundaries.load(campus_id)
    crs_m = boundaries.utm_crs(bnd)
    poly = bnd.to_crs(crs_m).geometry.iloc[0]

    seeds = _tile_polygons(poly, size_m, crs_m)
    seeds = seeds[seeds.intersects(poly)].copy()
    seeds = seeds.sort_values(["row", "col"]).reset_index(drop=True)
    seeds["seed_id"] = [f"{campus_id}_s{r:04d}_{c:04d}"
                        for r, c in zip(seeds["row"], seeds["col"])]
    seeds["campus_id"] = campus_id

    out = seeds.to_crs("EPSG:4326")
    b = out.geometry.bounds
    out["west"], out["south"] = b["minx"].values, b["miny"].values
    out["east"], out["north"] = b["maxx"].values, b["maxy"].values
    return out[["seed_id", "campus_id", "west", "south", "east", "north", "geometry"]]
