"""Cell data — the second deliverable.

Points are assigned to grid cells by **spatial join**, never by whichever fetch
unit happened to return them. This matters: Google panoramas are road-snapped
and Mapillary boxes overlap, so fetch-time attribution would put points in the
wrong cell.

This module aggregates and nothing more. No coverage ratios, decay curves,
Moran's I or figures — those belong in the analysis layer, which reads the
outputs written here.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from campus_svi import config, grids, points


def cells_path(campus_id: str) -> Path:
    return config.CELL_DIR / f"{campus_id}_cells.gpkg"


def _as_bool(s: pd.Series) -> pd.Series:
    """Coerce a boolean that survived a CSV/GPKG round trip. Returns float so
    NaN survives — some fields are genuinely unknown, not False."""
    if s.dtype == bool:
        return s.astype(float)
    m = {"true": 1.0, "1": 1.0, "1.0": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0}
    return s.astype(str).str.strip().str.lower().map(m)


def _assign(pts: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if pts.empty:
        return pts
    pts = pts.drop(columns=[c for c in ("grid_id",) if c in pts.columns])
    j = gpd.sjoin(pts.to_crs(grid.crs), grid[["grid_id", "geometry"]],
                  how="inner", predicate="within")
    return j.drop(columns=["index_right"], errors="ignore")


def _agg_mapillary(m: gpd.GeoDataFrame) -> pd.DataFrame:
    if m.empty:
        return pd.DataFrame(columns=["grid_id"])
    g = m.groupby("grid_id")
    out = pd.DataFrame({
        "mly_count": g.size(),
        "mly_n_sequences": g["sequence_id"].nunique(),
        "mly_n_creators": g["creator_id"].nunique(),
        "mly_n_orgs": g["organization_id"].nunique(),
    })
    if "year" in m.columns:
        out["mly_n_years"] = g["year"].nunique()
        out["mly_year_min"] = g["year"].min()
        out["mly_year_max"] = g["year"].max()
    if "year_month" in m.columns:
        out["mly_n_months"] = g["year_month"].nunique()
    if "is_pano" in m.columns:
        out["mly_pano_ratio"] = _as_bool(m["is_pano"]).groupby(m["grid_id"]).mean()
    return out.reset_index()


def _agg_google(x: gpd.GeoDataFrame) -> pd.DataFrame:
    if x.empty:
        return pd.DataFrame(columns=["grid_id"])
    x = x.copy()
    for c in ("is_third_party", "is_historical"):
        if c in x.columns:
            x[c] = _as_bool(x[c])

    g = x.groupby("grid_id")
    out = pd.DataFrame({"ggl_count": g.size()})

    if "is_historical" in x.columns:
        # Positions are the requests paid for; historical captures ride along
        # free inside each response, so the two are worth separating.
        out["ggl_n_positions"] = g["is_historical"].apply(lambda s: int((s == 0).sum()))
        out["ggl_n_historical"] = g["is_historical"].sum().astype("Int64")
    if "year" in x.columns:
        out["ggl_n_years"] = g["year"].nunique()
        out["ggl_year_min"] = g["year"].min()
        out["ggl_year_max"] = g["year"].max()
    if "year_month" in x.columns:
        out["ggl_n_captures"] = g["year_month"].nunique()
    if "is_third_party" in x.columns:
        out["ggl_third_party_ratio"] = g["is_third_party"].mean()
    if "capture_source" in x.columns:
        # `scout` is trekker/tripod capture, NOT snapped to roads. Its share
        # measures how much the road-snapping caveat actually bites.
        src = x["capture_source"].astype(str)
        for name in ("launch", "scout", "innerspace"):
            x[f"_{name}"] = (src == name).astype(float)
            out[f"ggl_{name}_ratio"] = x.groupby("grid_id")[f"_{name}"].mean()
    if "elevation" in x.columns:
        out["ggl_mean_elevation"] = g["elevation"].mean()
    return out.reset_index()


# Every cell table carries these columns whether or not the campus has data
# for them. A campus with no Google coverage would otherwise produce a table
# missing the ggl_* columns entirely, and anything consuming several campuses
# would break on the first one that differs.
SCHEMA_INT = ["mly_count", "mly_n_sequences", "mly_n_creators", "mly_n_orgs",
              "mly_n_years", "mly_n_months",
              "ggl_count", "ggl_n_positions", "ggl_n_historical",
              "ggl_n_years", "ggl_n_captures"]
SCHEMA_FLOAT = ["mly_pano_ratio", "mly_year_min", "mly_year_max",
                "ggl_year_min", "ggl_year_max", "ggl_third_party_ratio",
                "ggl_launch_ratio", "ggl_scout_ratio", "ggl_innerspace_ratio",
                "ggl_mean_elevation", "depth_diff"]


def _complete_schema(wide):
    for c in SCHEMA_INT:
        if c not in wide.columns:
            wide[c] = 0
        wide[c] = wide[c].fillna(0).astype(int)
    for c in SCHEMA_FLOAT:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide


def build_cells(campus_id: str, verbose: bool = True) -> Path:
    """Join both sources onto the grid and write ``{campus}_cells.gpkg``."""
    config.ensure_dirs()
    grid = grids.load_grid(campus_id)

    mly = _assign(points.load_points(campus_id, "mapillary"), grid)
    ggl = _assign(points.load_points(campus_id, "google"), grid)

    wide = grid.copy()
    for agg in (_agg_mapillary(mly), _agg_google(ggl)):
        if not agg.empty:
            wide = wide.merge(agg, on="grid_id", how="left")

    for c in wide.columns:
        if c.startswith(("mly_n_", "ggl_n_")) or c in ("mly_count", "ggl_count"):
            wide[c] = wide[c].fillna(0).astype(int)
    for c in ("mly_count", "ggl_count"):
        if c not in wide.columns:
            wide[c] = 0

    wide["mly_coverage"] = (wide["mly_count"] > 0).astype(int)
    wide["ggl_coverage"] = (wide["ggl_count"] > 0).astype(int)
    wide["either_coverage"] = (
        (wide["mly_coverage"] + wide["ggl_coverage"]) > 0).astype(int)

    def agree(r):
        m, g = bool(r["mly_coverage"]), bool(r["ggl_coverage"])
        return "both" if m and g else "mapillary_only" if m else \
               "google_only" if g else "neither"
    wide["agreement"] = wide.apply(agree, axis=1)

    if {"mly_n_years", "ggl_n_years"}.issubset(wide.columns):
        wide["depth_diff"] = wide["mly_n_years"] - wide["ggl_n_years"]
        wide.loc[wide["agreement"] != "both", "depth_diff"] = np.nan

    wide = _complete_schema(wide)
    # Carried so downstream tables and figures never have to re-derive it.
    from campus_svi import registry
    wide["display_name"] = registry.display_name(campus_id)

    out = cells_path(campus_id)
    wide.to_file(out, layer="cells", driver="GPKG")
    wide.drop(columns="geometry").to_csv(
        config.CELL_DIR / f"{campus_id}_cells.csv", index=False)

    if verbose:
        n = len(wide)
        print(f"[cells/{campus_id}] {n} cells")
        print(f"  mapillary {wide['mly_coverage'].sum():>5}/{n} "
              f"({wide['mly_coverage'].mean():.0%})   "
              f"google {wide['ggl_coverage'].sum():>5}/{n} "
              f"({wide['ggl_coverage'].mean():.0%})   "
              f"either {wide['either_coverage'].mean():.0%}")
        print(f"  -> {out.name}")
    return out


def load_cells(campus_id: str) -> gpd.GeoDataFrame:
    p = cells_path(campus_id)
    if not p.exists():
        raise FileNotFoundError(f"Run the cells step for '{campus_id}' first.")
    return gpd.read_file(p, layer="cells")


def combine(campus_ids: list[str], verbose: bool = True) -> pd.DataFrame:
    """Stack per-cell tables across campuses into one table for analysis."""
    frames = []
    for cid in campus_ids:
        try:
            frames.append(load_cells(cid).drop(columns="geometry"))
        except FileNotFoundError:
            if verbose:
                print(f"  ! {cid}: no cell data yet")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    p = config.CELL_DIR / "all_campuses_cells.csv"
    out.to_csv(p, index=False)
    if verbose:
        print(f"[combine] {len(out)} cells across "
              f"{out['campus_id'].nunique()} campuses -> {p.name}")
    return out
