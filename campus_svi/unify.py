"""Schema unification and per-cell aggregation.

The two sources have genuinely different native granularity: Mapillary gives
many independently timestamped images per cell, Google gives one canonical
panorama per location per capture period, dated to the month. A raw count
comparison is therefore not apples to apples, and the cross-source metrics
below deliberately rest on the two things that *are* comparable — binary
coverage and temporal depth — while density, contributor diversity and
sequence structure stay source-specific.

Points are re-assigned to grid cells by spatial join here rather than trusting
the ``grid_id`` recorded at fetch time, since Google panos are snapped and may
land in a neighbouring cell.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from campus_svi import config, finalize, grids

AGREEMENT_CLASSES = ["both", "mapillary_only", "google_only", "neither"]


def _assign_cells(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial-join points onto grid cells, replacing any fetch-time grid_id."""
    if points.empty:
        return points
    pts = points.drop(columns=[c for c in ("grid_id",) if c in points.columns])
    joined = gpd.sjoin(
        pts.to_crs(grid.crs), grid[["grid_id", "geometry"]], how="inner", predicate="within"
    )
    return joined.drop(columns=["index_right"], errors="ignore")


def _agg_mapillary(mly: gpd.GeoDataFrame) -> pd.DataFrame:
    if mly.empty:
        return pd.DataFrame(columns=["grid_id"])

    g = mly.groupby("grid_id")
    out = pd.DataFrame({
        "mly_count": g.size(),
        "mly_n_sequences": g["sequence_id"].nunique(),
        "mly_n_creators": g["creator_id"].nunique(),
        "mly_n_orgs": g["organization_id"].nunique(),
    })
    if "year" in mly.columns:
        out["mly_n_years"] = g["year"].nunique()
        out["mly_year_min"] = g["year"].min()
        out["mly_year_max"] = g["year"].max()
    if "year_month" in mly.columns:
        out["mly_n_months"] = g["year_month"].nunique()
    if "is_pano" in mly.columns:
        out["mly_pano_ratio"] = g["is_pano"].mean()
    return out.reset_index()


def _as_bool(s: pd.Series) -> pd.Series:
    """Coerce a boolean column that survived a CSV/GPKG round trip.

    True/False can come back as the strings 'True'/'False', and the column is
    nullable because copyright is absent for some panoramas — so this returns
    a float series (1.0/0.0/NaN) that ``mean()`` handles correctly.
    """
    if s.dtype == bool:
        return s.astype(float)
    m = {"true": 1.0, "1": 1.0, "1.0": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0}
    return s.astype(str).str.strip().str.lower().map(m)


def _agg_google(ggl: gpd.GeoDataFrame) -> pd.DataFrame:
    if ggl.empty:
        return pd.DataFrame(columns=["grid_id"])

    ggl = ggl.copy()
    for col in ("is_third_party", "is_historical"):
        if col in ggl.columns:
            ggl[col] = _as_bool(ggl[col])

    g = ggl.groupby("grid_id")
    out = pd.DataFrame({"ggl_count": g.size()})
    if "year" in ggl.columns:
        out["ggl_n_years"] = g["year"].nunique()
        out["ggl_year_min"] = g["year"].min()
        out["ggl_year_max"] = g["year"].max()
    if "year_month" in ggl.columns:
        out["ggl_n_captures"] = g["year_month"].nunique()
    if "is_third_party" in ggl.columns:
        # Third-party = uploaded by a user rather than captured by Google.
        out["ggl_third_party_ratio"] = g["is_third_party"].mean()
        out["ggl_official_ratio"] = 1.0 - out["ggl_third_party_ratio"]
    if "is_historical" in ggl.columns:
        out["ggl_n_historical"] = g["is_historical"].sum()
    if "capture_source" in ggl.columns:
        # `scout` is trekker/tripod coverage NOT snapped to roads — the share
        # of it is a direct measure of how much the road-snapping caveat bites.
        src = ggl["capture_source"].astype(str)
        ggl["_is_scout"] = (src == "scout").astype(float)
        ggl["_is_launch"] = (src == "launch").astype(float)
        out["ggl_scout_ratio"] = ggl.groupby("grid_id")["_is_scout"].mean()
        out["ggl_launch_ratio"] = ggl.groupby("grid_id")["_is_launch"].mean()
    return out.reset_index()


def classify(row) -> str:
    m, g = bool(row["mly_coverage"]), bool(row["ggl_coverage"])
    if m and g:
        return "both"
    if m:
        return "mapillary_only"
    if g:
        return "google_only"
    return "neither"


def unify_campus(campus_id: str, verbose: bool = True) -> Path:
    """Build the per-cell wide table for one campus.

    Returns the path to a GeoPackage whose ``cells`` layer carries the grid
    geometry joined to every per-cell metric, ready for mapping.
    """
    config.ensure_dirs()
    grid = grids.load_grid(campus_id)

    mly = _assign_cells(finalize.load_final(campus_id, "mapillary"), grid)
    ggl = _assign_cells(finalize.load_final(campus_id, "google"), grid)

    wide = grid.copy()
    for agg in (_agg_mapillary(mly), _agg_google(ggl)):
        if not agg.empty:
            wide = wide.merge(agg, on="grid_id", how="left")

    count_cols = [c for c in wide.columns if c.endswith(("_count", "_n_years",
                  "_n_sequences", "_n_creators", "_n_orgs", "_n_months",
                  "_n_captures"))]
    for c in count_cols:
        wide[c] = wide[c].fillna(0).astype(int)
    for c in ("mly_count", "ggl_count"):
        if c not in wide.columns:
            wide[c] = 0

    wide["mly_coverage"] = (wide["mly_count"] > 0).astype(int)
    wide["ggl_coverage"] = (wide["ggl_count"] > 0).astype(int)
    wide["either_coverage"] = ((wide["mly_coverage"] + wide["ggl_coverage"]) > 0).astype(int)
    wide["agreement"] = wide.apply(classify, axis=1)

    # Temporal depth difference, positive where Mapillary reaches further back
    # in time coverage than Google. Only meaningful where both are present.
    if {"mly_n_years", "ggl_n_years"}.issubset(wide.columns):
        wide["depth_diff"] = wide["mly_n_years"] - wide["ggl_n_years"]
        wide.loc[wide["agreement"] != "both", "depth_diff"] = np.nan

    out = config.PROCESSED_DIR / f"{campus_id}_cells.gpkg"
    wide.to_file(out, layer="cells", driver="GPKG")
    wide.drop(columns="geometry").to_csv(
        config.PROCESSED_DIR / f"{campus_id}_cells.csv", index=False
    )

    if verbose:
        n = len(wide)
        print(f"[unify/{campus_id}] {n} cells")
        print(f"  Mapillary coverage: {wide['mly_coverage'].sum()}/{n} "
              f"({wide['mly_coverage'].mean():.1%})")
        print(f"  Google coverage   : {wide['ggl_coverage'].sum()}/{n} "
              f"({wide['ggl_coverage'].mean():.1%})")
        print(f"  Either            : {wide['either_coverage'].sum()}/{n} "
              f"({wide['either_coverage'].mean():.1%})")
        for k in AGREEMENT_CLASSES:
            print(f"    {k:<15} {(wide['agreement'] == k).sum():>5}")
        print(f"  -> {out.name}")

    return out


def load_cells(campus_id: str) -> gpd.GeoDataFrame:
    path = config.PROCESSED_DIR / f"{campus_id}_cells.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"Run the unify step for '{campus_id}' first.")
    return gpd.read_file(path, layer="cells")


def combine(campus_ids: list[str]) -> pd.DataFrame:
    """Stack per-cell tables from several campuses for cross-campus analysis."""
    frames = []
    for cid in campus_ids:
        try:
            frames.append(load_cells(cid).drop(columns="geometry"))
        except FileNotFoundError:
            print(f"  ! skipping '{cid}' — not unified yet")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(config.PROCESSED_DIR / "all_campuses_cells.csv", index=False)
    return out
