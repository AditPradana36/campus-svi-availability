"""Per-campus finalisation: deduplicate, then reclip to the true boundary.

This is the gate between raw and delivery-ready. Nothing downstream should
read from ``data/raw/``.

Why both steps are necessary
----------------------------
*Duplicates* arise structurally, not accidentally. On the Google side,
coverage tiles overlap the campus edge and a panorama's historical list can
name a pano already returned by another tile. On the Mapillary side, an image
sitting exactly on a bbox seam can be returned by both neighbouring cell
queries.

*Reclipping* is needed because neither fetch unit respects the campus outline.
Google coverage tiles are ~304 m squares that overrun the boundary wholesale,
and a Mapillary bbox query on an edge cell legitimately returns images from
the street outside. Clipping the resulting points against the original
boundary polygon — not the grid or tile extent — is what actually enforces the
study area.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from campus_svi import boundaries, config, google, mapillary


def final_path(campus_id: str) -> Path:
    return config.PROCESSED_DIR / f"{campus_id}_svi_final.gpkg"


def _to_points(df: pd.DataFrame, lon_col: str, lat_col: str) -> gpd.GeoDataFrame:
    df = df.dropna(subset=[lon_col, lat_col]).copy()
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )


def _clip(points: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only points falling inside the campus boundary polygon."""
    if points.empty:
        return points
    poly = boundaries.dissolve(boundary)
    return points[points.geometry.within(poly)].copy()


def finalize_campus(campus_id: str, verbose: bool = True) -> Path:
    """Dedup + reclip both sources; write one GeoPackage per campus.

    Output layers:
        ``mapillary``  one row per unique image inside the boundary
        ``google``     one row per unique panorama inside the boundary
        ``coverage``   per-cell Google query outcomes, coverage flag included
    """
    config.ensure_dirs()
    bnd = boundaries.load(campus_id)
    out = final_path(campus_id)
    if out.exists():
        out.unlink()          # rewrite cleanly rather than appending layers

    report: dict[str, dict] = {}

    # -- Mapillary ---------------------------------------------------------
    mly_csv = mapillary.raw_path(campus_id)
    if mly_csv.exists():
        mly = pd.read_csv(mly_csv, low_memory=False)
        n_raw = len(mly)
        mly = mly.drop_duplicates(subset=["image_id"], keep="first")
        n_dedup = len(mly)

        pts = _to_points(mly, "lon", "lat")
        pts = _clip(pts, bnd)

        if not pts.empty:
            if "captured_at" in pts.columns:
                ts = pd.to_datetime(pts["captured_at"], unit="ms", errors="coerce")
                pts["captured_date"] = ts.dt.strftime("%Y-%m-%d")
                pts["year"] = ts.dt.year
                pts["year_month"] = ts.dt.strftime("%Y-%m")
            pts.to_file(out, layer="mapillary", driver="GPKG")

        report["mapillary"] = {"raw": n_raw, "deduped": n_dedup, "final": len(pts)}
    else:
        report["mapillary"] = {"raw": 0, "deduped": 0, "final": 0}

    # -- Google ------------------------------------------------------------
    ggl_csv = google.raw_path(campus_id)
    if ggl_csv.exists():
        ggl = pd.read_csv(ggl_csv, low_memory=False)
        n_raw = len(ggl)

        found = ggl.dropna(subset=["pano_id"])
        found = found.drop_duplicates(subset=["pano_id"], keep="first")
        n_dedup = len(found)

        pts = _to_points(found, "lon", "lat")
        pts = _clip(pts, bnd)

        if not pts.empty and "date" in pts.columns:
            pts["year"] = pd.to_numeric(
                pts["date"].astype(str).str.slice(0, 4), errors="coerce"
            )
            pts["year_month"] = pts["date"].astype(str).str.slice(0, 7)

        if not pts.empty:
            pts.to_file(out, layer="google", driver="GPKG")

        # Per-tile fetch outcome, retained so "fetched but empty" stays
        # distinguishable from "never fetched".
        if "tile_id" in ggl.columns:
            cov = (
                ggl.groupby("tile_id")
                .agg(n_panos=("pano_id", "count"))
                .reset_index()
            )
            cov["campus_id"] = campus_id
            cov.to_csv(
                config.PROCESSED_DIR / f"{campus_id}_google_tilecoverage.csv",
                index=False,
            )

        report["google"] = {"raw": n_raw, "deduped": n_dedup, "final": len(pts)}
    else:
        report["google"] = {"raw": 0, "deduped": 0, "final": 0}

    if verbose:
        print(f"[finalize/{campus_id}]")
        for src, r in report.items():
            dropped_dup = r["raw"] - r["deduped"]
            dropped_clip = r["deduped"] - r["final"]
            print(f"  {src:<10} raw={r['raw']:>7}  "
                  f"-{dropped_dup} duplicates  -{dropped_clip} outside boundary  "
                  f"final={r['final']:>7}")
        print(f"  -> {out.name}")

    return out


def load_final(campus_id: str, layer: str) -> gpd.GeoDataFrame:
    """Read one layer of a finalised campus file, empty frame if absent."""
    path = final_path(campus_id)
    if not path.exists():
        raise FileNotFoundError(f"Run the finalize step for '{campus_id}' first.")
    try:
        return gpd.read_file(path, layer=layer)
    except Exception:                                    # noqa: BLE001
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
