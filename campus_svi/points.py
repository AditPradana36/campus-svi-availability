"""Point data — the first deliverable.

Deduplicates each source, clips to the true campus boundary, and writes one
GeoPackage per campus with a layer per source.

Both steps are structural, not error handling:

*Duplicates.* On the Mapillary side an image on a seed or quadtree seam is
returned by both neighbouring boxes. On the Google side a coverage tile
overlaps its neighbours, and a panorama's historical list can name a pano
another tile already returned.

*Reclipping.* No fetch unit respects the campus outline. Google tiles are
~304 m squares that overrun it wholesale; Mapillary seed boxes are cut from the
bounding box, not the polygon. Clipping the resulting points against the
original boundary is what actually enforces the study area.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from campus_svi import boundaries, config, google, mapillary


def points_path(campus_id: str) -> Path:
    return config.POINT_DIR / f"{campus_id}_points.gpkg"


def _to_points(df: pd.DataFrame, lon="lon", lat="lat") -> gpd.GeoDataFrame:
    df = df.dropna(subset=[lon, lat]).copy()
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")


def build_points(campus_id: str, verbose: bool = True) -> Path:
    """Dedup + reclip both sources into ``{campus}_points.gpkg``."""
    config.ensure_dirs()
    bnd = boundaries.load(campus_id)
    poly = boundaries.dissolve(bnd)

    out = points_path(campus_id)
    if out.exists():
        out.unlink()                      # rewrite cleanly, don't stack layers

    report = {}

    # -- Mapillary ---------------------------------------------------------
    mly_csv = mapillary.raw_path(campus_id)
    if mly_csv.exists():
        df = pd.read_csv(mly_csv, low_memory=False)
        n_raw = len(df)
        df = df.drop_duplicates(subset=["image_id"], keep="first")
        n_dedup = len(df)

        pts = _to_points(df)
        pts = pts[pts.geometry.within(poly)].copy() if not pts.empty else pts

        if not pts.empty:
            ts = pd.to_datetime(pts["captured_at"], unit="ms", errors="coerce")
            pts["captured_date"] = ts.dt.strftime("%Y-%m-%d")
            pts["year"] = ts.dt.year
            pts["year_month"] = ts.dt.strftime("%Y-%m")
            pts.to_file(out, layer="mapillary", driver="GPKG")
        report["mapillary"] = (n_raw, n_dedup, len(pts))
    else:
        report["mapillary"] = (0, 0, 0)

    # -- Google ------------------------------------------------------------
    ggl_csv = google.raw_path(campus_id)
    if ggl_csv.exists():
        df = pd.read_csv(ggl_csv, low_memory=False)
        n_raw = len(df)
        df = df.dropna(subset=["pano_id"]).drop_duplicates(
            subset=["pano_id"], keep="first")
        n_dedup = len(df)

        pts = _to_points(df)
        pts = pts[pts.geometry.within(poly)].copy() if not pts.empty else pts

        if not pts.empty:
            if "date" in pts.columns:
                pts["year"] = pd.to_numeric(
                    pts["date"].astype(str).str.slice(0, 4), errors="coerce")
                pts["year_month"] = pts["date"].astype(str).str.slice(0, 7)
            pts.to_file(out, layer="google", driver="GPKG")
        report["google"] = (n_raw, n_dedup, len(pts))
    else:
        report["google"] = (0, 0, 0)

    if verbose:
        print(f"[points/{campus_id}]")
        for src, (raw, dedup, final) in report.items():
            print(f"  {src:<10} raw={raw:>7}  -{raw-dedup} dup  "
                  f"-{dedup-final} outside boundary  final={final:>7}")
        print(f"  -> {out.name}")
    return out


def load_points(campus_id: str, layer: str) -> gpd.GeoDataFrame:
    p = points_path(campus_id)
    if not p.exists():
        raise FileNotFoundError(f"Run the points step for '{campus_id}' first.")
    try:
        return gpd.read_file(p, layer=layer)
    except Exception:                                  # noqa: BLE001
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
