"""Campus boundary ingestion.

Boundary files live in ``boundaries/`` and are named by campus id:
``ui_main.gpkg``, ``itb_ganesha.gpkg``, ``unair_b.gpkg`` and so on. The
filename stem becomes the ``campus_id`` used as the join key throughout, so a
campus with several sites simply gets several files.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from campus_svi import config

SUPPORTED_EXT = (".shp", ".gpkg", ".geojson", ".json")


def dissolve(gdf: gpd.GeoDataFrame):
    """Union all features into a single geometry, across geopandas versions."""
    try:
        return gdf.union_all()          # geopandas >= 1.0
    except AttributeError:
        return gdf.unary_union          # geopandas < 1.0


def list_campuses(boundary_dir: str | Path | None = None) -> list[str]:
    """Campus ids discovered in the boundary directory, sorted."""
    d = Path(boundary_dir or config.BOUNDARY_DIR)
    found = set()
    for ext in SUPPORTED_EXT:
        for p in d.glob(f"*{ext}"):
            found.add(p.stem.lower())
    return sorted(found)


def resolve_path(campus_id: str, boundary_dir: str | Path | None = None) -> Path:
    """Locate the boundary file for a campus id, preferring GeoPackage."""
    d = Path(boundary_dir or config.BOUNDARY_DIR)
    for ext in (".gpkg", ".shp", ".geojson", ".json"):
        p = d / f"{campus_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No boundary file for campus '{campus_id}' in {d}. "
        f"Expected one of: {', '.join(campus_id + e for e in SUPPORTED_EXT)}"
    )


def load(campus_id: str, boundary_dir: str | Path | None = None) -> gpd.GeoDataFrame:
    """Load one campus boundary as a single-row GeoDataFrame in EPSG:4326.

    Repairs invalid geometry (a common artefact of hand-digitised polygons),
    dissolves multi-part boundaries into one feature, and assumes WGS84 when
    the source file carries no CRS.
    """
    path = resolve_path(campus_id, boundary_dir)
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"Boundary file {path} contains no features.")

    if gdf.crs is None:
        print(f"  ! {path.name} has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"  ! repairing {int(invalid.sum())} invalid geometry/geometries.")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    merged = gpd.GeoDataFrame(
        {"campus_id": [campus_id]},
        geometry=[dissolve(gdf)],
        crs=gdf.crs,
    )
    return merged.to_crs("EPSG:4326")


def utm_crs(gdf: gpd.GeoDataFrame):
    """Local metric CRS for accurate distance and area work."""
    try:
        return gdf.estimate_utm_crs()
    except Exception:
        # Fallback for older geopandas: derive the UTM zone from the centroid.
        c = gdf.to_crs("EPSG:4326").geometry.iloc[0].centroid
        zone = int((c.x + 180) // 6) + 1
        epsg = 32600 + zone if c.y >= 0 else 32700 + zone
        return f"EPSG:{epsg}"
