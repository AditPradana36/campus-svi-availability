"""Configuration for the campus SVI acquisition pipeline.

Acquisition only: this repo produces **point data** (every street-view image or
panorama inside a campus, with its metadata) and **cell data** (those points
aggregated onto a regular grid). Analysis lives elsewhere.

Paths default to the Drive layout already in use:

    /content/drive/MyDrive/campus-svi-availability/
        boundaries/          ui_main.gpkg, itb_ganesha.gpkg, ...
        data/                everything this pipeline writes
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DRIVE_ROOT = Path(os.environ.get(
    "CAMPUS_SVI_ROOT", "/content/drive/MyDrive/campus-svi-availability"))

BOUNDARY_DIR = Path(os.environ.get("CAMPUS_SVI_BOUNDARY_DIR", DRIVE_ROOT / "boundaries"))
DATA_DIR = DRIVE_ROOT / "data"

GRID_DIR = DATA_DIR / "grids"            # {campus}_grid.gpkg
RAW_DIR = DATA_DIR / "raw"               # per-source fetch output + checkpoints
RAW_MLY_DIR = RAW_DIR / "mapillary"
RAW_GGL_DIR = RAW_DIR / "google"
POINT_DIR = DATA_DIR / "points"          # {campus}_points.gpkg  <- deliverable
CELL_DIR = DATA_DIR / "cells"            # {campus}_cells.gpkg   <- deliverable
LOG_DIR = DATA_DIR / "logs"


def _derive() -> None:
    global DATA_DIR, GRID_DIR, RAW_DIR, RAW_MLY_DIR, RAW_GGL_DIR
    global POINT_DIR, CELL_DIR, LOG_DIR
    DATA_DIR = DRIVE_ROOT / "data"
    GRID_DIR = DATA_DIR / "grids"
    RAW_DIR = DATA_DIR / "raw"
    RAW_MLY_DIR = RAW_DIR / "mapillary"
    RAW_GGL_DIR = RAW_DIR / "google"
    POINT_DIR = DATA_DIR / "points"
    CELL_DIR = DATA_DIR / "cells"
    LOG_DIR = DATA_DIR / "logs"


def set_root(root: str | Path, boundary_dir: str | Path | None = None) -> dict:
    """Point the pipeline at a different project root.

        config.set_root('/content/drive/MyDrive/campus-svi-availability')
    """
    global DRIVE_ROOT, BOUNDARY_DIR
    DRIVE_ROOT = Path(root)
    BOUNDARY_DIR = Path(boundary_dir) if boundary_dir else DRIVE_ROOT / "boundaries"
    _derive()
    ensure_dirs()
    return paths()


def ensure_dirs() -> None:
    """Create output directories. BOUNDARY_DIR is never created — if it is
    missing the path is wrong, and a silent empty folder would hide that."""
    for d in (DATA_DIR, GRID_DIR, RAW_DIR, RAW_MLY_DIR, RAW_GGL_DIR,
              POINT_DIR, CELL_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def paths() -> dict:
    return {"boundaries": BOUNDARY_DIR, "grids": GRID_DIR,
            "raw_mapillary": RAW_MLY_DIR, "raw_google": RAW_GGL_DIR,
            "points": POINT_DIR, "cells": CELL_DIR, "logs": LOG_DIR}


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

# Mapillary token, "MLY|<app_id>|<token>". The only credential needed:
# streetlevel wraps Google's internal endpoints and requires no key.
MAPILLARY_TOKEN = os.environ.get("MAPILLARY_TOKEN", "")


def require_mapillary_token() -> str:
    tok = MAPILLARY_TOKEN or os.environ.get("MAPILLARY_TOKEN", "")
    if not tok:
        raise RuntimeError("Set MAPILLARY_TOKEN before running the Mapillary step.")
    return tok


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------

CELL_SIZE_M = 100          # analysis cell edge, metres, in local UTM
MIN_CELL_OVERLAP = 0.05    # drop boundary slivers from coverage denominators

# --------------------------------------------------------------------------
# Mapillary — adaptive subdivision over seed boxes
# --------------------------------------------------------------------------

GRAPH_URL = "https://graph.mapillary.com/images"

MLY_FIELDS = [
    "id", "captured_at", "geometry", "computed_geometry",
    "compass_angle", "computed_compass_angle",
    "altitude", "computed_altitude",
    "camera_type", "is_pano", "sequence", "creator", "organization_id",
]

# A whole-campus bbox is refused outright by the API, so the AOI is first cut
# into seed boxes of this size. Each seed is an atomic, resumable unit; the
# adaptive quadtree runs inside it.
MLY_SEED_SIZE_M = 500

MLY_PAGE_LIMIT = 500       # the API refuses on fields x limit, not area alone
MLY_MIN_LIMIT = 50         # floor before a box is declared too large
MLY_MAX_PAGES = 20
MLY_MAX_DEPTH = 10         # quadtree depth inside one seed box

# Mapillary refuses oversized boxes explicitly rather than truncating silently,
# so refusal drives subdivision and full verification is unnecessary. Only this
# fraction of accepted boxes gets the four-quadrant check, as evidence that
# nothing is being silently truncated. 1.0 verifies everything.
MLY_VERIFY_FRACTION = 0.1

MLY_CONCURRENCY = 8
MLY_SLEEP = 0.15
MLY_MAX_RETRIES = 3
MLY_BACKOFF = 5.0

# --------------------------------------------------------------------------
# Google — streetlevel, two stages
# --------------------------------------------------------------------------

# Stage A: coverage tiles at zoom 17 (~304 m square at Indonesian latitudes),
# returning every panorama on the tile but no dates.
# Stage B: one request per position for date, capture source, copyright and
# the historical list — the only source of temporal depth.
GOOGLE_ENRICH = True
GOOGLE_INCLUDE_HISTORICAL = True

GOOGLE_CONCURRENCY = 8
GOOGLE_SLEEP = 0.3
GOOGLE_TIMEOUT = 30
GOOGLE_MAX_RETRIES = 3
GOOGLE_BACKOFF = 5.0
GOOGLE_RETRY_ROUNDS = 2
GOOGLE_CHUNK = 100
