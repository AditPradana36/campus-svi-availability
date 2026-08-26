"""Google Street View metadata retrieval via streetlevel, async.

Everything here comes from ``streetlevel`` (verified against the installed
version). No API key is required — the library wraps Google's internal
endpoints, which also means it can break without warning if Google changes
them. Pin the version you publish with.

Two stages, both async over a shared ``aiohttp`` session.

Stage A — coverage tiles (spatial census)
    ``get_coverage_tile_async`` returns *every* panorama on a Slippy Map tile
    at zoom 17 (~304 m square at Indonesian latitudes). This is an area query,
    structurally the same shape as a Mapillary bbox query, and it reaches
    panoramas that a radius search never surfaces. Because tiles are coarser
    than the analysis grid, the campus needs far fewer requests than it has
    cells — roughly 65 tiles for a 6 km2 campus versus 600 cells at 100 m.

    The tile response carries geometry only: id, lat, lon, heading, pitch,
    roll, elevation, and links. No date, no source, no copyright — and only
    the *most recent* coverage at each position.

Stage B — per-panorama enrichment (attributes and temporal depth)
    ``find_panorama_by_id_async`` fills in what the tile omits: capture date,
    ``source`` (the capture programme), ``copyright_message``, ``uploader``,
    and the ``historical`` list of earlier panoramas at that position. The
    historical entries are expanded into their own rows, which is where
    temporal depth actually comes from.

Two payoffs specific to this project
------------------------------------
``is_third_party`` is derived from the pano ID string alone, so the
official-versus-user-contributed split costs nothing — it is available
straight from stage A.

``source`` records the capture programme, and this bears directly on the
road-snapping caveat. ``launch`` is regular car coverage snapped to roads;
``scout`` is trekker or tripod coverage *not* snapped to roads; ``innerspace``
is Business View tripods. On a campus with pedestrian-path trekker coverage,
the road-snapping asymmetry is weaker than a blanket statement would suggest,
and this field lets you measure that rather than assume it.
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

import pandas as pd

from campus_svi import boundaries, config, grids
from campus_svi.checkpoint import Checkpoint

TILE_ZOOM = 17


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def tiles_path(campus_id: str) -> Path:
    return config.RAW_GGL_DIR / f"{campus_id}_google_tiles.csv"


def meta_path(campus_id: str) -> Path:
    return config.RAW_GGL_DIR / f"{campus_id}_google_meta.csv"


def raw_path(campus_id: str) -> Path:
    """Merged stage A + stage B table, written by :func:`merge_stages`."""
    return config.RAW_GGL_DIR / f"{campus_id}_google_raw.csv"


def tile_progress_path(campus_id: str) -> Path:
    return config.RAW_GGL_DIR / f"{campus_id}_google_tiles_progress.jsonl"


def meta_progress_path(campus_id: str) -> Path:
    return config.RAW_GGL_DIR / f"{campus_id}_google_meta_progress.jsonl"


# --------------------------------------------------------------------------
# Event loop helper
# --------------------------------------------------------------------------

def run_async(coro):
    """Run a coroutine from a script or from a notebook.

    Colab and Jupyter already have a running event loop, so ``asyncio.run``
    raises there. ``nest_asyncio`` patches the loop to allow re-entry.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    try:
        import nest_asyncio
    except ImportError as exc:                                  # noqa: BLE001
        raise RuntimeError(
            "A running event loop was detected (notebook). Install nest_asyncio: "
            "pip install nest_asyncio"
        ) from exc
    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------
# Tile enumeration
# --------------------------------------------------------------------------

def tiles_for_campus(campus_id: str, zoom: int = TILE_ZOOM) -> pd.DataFrame:
    """Every zoom-17 tile intersecting the campus boundary.

    Tiles are the fetch unit; the grid remains the analysis unit. Panoramas
    are assigned to grid cells later by spatial join, so the two need not
    align.
    """
    from shapely.geometry import box
    import geopandas as gpd
    from streetlevel.geo import tile_coord_to_wgs84, wgs84_to_tile_coord

    bnd = boundaries.load(campus_id)
    poly = boundaries.dissolve(bnd)
    minx, miny, maxx, maxy = poly.bounds

    x0, y0 = wgs84_to_tile_coord(maxy, minx, zoom)   # NW corner
    x1, y1 = wgs84_to_tile_coord(miny, maxx, zoom)   # SE corner

    rows = []
    for tx in range(min(x0, x1), max(x0, x1) + 1):
        for ty in range(min(y0, y1), max(y0, y1) + 1):
            nw_lat, nw_lon = tile_coord_to_wgs84(tx, ty, zoom)
            se_lat, se_lon = tile_coord_to_wgs84(tx + 1, ty + 1, zoom)
            geom = box(min(nw_lon, se_lon), min(nw_lat, se_lat),
                       max(nw_lon, se_lon), max(nw_lat, se_lat))
            if geom.intersects(poly):
                rows.append({"tile_id": f"{tx}_{ty}", "tile_x": tx, "tile_y": ty,
                             "geometry": geom})

    if not rows:
        raise ValueError(f"No tiles cover '{campus_id}' — check the boundary CRS.")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# --------------------------------------------------------------------------
# Row assembly
# --------------------------------------------------------------------------

def _deg(x):
    """streetlevel returns angles in radians; store degrees for readability."""
    return None if x is None else round(math.degrees(x) % 360, 4)


def _date_parts(d):
    """(display string, year, month, day). Day is third-party only."""
    if d is None:
        return None, None, None, None
    y = getattr(d, "year", None)
    m = getattr(d, "month", None)
    day = getattr(d, "day", None)
    if y is None:
        return str(d), None, None, None
    out = f"{y}-{m:02d}" if m else str(y)
    if day:
        out += f"-{day:02d}"
    return out, y, m, day


def _tile_row(pano, tile_id: str, campus_id: str) -> dict:
    from streetlevel.streetview.util import is_third_party_panoid

    return {
        "source_dataset": "google",
        "campus_id": campus_id,
        "tile_id": tile_id,
        "pano_id": pano.id,
        "lat": pano.lat,
        "lon": pano.lon,
        "heading": _deg(pano.heading),
        "pitch": _deg(pano.pitch),
        "roll": _deg(pano.roll),
        "elevation": pano.elevation,
        "n_links": len(pano.links) if pano.links else 0,
        # Derived from the pano ID string alone — no request needed.
        "is_third_party": is_third_party_panoid(pano.id),
        "found": True,
    }


def _meta_row(pano, pano_id: str, campus_id: str, parent_id: str | None) -> dict:
    date_str, year, month, day = _date_parts(getattr(pano, "date", None))
    up = getattr(pano, "upload_date", None)
    addr = getattr(pano, "address", None)
    return {
        "campus_id": campus_id,
        "pano_id": pano_id,
        "date": date_str,
        "year": year,
        "month": month,
        # Day is present for third-party uploads only; official coverage is
        # month-level. Temporal analysis must bin to year or month, or the two
        # subsets are not comparable.
        "day": day,
        "upload_year": getattr(up, "year", None),
        "upload_month": getattr(up, "month", None),
        # Capture programme: launch = road-snapped car, scout = trekker/tripod
        # NOT snapped to roads, innerspace = Business View tripod.
        "capture_source": getattr(pano, "source", None),
        "copyright_message": getattr(pano, "copyright_message", None),
        "uploader": getattr(pano, "uploader", None),
        "country_code": getattr(pano, "country_code", None),
        "street_name": (addr[0].value if addr else None),
        "building_level": str(getattr(pano, "building_level", None) or "") or None,
        "n_neighbors": len(getattr(pano, "neighbors", None) or []),
        "elevation": getattr(pano, "elevation", None),
        "lat": getattr(pano, "lat", None),
        "lon": getattr(pano, "lon", None),
        "is_historical": parent_id is not None,
        "parent_pano_id": parent_id,
        "enriched": True,
    }


# --------------------------------------------------------------------------
# Stage A — coverage tiles
# --------------------------------------------------------------------------

async def _fetch_tile(sem, session, tile, campus_id: str, ckpt: Checkpoint, out_csv: Path, lock):
    from streetlevel import streetview

    tile_id = tile["tile_id"]
    for attempt in range(1, config.GOOGLE_MAX_RETRIES + 1):
        try:
            async with sem:
                panos = await streetview.get_coverage_tile_async(
                    int(tile["tile_x"]), int(tile["tile_y"]), session
                )
                await asyncio.sleep(config.GOOGLE_SLEEP)

            rows = [_tile_row(p, tile_id, campus_id) for p in (panos or [])]
            async with lock:
                if rows:
                    pd.DataFrame(rows).to_csv(
                        out_csv, mode="a", header=not out_csv.exists(), index=False
                    )
                # An empty tile is a real result: no coverage there.
                ckpt.mark_done(tile_id, n_records=len(rows))
            return len(rows)
        except Exception as exc:                                # noqa: BLE001
            if attempt == config.GOOGLE_MAX_RETRIES:
                async with lock:
                    ckpt.mark_failed(tile_id, error=exc)
                return 0
            await asyncio.sleep(config.GOOGLE_BACKOFF * attempt)
    return 0


async def _stage_a(campus_id: str, tiles: pd.DataFrame, verbose: bool) -> int:
    from aiohttp import ClientSession, ClientTimeout

    ckpt = Checkpoint(tile_progress_path(campus_id))
    pending = ckpt.pending(tiles["tile_id"].tolist())
    if verbose:
        print(f"  stage A: {len(tiles)} tiles total, {len(pending)} pending")
    if not pending:
        return 0

    todo = tiles[tiles["tile_id"].isin(pending)].to_dict("records")
    out_csv = tiles_path(campus_id)
    sem = asyncio.Semaphore(config.GOOGLE_CONCURRENCY)
    lock = asyncio.Lock()

    timeout = ClientTimeout(total=config.GOOGLE_TIMEOUT)
    async with ClientSession(timeout=timeout) as session:
        counts = await asyncio.gather(*[
            _fetch_tile(sem, session, t, campus_id, ckpt, out_csv, lock) for t in todo
        ])

    total = sum(counts)
    if verbose:
        print(f"  stage A: {ckpt.summary()} | {total} panoramas found")
    return total


# --------------------------------------------------------------------------
# Stage B — per-panorama enrichment
# --------------------------------------------------------------------------

async def _fetch_meta(sem, session, pano_id: str, campus_id: str,
                      ckpt: Checkpoint, out_csv: Path, lock,
                      failed: list, stats: dict):
    """Enrich one position. Appends to ``failed`` if it errored, so the
    caller can retry it in a later round."""
    from streetlevel import streetview

    try:
        async with sem:
            pano = await streetview.find_panorama_by_id_async(pano_id, session)
            await asyncio.sleep(config.GOOGLE_SLEEP)
    except Exception:                                           # noqa: BLE001
        async with lock:
            failed.append(pano_id)
            stats["error"] += 1
        return

    if pano is None:
        # A real answer, not a failure: nothing exists under that id.
        async with lock:
            stats["none"] += 1
            ckpt.mark_done(pano_id, n_records=0)
        return

    rows = [_meta_row(pano, pano_id, campus_id, None)]
    if config.GOOGLE_INCLUDE_HISTORICAL:
        for h in (getattr(pano, "historical", None) or []):
            hid = getattr(h, "id", None)
            if hid and hid != pano_id:
                rows.append(_meta_row(h, hid, campus_id, pano_id))

    async with lock:
        pd.DataFrame(rows).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False
        )
        stats["ok"] += 1
        stats["rows"] += len(rows)
        ckpt.mark_done(pano_id, n_records=len(rows))


async def _stage_b(campus_id: str, pano_ids: list[str], verbose: bool) -> int:
    """Enrich every pending position, with automatic retry rounds.

    Failures are collected and re-attempted in their own passes rather than
    being abandoned, because a long run against undocumented endpoints will
    always pick up some transient errors. A ``None`` response is a real answer
    — no panorama under that id — so it is recorded and never retried.
    """
    from aiohttp import ClientSession, ClientTimeout

    ckpt = Checkpoint(meta_progress_path(campus_id))
    pending = ckpt.pending(pano_ids)
    if verbose:
        print(f"  stage B: {len(pano_ids)} positions total, {len(pending)} pending")
    if not pending:
        return 0

    out_csv = meta_path(campus_id)
    sem = asyncio.Semaphore(config.GOOGLE_CONCURRENCY)
    lock = asyncio.Lock()
    stats = {"ok": 0, "none": 0, "error": 0, "rows": 0}

    timeout = ClientTimeout(total=config.GOOGLE_TIMEOUT)
    async with ClientSession(timeout=timeout) as session:
        for rnd in range(config.GOOGLE_RETRY_ROUNDS + 1):
            if not pending:
                break
            if rnd:
                if verbose:
                    print(f"    retry {rnd}: {len(pending)} that failed")
                await asyncio.sleep(config.GOOGLE_BACKOFF * rnd)
                stats["error"] = 0

            failed: list[str] = []
            t_pass = time.time()
            chunk = config.GOOGLE_CHUNK

            for i in range(0, len(pending), chunk):
                batch = pending[i:i + chunk]
                await asyncio.gather(*[
                    _fetch_meta(sem, session, pid, campus_id, ckpt, out_csv,
                                lock, failed, stats)
                    for pid in batch
                ])
                done_n = min(i + chunk, len(pending))
                rate = done_n / max(time.time() - t_pass, 1e-3)
                eta = (len(pending) - done_n) / max(rate, 1e-3)
                if verbose:
                    print(f"    {done_n}/{len(pending)} | ok {stats['ok']} · "
                          f"none {stats['none']} · err {stats['error']} | "
                          f"{stats['rows']} rows | {rate:.1f}/s | "
                          f"ETA {eta/60:.1f} min")

            pending = failed

    if verbose:
        print(f"  stage B: {ckpt.summary()} | {stats['rows']} metadata rows")
        if pending:
            print(f"  !! {len(pending)} unresolved after "
                  f"{config.GOOGLE_RETRY_ROUNDS} retries — re-run to try again, "
                  f"or lower GOOGLE_CONCURRENCY if this is throttling.")
    return stats["rows"]


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------

def merge_stages(campus_id: str, verbose: bool = True) -> Path:
    """Left-join stage B attributes onto stage A geometry.

    Historical panoramas exist only in stage B, so they are appended as their
    own rows, inheriting the position of their parent where they carry none.
    """
    tp, mp = tiles_path(campus_id), meta_path(campus_id)
    if not tp.exists():
        raise FileNotFoundError(f"No tile data for '{campus_id}'. Run stage A first.")

    tiles = pd.read_csv(tp, low_memory=False).drop_duplicates(subset=["pano_id"])

    if not mp.exists():
        out = tiles.copy()
        out["is_historical"] = False
    else:
        meta = pd.read_csv(mp, low_memory=False).drop_duplicates(subset=["pano_id"])
        current = meta[meta["is_historical"] != True]           # noqa: E712
        hist = meta[meta["is_historical"] == True]              # noqa: E712

        out = tiles.merge(
            current.drop(columns=["campus_id", "lat", "lon", "elevation"],
                         errors="ignore"),
            on="pano_id", how="left",
        )

        if not hist.empty:
            parents = tiles.set_index("pano_id")[["lat", "lon", "tile_id"]]
            h = hist.copy()
            h["campus_id"] = campus_id
            h["source_dataset"] = "google"
            h["found"] = True
            for col in ("lat", "lon"):
                fallback = h["parent_pano_id"].map(parents[col])
                h[col] = h[col].where(h[col].notna(), fallback)
            h["tile_id"] = h["parent_pano_id"].map(parents["tile_id"])
            from streetlevel.streetview.util import is_third_party_panoid
            h["is_third_party"] = h["pano_id"].astype(str).map(is_third_party_panoid)
            out = pd.concat([out, h], ignore_index=True)

    out = out.drop_duplicates(subset=["pano_id"])
    out.to_csv(raw_path(campus_id), index=False)

    if verbose:
        n_hist = int(out.get("is_historical", pd.Series(dtype=bool)).fillna(False).sum())
        print(f"  merged: {len(out)} panoramas ({n_hist} historical) "
              f"-> {raw_path(campus_id).name}")
    return raw_path(campus_id)


# --------------------------------------------------------------------------
# Campus runner
# --------------------------------------------------------------------------

def fetch_campus(
    campus_id: str,
    enrich: bool | None = None,
    limit_tiles: int | None = None,
    limit_panos: int | None = None,
    verbose: bool = True,
) -> Path:
    """Fetch Google metadata for one campus: tiles, then enrichment, then merge.

    Both stages are resumable and independently checkpointed, so an
    interrupted run picks up where it stopped. Re-running is safe.
    """
    config.ensure_dirs()
    enrich = config.GOOGLE_ENRICH if enrich is None else enrich

    tiles = tiles_for_campus(campus_id)
    if limit_tiles:
        tiles = tiles.head(limit_tiles)

    if verbose:
        print(f"[google/{campus_id}] streetlevel, async | "
              f"concurrency={config.GOOGLE_CONCURRENCY}")

    run_async(_stage_a(campus_id, tiles, verbose))

    if enrich and tiles_path(campus_id).exists():
        ids = (
            pd.read_csv(tiles_path(campus_id), usecols=["pano_id"])["pano_id"]
            .dropna().astype(str).drop_duplicates().tolist()
        )
        if limit_panos:
            ids = ids[:limit_panos]
        run_async(_stage_b(campus_id, ids, verbose))

    return merge_stages(campus_id, verbose=verbose)


def check_library() -> str:
    """Confirm streetlevel and aiohttp are importable."""
    import aiohttp
    import streetlevel

    ver = getattr(streetlevel, "__version__", "unknown")
    return f"streetlevel {ver}, aiohttp {aiohttp.__version__}"
