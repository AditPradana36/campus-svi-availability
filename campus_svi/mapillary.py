"""Mapillary metadata retrieval (Graph API v4), async.

Strategy, arrived at by testing rather than assumption:

A whole-campus bounding box is **refused outright** by the API — it returns
``HTTP 500: "Please reduce the amount of data you're asking for"`` rather than
truncating silently. That refusal is a usable signal, and it drives everything
here:

1. The campus is cut into **seed boxes** (``MLY_SEED_SIZE_M``), each an atomic
   checkpointed unit so a run resumes exactly where it stopped.
2. Inside a seed, an **adaptive quadtree** subdivides any box the API refuses.
   Refusal depends on fields x limit rather than area alone, so a refused box
   is first retried at half the page limit before being split.
3. A sampled **verification split** covers the other failure mode: a box that
   succeeded but was silently capped. Never test a result count against the
   limit you requested — the effective cap can be lower, in which case the
   check never fires. Verification instead splits the box and asks whether the
   quadrants jointly return more.

Only ``MLY_VERIFY_FRACTION`` of accepted boxes is verified, since explicit
refusal is the dominant mechanism; that sample is evidence rather than a
guarantee, and setting it to 1.0 verifies everything.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import pandas as pd

from campus_svi import config, grids
from campus_svi.checkpoint import Checkpoint


class BboxTooLarge(Exception):
    """The API refused a request as too large. Split the box or lower limit."""


def raw_path(campus_id: str) -> Path:
    return config.RAW_MLY_DIR / f"{campus_id}_mapillary_raw.csv"


def progress_path(campus_id: str) -> Path:
    return config.RAW_MLY_DIR / f"{campus_id}_mapillary_progress.jsonl"


def run_async(coro):
    """Run a coroutine from a script or a notebook."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    try:
        import nest_asyncio
    except ImportError as exc:
        raise RuntimeError(
            "Running event loop detected (notebook). pip install nest_asyncio"
        ) from exc
    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _too_large(text: str, status: int) -> bool:
    t = (text or "").lower()
    return status in (400, 500) and "reduce the amount of data" in t


async def _get(session, url, params, token, stats):
    stats["requests"] += 1
    headers = {"Authorization": f"OAuth {token}"}
    async with session.get(url, params=params, headers=headers) as r:
        if r.status == 429:
            raise RuntimeError("rate limited (429)")
        text = await r.text()
        if _too_large(text, r.status):
            raise BboxTooLarge(text[:160])
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {text[:160]}")
        return json.loads(text)


async def fetch_bbox(session, sem, token, stats, west, south, east, north,
                     limit: int | None = None):
    """Every image in a bbox. Raises BboxTooLarge if refused even at the floor.

    The page limit **ratchets down and is shared** across the whole campus run
    via ``stats['limit']``. Refusal depends on fields x limit, so once the API
    has rejected 500 there is no point every subsequent box rediscovering that
    by halving from 500 again — that turns one refusal into five wasted
    requests, multiplied by every box in the quadtree.
    """
    limit = limit or stats.get("limit") or config.MLY_PAGE_LIMIT
    params = {
        "fields": ",".join(config.MLY_FIELDS),
        "bbox": f"{west},{south},{east},{north}",
        "limit": limit,
    }
    out, url, pages = [], config.GRAPH_URL, 0

    while url and pages < config.MLY_MAX_PAGES:
        try:
            async with sem:
                payload = await _get(session, url, params, token, stats)
                await asyncio.sleep(config.MLY_SLEEP)
        except BboxTooLarge:
            stats["too_large"] += 1
            if limit > config.MLY_MIN_LIMIT:
                lower = max(config.MLY_MIN_LIMIT, limit // 2)
                # Share what we just learned with every later box.
                if lower < stats.get("limit", config.MLY_PAGE_LIMIT):
                    stats["limit"] = lower
                return await fetch_bbox(
                    session, sem, token, stats, west, south, east, north,
                    limit=lower)
            raise
        data = payload.get("data", []) or []
        out.extend(data)
        pages += 1
        nxt = (payload.get("paging") or {}).get("next")
        if not nxt:
            break
        url, params = nxt, None      # the cursor URL carries the query
    return out


# --------------------------------------------------------------------------
# Adaptive quadtree
# --------------------------------------------------------------------------

def _quads(w, s, e, n):
    mx, my = (w + e) / 2, (s + n) / 2
    return [(w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)]


def _uniq(records):
    seen, out = set(), []
    for r in records:
        rid = str(r.get("id"))
        if rid not in seen:
            seen.add(rid)
            out.append(r)
    return out


async def _try_fetch(session, sem, token, stats, q):
    try:
        return await fetch_bbox(session, sem, token, stats, *q)
    except BboxTooLarge:
        return None


async def adaptive(session, sem, token, stats, w, s, e, n,
                   depth=0, prefetched=None, rng=None):
    rng = rng or random
    if prefetched is None:
        try:
            recs = await fetch_bbox(session, sem, token, stats, w, s, e, n)
        except BboxTooLarge:
            # Refused outright: split, no verification needed.
            if depth >= config.MLY_MAX_DEPTH:
                stats["depth_exhausted"] += 1
                return []
            deep = await asyncio.gather(*[
                adaptive(session, sem, token, stats, *q, depth=depth + 1, rng=rng)
                for q in _quads(w, s, e, n)])
            return _uniq([r for g in deep for r in g])
    else:
        recs = prefetched

    stats["boxes"] += 1
    if depth >= config.MLY_MAX_DEPTH:
        if recs:
            stats["depth_exhausted"] += 1
        return recs

    # Verification split, on a sample. Guards against a silent cap.
    if rng.random() >= config.MLY_VERIFY_FRACTION:
        return recs

    stats["verified"] += 1
    quads = _quads(w, s, e, n)
    child = await asyncio.gather(*[
        _try_fetch(session, sem, token, stats, q) for q in quads])

    if any(c is None for c in child):
        deep = await asyncio.gather(*[
            adaptive(session, sem, token, stats, *q, depth=depth + 1,
                     prefetched=cr, rng=rng)
            for q, cr in zip(quads, child)])
        return _uniq([r for g in deep for r in g])

    merged = _uniq([r for g in child for r in g])
    if len(merged) <= len(recs):
        return recs                        # complete; nothing hidden

    stats["truncation_found"] += 1
    deep = await asyncio.gather(*[
        adaptive(session, sem, token, stats, *q, depth=depth + 1,
                 prefetched=cr, rng=rng)
        for q, cr in zip(quads, child)])
    return _uniq([r for g in deep for r in g])


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------

def flatten(rec: dict, campus_id: str, seed_id: str) -> dict:
    geo = (rec.get("geometry") or {}).get("coordinates") or [None, None]
    cgeo = (rec.get("computed_geometry") or {}).get("coordinates") or [None, None]
    creator = rec.get("creator") or {}
    return {
        "source": "mapillary",
        "campus_id": campus_id,
        "seed_id": seed_id,
        "image_id": str(rec.get("id")),
        "lon": geo[0], "lat": geo[1],
        "computed_lon": cgeo[0], "computed_lat": cgeo[1],
        "captured_at": rec.get("captured_at"),          # Unix ms
        "compass_angle": rec.get("compass_angle"),
        "computed_compass_angle": rec.get("computed_compass_angle"),
        "altitude": rec.get("altitude"),
        "computed_altitude": rec.get("computed_altitude"),
        "camera_type": rec.get("camera_type"),
        "is_pano": rec.get("is_pano"),
        "sequence_id": rec.get("sequence"),
        "creator_id": creator.get("id"),
        "creator_username": creator.get("username"),
        "organization_id": rec.get("organization_id"),
    }


# --------------------------------------------------------------------------
# Campus runner
# --------------------------------------------------------------------------

async def _run(campus_id: str, seeds, verbose: bool):
    from aiohttp import ClientSession, ClientTimeout

    token = config.require_mapillary_token()
    ckpt = Checkpoint(progress_path(campus_id))
    pending = ckpt.pending(seeds["seed_id"].tolist())
    if verbose:
        print(f"  {len(seeds)} seed boxes, {len(pending)} pending")
    if not pending:
        return {"requests": 0, "images": 0}

    todo = seeds[seeds["seed_id"].isin(pending)].to_dict("records")
    out_csv = raw_path(campus_id)
    sem = asyncio.Semaphore(config.MLY_CONCURRENCY)
    lock = asyncio.Lock()
    rng = random.Random(0)
    stats = {"requests": 0, "too_large": 0, "boxes": 0, "verified": 0,
             "truncation_found": 0, "depth_exhausted": 0, "images": 0,
             "limit": config.MLY_PAGE_LIMIT}
    t0 = time.time()

    async def one(seed):
        sid = seed["seed_id"]
        for attempt in range(1, config.MLY_MAX_RETRIES + 1):
            try:
                recs = await adaptive(
                    session, sem, token, stats,
                    seed["west"], seed["south"], seed["east"], seed["north"],
                    rng=rng)
                rows = [flatten(r, campus_id, sid) for r in _uniq(recs)]
                async with lock:
                    if rows:
                        pd.DataFrame(rows).to_csv(
                            out_csv, mode="a", header=not out_csv.exists(),
                            index=False)
                        stats["images"] += len(rows)
                    ckpt.mark_done(sid, n_records=len(rows))
                return
            except Exception as exc:                       # noqa: BLE001
                if attempt == config.MLY_MAX_RETRIES:
                    async with lock:
                        ckpt.mark_failed(sid, error=exc)
                    if verbose:
                        print(f"  ! {sid}: {type(exc).__name__}: {exc}")
                else:
                    await asyncio.sleep(config.MLY_BACKOFF * attempt)

    async with ClientSession(timeout=ClientTimeout(total=120)) as session:
        for i in range(0, len(todo), 8):
            await asyncio.gather(*[one(s) for s in todo[i:i + 8]])
            done_n = min(i + 8, len(todo))
            el = time.time() - t0
            eta = (len(todo) - done_n) / max(done_n / max(el, 1e-3), 1e-3)
            if verbose:
                print(f"    {done_n}/{len(todo)} seeds | {stats['images']} images | "
                      f"{stats['requests']} req | ETA {eta/60:.1f} min")

    if verbose:
        print(f"  {ckpt.summary()}")
        print(f"  requests {stats['requests']} | boxes {stats['boxes']} | "
              f"refusals {stats['too_large']} | verified {stats['verified']} | "
              f"silent truncation found {stats['truncation_found']}")
        if stats["limit"] < config.MLY_PAGE_LIMIT:
            print(f"  page limit settled at {stats['limit']} "
                  f"(started {config.MLY_PAGE_LIMIT})")
        if stats["depth_exhausted"]:
            print(f"  !! {stats['depth_exhausted']} box(es) hit MLY_MAX_DEPTH "
                  f"({config.MLY_MAX_DEPTH}) — raise it and re-run to be safe.")
    return stats


def fetch_campus(campus_id: str, seed_size_m: float | None = None,
                 verbose: bool = True) -> Path:
    """Fetch all Mapillary metadata for one campus. Resumable by seed box."""
    config.ensure_dirs()
    seeds = grids.seed_boxes(campus_id, size_m=seed_size_m)
    if verbose:
        print(f"[mapillary/{campus_id}]")
    run_async(_run(campus_id, seeds, verbose))
    return raw_path(campus_id)
