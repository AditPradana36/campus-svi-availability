"""Whole-campus runner and progress reporting.

With 22 campuses, the useful unit is "run this campus end to end, resuming
whatever is already done" plus "show me where everything stands". Both are
here.

Every stage is independently checkpointed, so ``run_campus`` is safe to call
repeatedly: finished work is skipped, unfinished work resumes.
"""

from __future__ import annotations

import time
import traceback

import pandas as pd

from campus_svi import boundaries, cells, config, google, grids, mapillary, points

STAGES = ("grid", "mapillary", "google", "points", "cells")


def run_campus(campus_id: str, stages=STAGES, cell_size_m=None,
               seed_size_m=None, enrich=None, verbose=True) -> dict:
    """Run one campus end to end. Returns a per-stage status dict."""
    config.ensure_dirs()
    result, t_start = {}, time.time()

    if verbose:
        print(f"\n{'='*62}\n{campus_id}\n{'='*62}")

    for stage in stages:
        t0 = time.time()
        try:
            if stage == "grid":
                g = grids.build_grid(campus_id, cell_size_m=cell_size_m)
                grids.save_grid(g, campus_id)
                if verbose:
                    print(f"[grid/{campus_id}] {len(g)} cells at "
                          f"{g['cell_size_m'].iloc[0]:g} m")
            elif stage == "mapillary":
                mapillary.fetch_campus(campus_id, seed_size_m=seed_size_m,
                                       verbose=verbose)
            elif stage == "google":
                google.fetch_campus(campus_id, enrich=enrich, verbose=verbose)
            elif stage == "points":
                points.build_points(campus_id, verbose=verbose)
            elif stage == "cells":
                cells.build_cells(campus_id, verbose=verbose)
            result[stage] = {"ok": True, "seconds": round(time.time() - t0, 1)}
        except Exception as exc:                                # noqa: BLE001
            result[stage] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                             "seconds": round(time.time() - t0, 1)}
            if verbose:
                print(f"  !! {stage} failed: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)
            # Later stages depend on earlier ones; stop this campus here.
            break

    result["_total_seconds"] = round(time.time() - t_start, 1)
    if verbose:
        print(f"\n{campus_id}: {result['_total_seconds']}s")
    return result


def run_all(campus_ids=None, stages=STAGES, stop_on_error=False, **kw) -> pd.DataFrame:
    """Run several campuses in sequence.

    One campus at a time by design: it keeps request rates civil and makes a
    failure easy to attribute. A campus that fails does not stop the rest
    unless ``stop_on_error``.
    """
    campus_ids = campus_ids or boundaries.list_campuses()
    rows = []
    for i, cid in enumerate(campus_ids, 1):
        print(f"\n### {i}/{len(campus_ids)}")
        r = run_campus(cid, stages=stages, **kw)
        row = {"campus_id": cid, "seconds": r.pop("_total_seconds")}
        failed = [s for s, v in r.items() if not v["ok"]]
        row["failed_stage"] = failed[0] if failed else None
        row["error"] = r[failed[0]]["error"] if failed else None
        rows.append(row)
        if failed and stop_on_error:
            print("stopping: stop_on_error=True")
            break

    df = pd.DataFrame(rows)
    p = config.LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    config.ensure_dirs()
    df.to_csv(p, index=False)
    print(f"\nrun log -> {p}")
    return df


def status(campus_ids=None) -> pd.DataFrame:
    """What exists on disk for each campus, and how far each fetch has got."""
    from campus_svi.checkpoint import Checkpoint

    campus_ids = campus_ids or boundaries.list_campuses()
    rows = []
    for cid in campus_ids:
        row = {"campus_id": cid}

        row["grid"] = grids.grid_path(cid).exists()
        try:
            row["n_cells"] = len(grids.load_grid(cid)) if row["grid"] else 0
        except Exception:                                       # noqa: BLE001
            row["n_cells"] = 0

        try:
            seeds = len(grids.seed_boxes(cid)) if row["grid"] else 0
        except Exception:                                       # noqa: BLE001
            seeds = 0
        ck = Checkpoint(mapillary.progress_path(cid))
        row["mly_seeds"] = f"{len(ck.done)}/{seeds}" if seeds else "-"
        row["mly_failed"] = len(ck.failed)

        ckt = Checkpoint(google.tile_progress_path(cid))
        ckm = Checkpoint(google.meta_progress_path(cid))
        row["ggl_tiles"] = len(ckt.done)
        row["ggl_enriched"] = len(ckm.done)

        row["points"] = points.points_path(cid).exists()
        row["cells"] = cells.cells_path(cid).exists()
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
