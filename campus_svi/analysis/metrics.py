"""Analysis metrics.

Pure data layer: every function returns a DataFrame and draws nothing. Figures
import from here, so a number that appears in a plot can always be printed as a
table and checked.

Reads the acquisition deliverables in ``data/cells/`` and ``data/points/``.
Nothing here re-fetches from an API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from campus_svi import boundaries, cells as cellsmod, config, grids, points


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def available(campus_ids=None) -> list[str]:
    """Campuses that have cell data on disk."""
    ids = campus_ids or boundaries.list_campuses()
    return [c for c in ids if cellsmod.cells_path(c).exists()]


def load_cells(campus_id: str, with_distance: bool = True):
    """Cell table for one campus, optionally with distance-to-boundary."""
    g = cellsmod.load_cells(campus_id)
    if with_distance:
        g = add_boundary_distance(g, campus_id)
    return g


# --------------------------------------------------------------------------
# Geometry — distance to boundary, compactness
# --------------------------------------------------------------------------

def add_boundary_distance(cells, campus_id: str):
    """Distance from each cell centroid to the campus edge, in metres.

    This is the road-free openness measure. ``dist_norm`` rescales by the
    campus's own maximum so campuses of different size are comparable: 0 is the
    perimeter, 1 the deepest interior point.
    """
    bnd = boundaries.load(campus_id)
    crs_m = boundaries.utm_crs(bnd)
    poly = bnd.to_crs(crs_m).geometry.iloc[0]

    c = cells.to_crs(crs_m)
    cent = c.geometry.centroid
    d = cent.distance(poly.exterior)
    # Negative would mean outside; cells are clipped to the campus already.
    cells = cells.copy()
    cells["dist_edge_m"] = d.values
    mx = float(np.nanmax(d.values)) if len(d) else 0.0
    cells["dist_norm"] = cells["dist_edge_m"] / mx if mx > 0 else 0.0
    return cells


def geometry_table(campus_ids) -> pd.DataFrame:
    """Area, perimeter and Polsby-Popper compactness per campus."""
    rows = []
    for cid in campus_ids:
        bnd = boundaries.load(cid)
        crs_m = boundaries.utm_crs(bnd)
        poly = bnd.to_crs(crs_m).geometry.iloc[0]
        a, p = poly.area, poly.length
        rows.append({
            "campus_id": cid,
            "area_km2": a / 1e6,
            "perimeter_km": p / 1e3,
            # 1.0 is a circle; lower means a more convoluted outline.
            "compactness": (4 * np.pi * a / p ** 2) if p else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. Coverage
# --------------------------------------------------------------------------

def coverage_table(campus_ids) -> pd.DataFrame:
    """One row per campus: coverage ratios, totals, agreement composition."""
    from campus_svi import registry

    rows = []
    for cid in campus_ids:
        c = cellsmod.load_cells(cid)
        n = len(c)
        row = {
            "campus_id": cid, "display_name": registry.display_name(cid),
            "n_cells": n,
            "cell_size_m": int(c["cell_size_m"].iloc[0]) if n else np.nan,
            "mly_coverage": c["mly_coverage"].mean(),
            "ggl_coverage": c["ggl_coverage"].mean(),
            "either_coverage": c["either_coverage"].mean(),
            "mly_images": int(c["mly_count"].sum()),
            "ggl_panoramas": int(c["ggl_count"].sum()),
        }
        for k in ("both", "mapillary_only", "google_only", "neither"):
            row[f"prop_{k}"] = (c["agreement"] == k).mean()
        if "depth_diff" in c.columns:
            row["mean_depth_diff"] = c["depth_diff"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Depth decay — the enclosure test and the openness index
# --------------------------------------------------------------------------

def decay_profile(campus_ids, n_bins: int = 8, normalize: bool = True) -> pd.DataFrame:
    """Coverage as a function of distance into the campus.

    ``normalize`` bins on ``dist_norm`` (0 = perimeter, 1 = deepest point) so
    campuses of different size share an x axis. Set False to bin on raw metres.
    """
    col = "dist_norm" if normalize else "dist_edge_m"
    out = []
    for cid in campus_ids:
        c = load_cells(cid)
        if c.empty:
            continue
        edges = np.linspace(0, c[col].max() if c[col].max() > 0 else 1, n_bins + 1)
        c = c.copy()
        c["_bin"] = pd.cut(c[col], edges, include_lowest=True, labels=False)
        g = c.groupby("_bin")
        prof = pd.DataFrame({
            "mly": g["mly_coverage"].mean(),
            "ggl": g["ggl_coverage"].mean(),
            "either": g["either_coverage"].mean(),
            "n_cells": g.size(),
        }).reset_index()
        prof["campus_id"] = cid
        prof["bin_mid"] = [(edges[int(b)] + edges[int(b) + 1]) / 2
                           for b in prof["_bin"]]
        out.append(prof)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def decay_slope(campus_ids, source: str = "either") -> pd.DataFrame:
    """Openness index: the slope of coverage against normalised depth.

    Fitted per campus by ordinary least squares on the cell-level data, not on
    the binned profile, so bin choice does not drive the coefficient. A steeply
    negative slope means coverage collapses as you move inward — a more
    enclosed campus. Near zero means coverage holds to the core.
    """
    key = {"mly": "mly_coverage", "ggl": "ggl_coverage",
           "either": "either_coverage"}[source]
    rows = []
    for cid in campus_ids:
        c = load_cells(cid)
        if len(c) < 10:
            continue
        x, y = c["dist_norm"].to_numpy(float), c[key].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 10 or x.std() == 0:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        # A campus with uniform coverage has no correlation to report; np would
        # emit a divide warning and return nan.
        r = np.corrcoef(x, y)[0, 1] if y.std() > 0 else np.nan
        rows.append({
            "campus_id": cid, "source": source,
            "openness_slope": slope,        # negative = coverage decays inward
            "intercept": intercept,
            "r": r, "r2": r ** 2,
            "edge_coverage": y[x < 0.25].mean() if (x < 0.25).any() else np.nan,
            "core_coverage": y[x > 0.75].mean() if (x > 0.75).any() else np.nan,
            "n_cells": len(x),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["edge_core_gap"] = df["edge_coverage"] - df["core_coverage"]
    return df


# --------------------------------------------------------------------------
# 4 & 5. Temporal
# --------------------------------------------------------------------------

def temporal_annual(campus_ids) -> pd.DataFrame:
    """Records per capture year, per source, per campus (long format)."""
    out = []
    for cid in campus_ids:
        for layer, name in (("mapillary", "mapillary"), ("google", "google")):
            g = points.load_points(cid, layer)
            if g.empty or "year" not in g.columns:
                continue
            s = (pd.to_numeric(g["year"], errors="coerce").dropna()
                 .astype(int).value_counts().sort_index())
            out.append(pd.DataFrame({"campus_id": cid, "source": name,
                                     "year": s.index, "n": s.values}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def monthly_mapillary(campus_ids) -> pd.DataFrame:
    """Mapillary images per calendar month — where contribution bursts show."""
    out = []
    for cid in campus_ids:
        g = points.load_points(cid, "mapillary")
        if g.empty or "year_month" not in g.columns:
            continue
        s = g["year_month"].dropna().value_counts().sort_index()
        out.append(pd.DataFrame({"campus_id": cid, "year_month": s.index,
                                 "n": s.values}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def temporal_signature(campus_ids) -> pd.DataFrame:
    """Burstiness and contributor concentration per campus.

    ``cv_monthly`` is the coefficient of variation of monthly Mapillary counts
    over the observed span, zero-filled for months with no capture. High values
    mean contribution arrived in bursts — a mapping party, a thesis — rather
    than steadily. ``top_creator_share`` is the largest single contributor's
    share of images: a campus mapped by one student is a different phenomenon
    from one mapped by twenty.
    """
    rows = []
    for cid in campus_ids:
        g = points.load_points(cid, "mapillary")
        row = {"campus_id": cid}
        if not g.empty and "year_month" in g.columns:
            s = g["year_month"].dropna()
            if len(s):
                idx = pd.period_range(min(s), max(s), freq="M")
                counts = (s.value_counts().reindex(idx.astype(str))
                          .fillna(0).to_numpy(float))
                row["n_months_span"] = len(counts)
                row["n_months_active"] = int((counts > 0).sum())
                row["cv_monthly"] = (counts.std() / counts.mean()
                                     if counts.mean() else np.nan)
        if not g.empty and "creator_id" in g.columns:
            vc = g["creator_id"].value_counts()
            if len(vc):
                row["n_creators"] = int(len(vc))
                row["top_creator_share"] = vc.iloc[0] / vc.sum()
        if not g.empty and "sequence_id" in g.columns:
            row["n_sequences"] = int(g["sequence_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def temporal_depth(campus_ids) -> pd.DataFrame:
    """Distinct capture years per cell, both sources, plus their difference."""
    out = []
    for cid in campus_ids:
        c = cellsmod.load_cells(cid)
        keep = [k for k in ("grid_id", "campus_id", "mly_n_years", "ggl_n_years",
                            "depth_diff", "agreement") if k in c.columns]
        out.append(c[keep])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------
# 6. Capture programme
# --------------------------------------------------------------------------

def programme_table(campus_ids) -> pd.DataFrame:
    """Google capture programme composition per campus.

    Computed from the point data rather than averaging the per-cell ratios,
    which would weight a cell with one panorama the same as one with fifty.
    """
    rows = []
    for cid in campus_ids:
        g = points.load_points(cid, "google")
        row = {"campus_id": cid, "n_panoramas": len(g)}
        if not g.empty and "capture_source" in g.columns:
            src = g["capture_source"].astype(str)
            n = len(g)
            for name in ("launch", "scout", "innerspace"):
                row[name] = (src == name).sum() / n
            row["third_party"] = (
                src.str.startswith("photos:").sum() / n
                if src.str.startswith("photos:").any() else 0.0)
            row["other"] = max(0.0, 1.0 - sum(
                row.get(k, 0.0) for k in ("launch", "scout", "innerspace",
                                          "third_party")))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 8. Spatial autocorrelation
# --------------------------------------------------------------------------

def morans_i(cells, column: str, queen: bool = True,
             permutations: int = 199, seed: int = 0) -> dict:
    """Moran's I on a regular grid, using row/col adjacency.

    Because the cells are a lattice, neighbours come straight from the row and
    column indices — no spatial weights library needed, and no ambiguity about
    what counts as adjacent. Significance is by permutation, which makes no
    distributional assumption.
    """
    df = cells[["row", "col", column]].dropna()
    if len(df) < 10:
        return {"column": column, "I": np.nan, "p_sim": np.nan, "n": len(df)}

    pos = {(int(r), int(c)): i for i, (r, c) in
           enumerate(zip(df["row"], df["col"]))}
    x = df[column].to_numpy(float)
    n = len(x)

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if queen:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    ii, jj = [], []
    for (r, c), i in pos.items():
        for dr, dc in offsets:
            j = pos.get((r + dr, c + dc))
            if j is not None:
                ii.append(i)
                jj.append(j)
    if not ii:
        return {"column": column, "I": np.nan, "p_sim": np.nan, "n": n}

    ii, jj = np.array(ii), np.array(jj)
    W = len(ii)

    def _I(v):
        z = v - v.mean()
        denom = (z ** 2).sum()
        if denom == 0:
            return np.nan
        return (n / W) * (z[ii] * z[jj]).sum() / denom

    obs = _I(x)
    rng = np.random.default_rng(seed)
    sims = np.array([_I(rng.permutation(x)) for _ in range(permutations)])
    sims = sims[np.isfinite(sims)]
    if not len(sims) or not np.isfinite(obs):
        p = np.nan
    else:
        # Two-sided pseudo p-value.
        p = (1 + (np.abs(sims - sims.mean()) >= abs(obs - sims.mean())).sum()) \
            / (1 + len(sims))
    return {"column": column, "I": obs, "expected": -1 / (n - 1),
            "p_sim": p, "n": n, "n_links": W}


def autocorrelation_table(campus_ids, columns=("either_coverage",
                                               "mly_coverage", "ggl_coverage"),
                          permutations: int = 199) -> pd.DataFrame:
    """Moran's I per campus per variable.

    Adjacent cells are not independent, so ordinary significance tests on
    cell-level data are invalid. Reporting I is what licenses any inferential
    claim made at cell level.
    """
    rows = []
    for cid in campus_ids:
        c = cellsmod.load_cells(cid)
        for col in columns:
            if col not in c.columns:
                continue
            r = morans_i(c, col, permutations=permutations)
            r["campus_id"] = cid
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 7. MAUP sensitivity
# --------------------------------------------------------------------------

def rebuild_cells_at(campus_id: str, cell_size_m: float):
    """Re-aggregate existing points onto a grid of a different cell size.

    No refetching: the point deliverable is resolution-independent, so the
    whole MAUP check is a local reprojection exercise.
    """
    import geopandas as gpd

    grid = grids.build_grid(campus_id, cell_size_m=cell_size_m)
    mly = cellsmod._assign(points.load_points(campus_id, "mapillary"), grid)
    ggl = cellsmod._assign(points.load_points(campus_id, "google"), grid)

    wide = grid.copy()
    for agg in (cellsmod._agg_mapillary(mly), cellsmod._agg_google(ggl)):
        if not agg.empty:
            wide = wide.merge(agg, on="grid_id", how="left")
    for c in ("mly_count", "ggl_count"):
        if c not in wide.columns:
            wide[c] = 0
        wide[c] = wide[c].fillna(0).astype(int)
    wide["mly_coverage"] = (wide["mly_count"] > 0).astype(int)
    wide["ggl_coverage"] = (wide["ggl_count"] > 0).astype(int)
    wide["either_coverage"] = (
        (wide["mly_coverage"] + wide["ggl_coverage"]) > 0).astype(int)
    return wide


def maup_profile(campus_ids, sizes=(20, 50, 100)) -> pd.DataFrame:
    """Coverage ratios recomputed at several cell sizes.

    Grid-based results depend on cell size, and a reviewer will ask. Coverage
    ratios rise with cell size by construction — a bigger cell is easier to
    intersect — so what matters is whether the *ranking* of campuses and the
    Mapillary/Google gap survive.
    """
    sizes = tuple(sizes or config.MAUP_SIZES)
    rows = []
    for cid in campus_ids:
        for s in sizes:
            try:
                w = rebuild_cells_at(cid, s)
            except Exception as exc:                     # noqa: BLE001
                print(f"  ! {cid} @ {s} m: {type(exc).__name__}: {exc}")
                continue
            rows.append({
                "campus_id": cid, "cell_size_m": s, "n_cells": len(w),
                "mly_coverage": w["mly_coverage"].mean(),
                "ggl_coverage": w["ggl_coverage"].mean(),
                "either_coverage": w["either_coverage"].mean(),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 9-11. Source descriptives
# --------------------------------------------------------------------------

def descriptives(campus_ids) -> pd.DataFrame:
    """Density, contributor structure, and the position/history split."""
    rows = []
    for cid in campus_ids:
        c = cellsmod.load_cells(cid)
        cov = c[c["mly_coverage"] == 1]
        row = {
            "campus_id": cid,
            "mly_median_per_covered_cell": cov["mly_count"].median() if len(cov) else 0,
        }
        for k in ("mly_n_sequences", "mly_n_creators", "mly_pano_ratio",
                  "ggl_n_positions", "ggl_n_historical"):
            if k in c.columns:
                row[k] = c[k].sum() if k.startswith(("mly_n", "ggl_n")) else c[k].mean()
        g = points.load_points(cid, "mapillary")
        if not g.empty:
            row["mly_total_sequences"] = g["sequence_id"].nunique()
            row["mly_total_creators"] = g["creator_id"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(campus_ids, outdir=None) -> dict:
    """Compute every table and write it to CSV. Returns {name: path}."""
    from pathlib import Path

    outdir = Path(outdir) if outdir else config.DATA_DIR / "analysis" / "tables"
    outdir.mkdir(parents=True, exist_ok=True)

    built = {
        "coverage": coverage_table(campus_ids),
        "geometry": geometry_table(campus_ids),
        "decay_profile": decay_profile(campus_ids),
        "decay_slope": decay_slope(campus_ids),
        "temporal_annual": temporal_annual(campus_ids),
        "temporal_signature": temporal_signature(campus_ids),
        "programme": programme_table(campus_ids),
        "descriptives": descriptives(campus_ids),
    }
    out = {}
    for name, df in built.items():
        if df is None or df.empty:
            continue
        p = outdir / f"{name}.csv"
        df.to_csv(p, index=False)
        out[name] = p
        print(f"  -> {p.name}  ({len(df)} rows)")
    return out
