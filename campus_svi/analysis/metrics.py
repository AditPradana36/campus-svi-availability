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
# Local Moran's I (LISA)
# --------------------------------------------------------------------------

QUADRANTS = ["HH", "LH", "LL", "HL", "ns"]
QUADRANT_LABELS = {
    "HH": "High-High (covered cluster)",
    "LL": "Low-Low (gap cluster)",
    "LH": "Low-High (outlier in cluster)",
    "HL": "High-Low (isolated coverage)",
    "ns": "Not significant",
}


def local_morans(cells, column: str, queen: bool = True,
                 permutations: int = 199, seed: int = 0,
                 alpha: float = 0.05) -> pd.DataFrame:
    """Local Moran's I per cell, with cluster quadrant.

    Global Moran's I says whether a campus is clustered; it cannot say *where*.
    The local statistic decomposes it per cell, so a coverage gap in the campus
    core becomes visible as a Low-Low cluster rather than being averaged into
    a single number.

    Quadrants come from the sign of the cell's own standardised value against
    its neighbourhood mean: HH covered cells among covered neighbours, LL gaps
    among gaps, and LH/HL the outliers. Cells that fail the permutation test at
    ``alpha`` are labelled ``ns`` and should be read as no signal rather than
    as a weak one.

    Weights are row-standardised lattice adjacency taken from the row/column
    indices, so no spatial-weights library is needed and adjacency is exact.
    """
    df = cells[["grid_id", "row", "col", column]].dropna().reset_index(drop=True)
    n = len(df)
    if n < 20:
        return pd.DataFrame(columns=["grid_id", "Ii", "z", "lag_z",
                                     "quadrant", "p_sim"])

    pos = {(int(r), int(c)): i for i, (r, c) in
           enumerate(zip(df["row"], df["col"]))}
    x = df[column].to_numpy(float)
    z = x - x.mean()
    sd = z.std()
    if sd == 0:
        out = df[["grid_id"]].copy()
        out["Ii"] = np.nan
        out["z"] = 0.0
        out["lag_z"] = 0.0
        out["quadrant"] = "ns"
        out["p_sim"] = np.nan
        return out

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if queen:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Flat (source, neighbour) index pairs rather than a list of lists: the
    # spatial lag then reduces to one bincount per permutation, which is the
    # difference between this running in seconds and in minutes at 20 m.
    ii, jj = [], []
    for (r, c), i in pos.items():
        for dr, dc in offsets:
            j = pos.get((r + dr, c + dc))
            if j is not None:
                ii.append(i)
                jj.append(j)
    if not ii:
        out = df[["grid_id"]].copy()
        out["Ii"] = np.nan
        out["z"] = 0.0
        out["lag_z"] = 0.0
        out["quadrant"] = "ns"
        out["p_sim"] = np.nan
        return out

    ii = np.asarray(ii)
    jj = np.asarray(jj)
    deg = np.bincount(ii, minlength=n).astype(float)
    deg[deg == 0] = 1.0                      # isolated cells get zero lag

    m2 = (z ** 2).sum() / n

    def _lag(vec):
        """Row-standardised spatial lag: mean of each cell's neighbours."""
        return np.bincount(ii, weights=vec[jj], minlength=n) / deg

    lag = _lag(z)
    Ii = z * lag / m2

    # Conditional permutation: hold each cell fixed, shuffle the rest, and ask
    # how often a random neighbourhood is this extreme.
    rng = np.random.default_rng(seed)
    counts = np.zeros(n)
    abs_Ii = np.abs(Ii)
    for _ in range(permutations):
        perm = rng.permutation(z)
        counts += np.abs(z * _lag(perm) / m2) >= abs_Ii
    p_sim = (counts + 1) / (permutations + 1)

    zs = z / sd
    lag_s = lag / sd
    quad = np.where((zs > 0) & (lag_s > 0), "HH",
           np.where((zs < 0) & (lag_s < 0), "LL",
           np.where((zs < 0) & (lag_s > 0), "LH", "HL")))
    quad = np.where(p_sim < alpha, quad, "ns")

    out = df[["grid_id"]].copy()
    out["Ii"] = Ii
    out["z"] = zs
    out["lag_z"] = lag_s
    out["quadrant"] = quad
    out["p_sim"] = p_sim
    return out


def local_morans_augmenter(column: str, prefix: str, **kw):
    """Return an ``augment(campus_id, cells)`` callable for the map engine.

    Computed lazily per campus at draw time, because the permutation test is
    expensive and only the campuses actually being mapped need it.
    """
    def augment(campus_id, cells):
        res = local_morans(cells, column, **kw)
        if res.empty:
            cells = cells.copy()
            cells[f"{prefix}_Ii"] = np.nan
            cells[f"{prefix}_quadrant"] = "ns"
            return cells
        res = res.rename(columns={"Ii": f"{prefix}_Ii",
                                  "quadrant": f"{prefix}_quadrant",
                                  "p_sim": f"{prefix}_p"})
        keep = ["grid_id", f"{prefix}_Ii", f"{prefix}_quadrant", f"{prefix}_p"]
        return cells.merge(res[keep], on="grid_id", how="left")
    return augment


def local_morans_summary(campus_ids, column: str = "mly_count",
                         permutations: int = 199) -> pd.DataFrame:
    """Share of cells in each LISA quadrant, per campus."""
    from campus_svi import cells as cellsmod

    rows = []
    for cid in campus_ids:
        c = cellsmod.load_cells(cid)
        if column not in c.columns:
            continue
        res = local_morans(c, column, permutations=permutations)
        row = {"campus_id": cid, "n_cells": len(c)}
        if not res.empty:
            vc = res["quadrant"].value_counts(normalize=True)
            for q in QUADRANTS:
                row[f"prop_{q}"] = float(vc.get(q, 0.0))
            row["prop_significant"] = float((res["quadrant"] != "ns").mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Contributors
# --------------------------------------------------------------------------

def contributor_profile(campus_ids) -> pd.DataFrame:
    """Per-contributor image counts, long format, one row per contributor."""
    out = []
    for cid in campus_ids:
        g = points.load_points(cid, "mapillary")
        if g.empty or "creator_id" not in g.columns:
            continue
        vc = (g.groupby(["creator_id", "creator_username"])
              .size().sort_values(ascending=False).reset_index(name="n_images"))
        seqs = g.groupby("creator_id")["sequence_id"].nunique()
        vc["n_sequences"] = vc["creator_id"].map(seqs).fillna(0).astype(int)
        vc["campus_id"] = cid
        vc["rank"] = np.arange(1, len(vc) + 1)
        vc["share"] = vc["n_images"] / vc["n_images"].sum()
        vc["cum_share"] = vc["share"].cumsum()
        out.append(vc)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def contributor_summary(campus_ids) -> pd.DataFrame:
    """Contributor count and concentration per campus.

    ``gini`` is inequality of images across contributors: 0 is every
    contributor mapping equally, near 1 is one person mapping the campus. This
    matters for interpretation — a campus mapped by one student is a different
    phenomenon from one mapped by twenty, even at identical coverage.
    """
    prof = contributor_profile(campus_ids)
    if prof.empty:
        return pd.DataFrame()

    rows = []
    for cid, g in prof.groupby("campus_id"):
        v = np.sort(g["n_images"].to_numpy(float))
        n = len(v)
        gini = ((2 * np.arange(1, n + 1) - n - 1) * v).sum() / (n * v.sum()) \
            if n and v.sum() else np.nan
        rows.append({
            "campus_id": cid,
            "n_contributors": n,
            "n_images": int(v.sum()),
            "top1_share": g["share"].iloc[0] if len(g) else np.nan,
            "top3_share": g["share"].head(3).sum(),
            "median_images": float(np.median(v)),
            "gini": gini,
            # Contributors accounting for the first 90% of images: a compact
            # measure of how many people actually carry the coverage.
            "n_for_90pct": int((g["cum_share"] < 0.9).sum() + 1),
        })
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


def maup_table(campus_ids, sizes=None) -> pd.DataFrame:
    """MAUP results in wide form, one row per campus, one column per size.

    Coverage rises with cell size by construction — a bigger cell is easier to
    intersect — so the absolute numbers are not the finding. What matters is
    whether the campus *ranking* survives, which the correlation row at the
    bottom of the notebook reports.
    """
    long = maup_profile(campus_ids, sizes=sizes)
    if long.empty:
        return long
    from campus_svi import registry

    wide = long.pivot_table(index="campus_id", columns="cell_size_m",
                            values=["mly_coverage", "ggl_coverage",
                                    "either_coverage", "n_cells"])
    wide.columns = [f"{a}_{int(b)}m" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.insert(1, "display_name", registry.display_names(wide["campus_id"]))
    return wide


def write_tables(campus_ids, outdir=None, include_maup: bool = False,
                 maup_sizes=None) -> dict:
    """Compute every table and write it to CSV. Returns {name: path}.

    ``include_maup`` is off by default: it re-grids every campus at each cell
    size, which is the slowest thing in the analysis layer. Turn it on when you
    want the sensitivity table.
    """
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
        "contributors": contributor_summary(campus_ids),
        "contributor_profile": contributor_profile(campus_ids),
    }
    if include_maup:
        built["maup_long"] = maup_profile(campus_ids, sizes=maup_sizes)
        built["maup_wide"] = maup_table(campus_ids, sizes=maup_sizes)
    out = {}
    for name, df in built.items():
        if df is None or df.empty:
            continue
        p = outdir / f"{name}.csv"
        df.to_csv(p, index=False)
        out[name] = p
        print(f"  -> {p.name}  ({len(df)} rows)")
    return out
