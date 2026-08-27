"""Statistics and figures.

Analyses are grouped by what each source can actually support:

Mapillary only   density, sequence and contributor diversity, panoramic ratio,
                 fine-grained temporal distribution (millisecond timestamps).
Google only      binary coverage, official versus user-contributed split,
                 historical capture-year depth (month-level dates).
Cross-source     the cell-level agreement matrix, coverage ratios, temporal
                 depth comparison — the metrics that are genuinely comparable.

Figures use the paper-figure-style house format (``paperstyle.py``), authored
at final printed size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from campus_svi import config, finalize, unify
from campus_svi import paperstyle as ps

AGREEMENT_COLORS = {
    "both": "#4c4c4c",
    "mapillary_only": "#7b9e87",
    "google_only": "#9e7b7b",
    "neither": "#e8e8e8",
}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def coverage_summary(campus_id: str) -> pd.DataFrame:
    """Cross-source coverage summary for one campus, one row."""
    cells = unify.load_cells(campus_id)
    n = len(cells)
    row = {
        "campus_id": campus_id,
        "n_cells": n,
        "cell_size_m": int(cells["cell_size_m"].iloc[0]) if n else None,
        "mly_coverage_ratio": cells["mly_coverage"].mean(),
        "ggl_coverage_ratio": cells["ggl_coverage"].mean(),
        "either_coverage_ratio": cells["either_coverage"].mean(),
        "mly_total_images": int(cells["mly_count"].sum()),
        "ggl_total_panos": int(cells["ggl_count"].sum()),
    }
    for k in unify.AGREEMENT_CLASSES:
        row[f"prop_{k}"] = (cells["agreement"] == k).mean()
    if "depth_diff" in cells.columns:
        row["mean_depth_diff"] = cells["depth_diff"].mean()
    return pd.DataFrame([row])


def mapillary_profile(campus_id: str) -> pd.DataFrame:
    """Mapillary-only descriptors: density, diversity, temporal spread."""
    mly = finalize.load_final(campus_id, "mapillary")
    if mly.empty:
        return pd.DataFrame()
    cells = unify.load_cells(campus_id)
    covered = cells[cells["mly_coverage"] == 1]
    row = {
        "campus_id": campus_id,
        "n_images": len(mly),
        "n_sequences": mly["sequence_id"].nunique(),
        "n_creators": mly["creator_id"].nunique(),
        "n_organizations": mly["organization_id"].nunique(),
        "pano_ratio": mly["is_pano"].mean() if "is_pano" in mly else None,
        "median_images_per_covered_cell": covered["mly_count"].median() if len(covered) else 0,
        "year_min": mly["year"].min() if "year" in mly else None,
        "year_max": mly["year"].max() if "year" in mly else None,
    }
    return pd.DataFrame([row])


def google_profile(campus_id: str) -> pd.DataFrame:
    """Google-only descriptors, including the official/user-contributed split."""
    ggl = finalize.load_final(campus_id, "google")
    if ggl.empty:
        return pd.DataFrame()
    row = {
        "campus_id": campus_id,
        "n_panos": len(ggl),
        "n_capture_periods": ggl["year_month"].nunique() if "year_month" in ggl else None,
        "n_unique_positions": ggl["pano_id"].nunique(),
        "year_min": ggl["year"].min() if "year" in ggl else None,
        "year_max": ggl["year"].max() if "year" in ggl else None,
    }
    from campus_svi.unify import _as_bool

    if "is_third_party" in ggl.columns:
        tp = _as_bool(ggl["is_third_party"])
        row["third_party_ratio"] = tp.mean()
        row["official_google_ratio"] = 1.0 - tp.mean()
    if "is_historical" in ggl.columns:
        row["n_historical"] = int(_as_bool(ggl["is_historical"]).fillna(0).sum())
    if "capture_source" in ggl.columns:
        counts = ggl["capture_source"].astype(str).value_counts()
        row["capture_sources"] = "; ".join(f"{k}={v}" for k, v in counts.items())
        # `scout` is trekker/tripod coverage not snapped to roads.
        row["scout_ratio"] = float((ggl["capture_source"].astype(str) == "scout").mean())
    return pd.DataFrame([row])


def temporal_profile(campus_id: str) -> pd.DataFrame:
    """Images/panos per capture year, long format, both sources."""
    frames = []
    for layer, label, col in (("mapillary", "mapillary", "year"), ("google", "google", "year")):
        gdf = finalize.load_final(campus_id, layer)
        if gdf.empty or col not in gdf.columns:
            continue
        s = gdf[col].dropna().astype(int).value_counts().sort_index()
        frames.append(pd.DataFrame({
            "campus_id": campus_id, "source": label,
            "year": s.index, "n": s.values,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_agreement_map(campus_id: str, save: bool = True):
    """Cell-level agreement map — the core discrepancy figure."""
    ps.apply()
    cells = unify.load_cells(campus_id)

    fig, ax = plt.subplots(figsize=(ps.HALF_W, ps.HALF_W))
    handles = []
    for cls, colour in AGREEMENT_COLORS.items():
        sub = cells[cells["agreement"] == cls]
        if sub.empty:
            continue
        sub.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.2)
        n = len(sub)
        handles.append(Patch(facecolor=colour, edgecolor="none",
                             label=f"{cls.replace('_', ' ')} ({n})"))

    ax.set_axis_off()
    ax.set_title(f"{campus_id.upper()} — SVI source agreement", pad=6)
    # GeoDataFrame.plot draws a PatchCollection, which matplotlib will not
    # accept as a legend handle — build the swatches explicitly instead.
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=6)
    fig.tight_layout()

    if save:
        _save(fig, f"{campus_id}_agreement_map")
    return fig, ax


def plot_coverage_bars(campus_ids: list[str], save: bool = True):
    """Coverage ratio by source across campuses."""
    ps.apply()
    rows = [coverage_summary(c) for c in campus_ids]
    df = pd.concat([r for r in rows if not r.empty], ignore_index=True)

    fig, ax = plt.subplots(figsize=(ps.COL_W, 2.4))
    x = range(len(df))
    w = 0.38
    ax.bar([i - w / 2 for i in x], df["mly_coverage_ratio"], width=w,
           color=ps.MUTED[0], label="Mapillary")
    ax.bar([i + w / 2 for i in x], df["ggl_coverage_ratio"], width=w,
           color=ps.FOCAL, label="Google")

    ax.set_xticks(list(x))
    ax.set_xticklabels([c.upper() for c in df["campus_id"]])
    ax.set_ylabel("Proportion of cells with coverage")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5)
    ps.finish(ax)
    fig.tight_layout()

    if save:
        _save(fig, "coverage_by_campus")
    return fig, ax


def plot_temporal(campus_id: str, save: bool = True):
    """Capture year distribution, both sources."""
    ps.apply()
    df = temporal_profile(campus_id)
    if df.empty:
        print(f"  ! no temporal data for {campus_id}")
        return None

    fig, ax = plt.subplots(figsize=(ps.COL_W, 2.2))
    for src, colour in (("mapillary", ps.MUTED[0]), ("google", ps.FOCAL)):
        sub = df[df["source"] == src]
        if sub.empty:
            continue
        ax.plot(sub["year"], sub["n"], marker="o", ms=3, lw=1.2,
                color=colour, label=src)
        last = sub.iloc[-1]
        ax.annotate(src, (last["year"], last["n"]), xytext=(3, 0),
                    textcoords="offset points", color=colour,
                    fontsize=6.5, va="center")

    ax.set_xlabel("Capture year")
    ax.set_ylabel("Records")
    ax.set_title(f"{campus_id.upper()} — capture activity by year", pad=6)
    ps.finish(ax)
    fig.tight_layout()

    if save:
        _save(fig, f"{campus_id}_temporal")
    return fig, ax


def _save(fig, stem: str) -> None:
    config.ensure_dirs()
    fig.savefig(config.FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(config.FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    print(f"  -> figures/{stem}.pdf, figures/{stem}.png")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_campus(campus_id: str, figures: bool = True) -> dict[str, Path]:
    """Write every table for one campus, optionally rendering its figures."""
    config.ensure_dirs()
    written: dict[str, Path] = {}

    for name, builder in (
        ("coverage_summary", coverage_summary),
        ("mapillary_profile", mapillary_profile),
        ("google_profile", google_profile),
        ("temporal_profile", temporal_profile),
    ):
        df = builder(campus_id)
        if df is None or df.empty:
            continue
        p = config.TABLE_DIR / f"{campus_id}_{name}.csv"
        df.to_csv(p, index=False)
        written[name] = p

    print(f"[analysis/{campus_id}] {len(written)} tables written")
    for p in written.values():
        print(f"  -> tables/{p.name}")

    if figures:
        plot_agreement_map(campus_id)
        plot_temporal(campus_id)
        plt.close("all")

    return written
