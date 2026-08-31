"""Figures.

One function per analysis in the plan. Each returns ``(fig, ax)`` and writes
PDF plus PNG when given ``save``.

Style rules that apply throughout: authored at final printed width, no grid,
direct labelling in preference to legends, one sequential ramp, and the
diverging ramp only where zero is a real midpoint.
"""

from __future__ import annotations

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from campus_svi import config, points, registry
from campus_svi.analysis import maps, metrics, paperstyle as ps
from campus_svi.analysis import style as st


# --------------------------------------------------------------------------
# 1. Coverage ratio by campus
# --------------------------------------------------------------------------

def fig_coverage(campus_ids, save="fig1_coverage", sort: bool = False):
    """Horizontal paired bars: Mapillary against Google, one row per campus.

    Horizontal because 40 campus names will not fit as rotated x labels. The
    left spine is dropped: the campus names *are* the axis.
    """
    ps.apply()
    df = metrics.coverage_table(campus_ids)
    # Canonical order by default so campuses hold position across the figure
    # set; sort=True re-sorts this one figure by coverage instead.
    df = df.set_index("campus_id").loc[registry.ordered(df["campus_id"])].reset_index()
    if sort:
        df = df.sort_values("either_coverage")

    n = len(df)
    fig, ax = plt.subplots(figsize=(ps.COL_W, max(2.4, 0.155 * n + 0.9)))
    y = np.arange(n)
    h = 0.38

    ax.barh(y + h / 2, df["mly_coverage"], height=h,
            color=st.SOURCE_COLORS["mapillary"], label="Mapillary")
    ax.barh(y - h / 2, df["ggl_coverage"], height=h,
            color=st.SOURCE_COLORS["google"], label="Google")

    ax.set_yticks(y)
    ax.set_yticklabels(registry.display_names(df["campus_id"]), fontsize=5.8)
    ax.set_xlabel("Proportion of grid cells with coverage")
    # Headroom so the direct labels are not clipped against the axis edge.
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-0.8, n - 0.2)

    # Direct labelling at the top pair rather than a legend.
    ax.text(df["mly_coverage"].iloc[-1] + 0.015, n - 1 + h / 2, "Mapillary",
            va="center", fontsize=6, color=st.SOURCE_COLORS["mapillary"])
    ax.text(df["ggl_coverage"].iloc[-1] + 0.015, n - 1 - h / 2, "Google",
            va="center", fontsize=6, color=st.SOURCE_COLORS["google"])

    ps.finish(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, ax


# --------------------------------------------------------------------------
# 2. Agreement — small multiples
# --------------------------------------------------------------------------

def fig_agreement_maps(campus_ids, save="fig2_agreement_maps", ncols=8,
                       sort_by_coverage=False, show_area=False):
    """Cell-level source agreement, one map face per campus.

    Panels are ordered by either-source coverage, so the figure reads as a
    gradient from best- to least-covered rather than alphabetically. Each
    panel is fitted to its own campus and carries its own scale bar.
    """
    order = None
    if sort_by_coverage:
        cov = metrics.coverage_table(campus_ids).sort_values(
            "either_coverage", ascending=False)
        order = cov["campus_id"].tolist()
    # Default: canonical order, so this map lines up with every other figure.
    return maps.small_multiples(
        campus_ids, column="agreement", kind="categorical",
        colors=st.AGREEMENT_COLORS, order=st.AGREEMENT_ORDER,
        labels=st.AGREEMENT_LABELS, ncols=ncols, sort_by=order,
        show_area=show_area, save=save)


def fig_agreement_composition(campus_ids, save="fig2b_agreement_composition"):
    """Stacked composition of agreement classes per campus."""
    ps.apply()
    df = metrics.coverage_table(campus_ids).sort_values("prop_both")

    n = len(df)
    fig, ax = plt.subplots(figsize=(ps.COL_W, max(2.4, 0.155 * n + 0.9)))
    y = np.arange(n)
    left = np.zeros(n)
    for cls in st.AGREEMENT_ORDER:
        v = df[f"prop_{cls}"].to_numpy(float)
        ax.barh(y, v, left=left, height=0.72,
                color=st.AGREEMENT_COLORS[cls],
                label=st.AGREEMENT_LABELS[cls])
        left += v

    ax.set_yticks(y)
    ax.set_yticklabels(registry.display_names(df["campus_id"]), fontsize=5.8)
    ax.set_xlabel("Proportion of grid cells")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, n - 0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              frameon=False, fontsize=6, handlelength=1.1, columnspacing=1.2)
    ps.finish(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, ax


# --------------------------------------------------------------------------
# 3. Depth decay — the enclosure test
# --------------------------------------------------------------------------

def fig_decay(campus_ids, save="fig3_decay", source="either", highlight=None):
    """Coverage against depth into campus, with the openness index beside it.

    Two panels, one argument. (a) is the evidence — every campus as a grey
    line, the mean as the focal series. (b) is what the slope of each of those
    lines amounts to, which is the openness index.

    Panel (b) is the one figure in the set sorted by its own value rather than
    canonical order: it is a ranking, and a ranking in arbitrary order is not a
    ranking. Everything else holds position so campuses stay comparable across
    figures.
    """
    ps.apply()
    prof = metrics.decay_profile(campus_ids, normalize=True)
    sl = metrics.decay_slope(campus_ids, source=source)
    if prof.empty:
        return None
    key = {"mly": "mly", "ggl": "ggl", "either": "either"}[source]

    height = max(3.2, 0.135 * max(len(sl), 1) + 1.0)
    fig, axes = plt.subplots(
        1, 2, figsize=(ps.FULL_W, height),
        gridspec_kw={"width_ratios": [1.15, 1.0]})

    # -- (a) decay curves --------------------------------------------------
    ax = axes[0]
    for _, g in prof.groupby("campus_id"):
        g = g.sort_values("bin_mid")
        ax.plot(g["bin_mid"], g[key], color="#cfcfcf", lw=0.55, zorder=1)
    for cid in (highlight or []):
        g = prof[prof["campus_id"] == cid].sort_values("bin_mid")
        if not g.empty:
            ax.plot(g["bin_mid"], g[key], color=registry.color(cid), lw=1.3,
                    zorder=3)
            ax.annotate(registry.display_name(cid),
                        (g["bin_mid"].iloc[-1], g[key].iloc[-1]),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=5.8, color=registry.color(cid), va="center")

    mean = prof.groupby("bin_mid")[key].mean().sort_index()
    ax.plot(mean.index, mean.values, color=ps.FOCAL, lw=1.9, zorder=5)
    ax.annotate("mean", (mean.index[-1], mean.values[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=6.5, color=ps.FOCAL,
                va="center", fontweight="bold")

    ax.set_xlabel("Normalised depth into campus\n(0 = perimeter, 1 = deepest point)")
    ax.set_ylabel("Proportion of cells covered")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ps.finish(ax)

    if not sl.empty:
        edge, core = sl["edge_coverage"].mean(), sl["core_coverage"].mean()
        neg = int((sl["openness_slope"] < 0).sum())
        # Percentage points: the difference of two shares is in pp, and
        # formatting it as a proportion rounds -0.04 away to "-0".
        lines = [
            f"n = {len(sl)} campuses",
            f"edge {edge:.0%} \u2192 core {core:.0%}  ({(core - edge) * 100:+.1f} pp)",
            f"{neg}/{len(sl)} campuses decay inward",
            f"median slope {sl['openness_slope'].median():+.3f}",
            f"median R\u00b2 {sl['r2'].median():.3f}",
        ]
        # Opaque backing: the line bundle is dense and unpredictable, so text
        # placed over it is unreadable on some datasets.
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=5.6, color="#3f3f3f",
                linespacing=1.6, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                          boxstyle="round,pad=0.35"))
    ps.panel_letter(fig, ax, "a")

    # -- (b) openness index, sorted by value -------------------------------
    ax2 = axes[1]
    if not sl.empty:
        d = sl.sort_values("openness_slope")
        y = np.arange(len(d))
        # Focal marks the enclosed campuses — the ones the argument is about.
        cols = [ps.FOCAL if v < 0 else "#c2c2c2" for v in d["openness_slope"]]
        ax2.barh(y, d["openness_slope"], height=0.72, color=cols)
        ax2.axvline(0, color="#2b2b2b", lw=0.6)
        ax2.set_yticks(y)
        ax2.set_yticklabels(registry.display_names(d["campus_id"]), fontsize=5.2)
        ax2.set_xlabel("Openness slope\n(negative = coverage decays inward)")
        ax2.set_ylim(-0.7, len(d) - 0.3)
        ps.finish(ax2)
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(axis="y", length=0)
    ps.panel_letter(fig, ax2, "b")

    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, axes


# --------------------------------------------------------------------------
# 4. Temporal depth
# --------------------------------------------------------------------------

def fig_temporal_depth(campus_ids, source: str = "google", ncols=8,
                       save=None, show_area=False):
    """Distinct capture years per cell, one source at a time.

    Kept as separate figures rather than a shared scale: the two sources reach
    very different depths — Google revisits a position across years while a
    Mapillary cell often holds a single capture date — and one colour scale
    across both would render Mapillary almost flat.
    """
    col = "mly_n_years" if source == "mapillary" else "ggl_n_years"
    name = "Mapillary" if source == "mapillary" else "Google"
    stem = save or f"fig4_{'mly' if source == 'mapillary' else 'ggl'}_depth"
    return maps.small_multiples(
        campus_ids, column=col, kind="continuous", cmap=st.CMAP,
        ncols=ncols, show_area=show_area,
        cbar_label=f"Distinct {name} capture years per cell", save=stem)


def fig_depth_diff(campus_ids, save="fig4b_depth_diff", ncols=8,
                   show_area=False):
    """Mapillary minus Google capture years, where both sources are present.

    Diverging ramp centred on zero, which here is a real midpoint: it separates
    cells where crowdsourcing offers deeper history from cells where it offers
    less. Cells covered by a single source are blank, since the comparison is
    undefined there.
    """
    ps.apply()
    prepared = maps.prepare(campus_ids)
    vals = np.concatenate([
        p[0]["depth_diff"].to_numpy(float) for p in prepared.values()
        if "depth_diff" in p[0].columns])
    vals = vals[np.isfinite(vals)]
    lim = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0

    return maps.small_multiples(
        campus_ids, column="depth_diff", kind="continuous", cmap=st.CMAP_DIV,
        vmin=-lim, vmax=lim, ncols=ncols, show_area=show_area,
        cbar_label="Mapillary \u2212 Google capture years", save=save)


# --------------------------------------------------------------------------
# 5. Temporal signature
# --------------------------------------------------------------------------

def fig_temporal_signature(campus_ids, save="fig5_temporal"):
    """Annual volume by source, monthly Mapillary series, and burstiness."""
    ps.apply()
    ann = metrics.temporal_annual(campus_ids)
    mon = metrics.monthly_mapillary(campus_ids)
    sig = metrics.temporal_signature(campus_ids)

    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.3))

    ax = axes[0]
    if not ann.empty:
        for src, colour in (("mapillary", st.SOURCE_COLORS["mapillary"]),
                            ("google", st.SOURCE_COLORS["google"])):
            s = ann[ann["source"] == src].groupby("year")["n"].sum().sort_index()
            if len(s):
                ax.plot(s.index, s.values, color=colour, lw=1.3, marker="o", ms=2.4)
                ax.annotate(st.SOURCE_LABELS[src], (s.index[-1], s.values[-1]),
                            xytext=(3, 0), textcoords="offset points",
                            fontsize=6, color=colour, va="center")
    ax.set_xlabel("Capture year")
    ax.set_ylabel("Records (all campuses)")
    ps.finish(ax)

    ax = axes[1]
    if not mon.empty:
        s = mon.groupby("year_month")["n"].sum().sort_index()
        ax.plot(range(len(s)), s.values, color=st.SOURCE_COLORS["mapillary"], lw=0.9)
        step = max(1, len(s) // 6)
        ax.set_xticks(range(0, len(s), step))
        ax.set_xticklabels([str(v) for v in s.index[::step]], fontsize=5.6)
        ps.rotate_xlabels(ax, 45)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mapillary images")
    ax.set_title("Contribution arrives in bursts", fontsize=7, pad=3)
    ps.finish(ax)

    ax = axes[2]
    if not sig.empty and "cv_monthly" in sig.columns:
        d = sig.dropna(subset=["cv_monthly"]).sort_values("cv_monthly")
        ax.scatter(d["cv_monthly"], d.get("top_creator_share", np.nan),
                   s=9, color=ps.MUTED[0], alpha=0.75, edgecolor="none")
        ax.set_xlabel("Burstiness (CV of monthly counts)")
        ax.set_ylabel("Top contributor's share")
        ps.finish(ax)

    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, axes


# --------------------------------------------------------------------------
# 6. Capture programme
# --------------------------------------------------------------------------

def fig_programme(campus_ids, save="fig6_programme"):
    """Google capture programme composition, stacked, sorted by trekker share.

    The trekker share is the point: that coverage is not road-snapped, so it
    measures how far the road-constraint caveat actually applies rather than
    assuming it applies everywhere.
    """
    ps.apply()
    df = metrics.programme_table(campus_ids)
    df = df[df["n_panoramas"] > 0]
    if df.empty:
        return None
    df = df.sort_values("scout", ascending=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=(ps.COL_W, max(2.4, 0.155 * n + 1.0)))
    y = np.arange(n)
    left = np.zeros(n)
    for cls in st.PROGRAMME_ORDER:
        if cls not in df.columns:
            continue
        v = df[cls].fillna(0).to_numpy(float)
        ax.barh(y, v, left=left, height=0.72, color=st.PROGRAMME_COLORS[cls],
                label=st.PROGRAMME_LABELS[cls])
        left += v

    ax.set_yticks(y)
    ax.set_yticklabels(registry.display_names(df["campus_id"]), fontsize=5.8)
    ax.set_xlabel("Proportion of Google panoramas")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, n - 0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=6, handlelength=1.1, columnspacing=1.2)
    ps.finish(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, ax


# --------------------------------------------------------------------------
# 7. MAUP sensitivity
# --------------------------------------------------------------------------

def fig_maup(campus_ids, sizes=(20, 50, 100), save="figS1_maup"):
    """Coverage at several cell sizes.

    Ratios rise with cell size by construction — a bigger cell is easier to
    intersect — so the question is whether the campus ranking and the
    between-source gap survive. Parallel lines mean they do.
    """
    ps.apply()
    sizes = tuple(sizes or config.MAUP_SIZES)
    df = metrics.maup_profile(campus_ids, sizes=sizes)
    if df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.4))

    for ax, key, name in zip(axes, ("mly_coverage", "ggl_coverage"),
                             ("Mapillary", "Google")):
        for cid, g in df.groupby("campus_id"):
            g = g.sort_values("cell_size_m")
            ax.plot(g["cell_size_m"], g[key], color="#c9c9c9", lw=0.6,
                    marker="o", ms=2)
        m = df.groupby("cell_size_m")[key].mean().sort_index()
        ax.plot(m.index, m.values, color=ps.FOCAL, lw=1.8, marker="o", ms=3.2)
        ax.set_xlabel("Cell size (m)")
        ax.set_ylabel(f"{name} coverage")
        ax.set_xticks(list(sizes))
        ax.set_ylim(0, 1)
        ps.finish(ax)

    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, axes


# --------------------------------------------------------------------------
# 8. Spatial autocorrelation
# --------------------------------------------------------------------------

def fig_autocorrelation(campus_ids, save="figS2_morans", permutations=499):
    """Moran's I per campus, with significance marked.

    Reported because adjacent cells are not independent: without this, any
    cell-level significance test in the paper is unsupported.
    """
    ps.apply()
    df = metrics.autocorrelation_table(campus_ids, permutations=permutations)
    if df.empty:
        return None
    d = df[df["column"] == "either_coverage"].sort_values("I")

    n = len(d)
    fig, ax = plt.subplots(figsize=(ps.COL_W, max(2.2, 0.15 * n + 0.9)))
    y = np.arange(n)
    sig = d["p_sim"] < 0.05
    ax.barh(y, d["I"], height=0.7,
            color=[ps.FOCAL if s else "#c9c9c9" for s in sig])
    ax.axvline(0, color="#2b2b2b", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(registry.display_names(d["campus_id"]), fontsize=5.6)
    ax.set_xlabel("Moran's I of either-source coverage")
    ax.text(0.98, 0.02, "filled: p < 0.05", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.8, color=ps.FOCAL)
    ps.finish(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, ax


# --------------------------------------------------------------------------
# 7. Image and panorama counts
# --------------------------------------------------------------------------

def fig_count_bars(campus_ids, save="fig7c_totals", width=None, height=None):
    """Total records per campus: Google left, Mapillary right.

    A back-to-back bar pair sharing one axis of campus names. The two sides
    have **independent scales**, printed on their own axes, because the sources
    differ in volume by an order of magnitude — forcing a common scale would
    render one side as a row of stubs and hide its internal variation.

    That independence is the trade-off to state plainly: bar *lengths* are not
    comparable across the centre line. What the figure supports is comparing
    campuses within a source, and seeing at a glance whether a campus is
    well-covered by one source and not the other.

    Canonical order, so rows line up with every other figure in the set.
    """
    ps.apply()
    ids = registry.ordered(campus_ids)
    cov = metrics.coverage_table(ids).set_index("campus_id").loc[ids].reset_index()

    n = len(cov)
    width = width or ps.FULL_W
    height = height or max(3.4, 0.185 * n + 1.0)
    # Wide gutter: the campus names live between the two panels, so the gap
    # has to fit the longest of them.
    # Gutter wide enough for the longest campus name at 5.4pt, centred.
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True,
                             gridspec_kw={"wspace": 0.42})
    y = np.arange(n)
    cols = registry.colors(cov["campus_id"])

    # -- left: Google, growing leftwards ----------------------------------
    axl = axes[0]
    axl.barh(y, cov["ggl_panoramas"], height=0.74, color=cols)
    axl.invert_xaxis()
    axl.set_xlabel("Google panoramas")
    for yy, v in zip(y, cov["ggl_panoramas"]):
        axl.text(v, yy, f"{int(v):,} ", va="center", ha="right", fontsize=4.6,
                 color="#6f6f6f")
    axl.set_xlim(cov["ggl_panoramas"].max() * 1.20, 0)

    # -- right: Mapillary, growing rightwards ------------------------------
    axr = axes[1]
    axr.barh(y, cov["mly_images"], height=0.74, color=cols)
    axr.set_xlabel("Mapillary images")
    for yy, v in zip(y, cov["mly_images"]):
        axr.text(v, yy, f" {int(v):,}", va="center", ha="left", fontsize=4.6,
                 color="#6f6f6f")
    axr.set_xlim(0, cov["mly_images"].max() * 1.20)

    # Campus names appear once, centred in the gutter between the panels.
    # Tick labels cannot do this: a tick label anchors to its own axis edge,
    # so long names drift across one panel while short ones sit far from the
    # other. Figure text placed at the true midpoint stays centred whatever
    # the name length.
    axl.set_yticks(y)
    axl.set_yticklabels([])
    axr.set_yticks(y)

    for ax in (axl, axr):
        ax.set_ylim(-0.7, n - 0.3)
        # Canonical order reads top-down, so IPB is the first row rather than
        # the last — barh would otherwise start numbering from the bottom.
        ax.invert_yaxis()
        ps.finish(ax)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    # Names are drawn after the layout is final, so the gutter midpoint and
    # the row positions are the ones actually used.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.075)
    pl, pr = axl.get_position(), axr.get_position()
    x_mid = (pl.x1 + pr.x0) / 2
    inv = fig.transFigure.inverted()
    for yy, cid in zip(y, cov["campus_id"]):
        y_fig = inv.transform(axl.transData.transform((0, yy)))[1]
        fig.text(x_mid, y_fig, registry.display_name(cid), ha="center",
                 va="center", fontsize=5.4, color="#2b2b2b")

    if save:
        maps.save_figure(fig, save)
    return fig, axes


def fig_count_maps(campus_ids, source: str = "mapillary", ncols=8,
                   save=None, show_area=False, log: bool = True):
    """Literal record counts per cell, one source at a time.

    Drawn on a log scale. Counts are heavily right-skewed — a few cells on a
    well-travelled path can hold dozens of images while most hold one or two —
    and a linear ramp would flatten that whole tail to the bottom colour.
    Cells with zero records are drawn as "no data" rather than as the darkest
    value, since zero has no position on a log axis.
    """
    col = "mly_count" if source == "mapillary" else "ggl_count"
    label = ("Mapillary images per cell" if source == "mapillary"
             else "Google panoramas per cell")
    tag = "mly" if source == "mapillary" else "ggl"
    stem = save or f"fig7_{tag}_counts_{'log' if log else 'raw'}"
    return maps.small_multiples(
        campus_ids, column=col, kind="continuous", cmap=st.CMAP, log=log,
        ncols=ncols, show_area=show_area,
        cbar_label=label + (" (log)" if log else ""), save=stem)


# --------------------------------------------------------------------------
# 8. Annual capture volume per campus
# --------------------------------------------------------------------------

def fig_annual_bars(campus_ids, source: str = "mapillary", ncols=8,
                    width=None, save=None, share_y: bool = False):
    """Records per capture year, one small bar chart per campus.

    ``share_y`` off by default: campuses differ in volume by orders of
    magnitude, and a shared axis would flatten the small ones into a flat line.
    Each panel is therefore about *timing*, not amount — the peak year is
    annotated so magnitude is still recoverable.
    """
    ps.apply()
    ids = registry.ordered([c for c in campus_ids])
    ann = metrics.temporal_annual(ids)
    if ann.empty:
        return None
    ann = ann[ann["source"] == source]
    if ann.empty:
        print(f"  ! no {source} temporal data")
        return None

    years = ann["year"].astype(int)
    y0, y1 = int(years.min()), int(years.max())
    ymax_all = ann.groupby("campus_id")["n"].max().max()

    width = width or ps.FULL_W
    n = len(ids)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    panel = width / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, nrows * panel * 0.85 + 0.7))
    axes = np.atleast_1d(axes).ravel()

    for k, cid in enumerate(ids):
        ax = axes[k]
        g = ann[ann["campus_id"] == cid]
        if not g.empty:
            # Institutional colour, so a campus is recognisable here and in
            # the study-area map without reading every label.
            ax.bar(g["year"].astype(int), g["n"], color=registry.color(cid),
                   width=0.78, linewidth=0)
            top = g.loc[g["n"].idxmax()]
            # Top-left in every panel: a fixed corner is scannable down a
            # column of 40, where a corner chosen per panel is not.
            ax.text(0.03, 0.97, f"max {int(top['n'])}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=4.6, color="#6f6f6f")
        ax.set_xlim(y0 - 0.7, y1 + 0.7)
        if share_y:
            ax.set_ylim(0, ymax_all * 1.05)
        ax.set_title(registry.display_name(cid), fontsize=5.4, pad=1.6)
        ax.tick_params(labelsize=4.4, length=1.5, pad=1)
        ax.set_xticks([y0, (y0 + y1) // 2, y1])
        ps.sparse_yticks(ax, n=3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.5)

    for ax in axes[n:]:
        ax.set_axis_off()

    name = "Mapillary images" if source == "mapillary" else "Google panoramas"
    fig.suptitle(f"{name} per capture year", fontsize=8, y=0.995)
    fig.supxlabel("Capture year", fontsize=6.5, y=0.012)
    fig.tight_layout(rect=[0, 0.02, 1, 0.975])

    stem = save or f"fig8_{'mly' if source == 'mapillary' else 'ggl'}_annual"
    maps.save_figure(fig, stem)
    return fig, axes


# --------------------------------------------------------------------------
# 9. Local Moran's I — cluster maps and quadrants
# --------------------------------------------------------------------------

def fig_calendar_heatmap(campus_ids, source: str = "mapillary", ncols=8,
                         width=None, save=None, cmap=None, log: bool = True,
                         vmax=None, cell_size: float = 0.16):
    """Year x month capture-activity heatmap, one panel per campus.

    Each panel is a small grid with months on the x axis and years on the y
    axis, cell color giving record count. This adds the within-year pattern
    that fig8's annual bars cannot show: a campus with steady bars can still
    turn out to be active only in a few months of each year, or only since a
    particular year, and the calendar view makes that visible at a glance
    across the full 40-campus set.

    Year and month labels appear only on the outer edge of the grid, the same
    convention as the other small-multiple figures in this module: the left
    column carries year labels, the bottom row carries month labels. Every
    panel still needs to be individually legible, so panels are drawn larger
    and spacing between them is kept tight, rather than adding labels to
    every panel and spreading them out to make room.

    Color is log-scaled by default (``log10(count + 1)``), since capture
    activity is as skewed month-to-month as it is cell-to-cell: a handful of
    month/year cells often account for most of a campus's total volume, and a
    linear scale would show a single bright cell against an otherwise flat
    grid. Months and years with zero records are drawn at the ramp's own
    floor color, not blanked out: for Mapillary in particular, an inactive
    month is itself part of the pattern, not a missing observation, so it is
    shown as a true zero rather than treated like a "no data" cell in the
    spatial maps.
    """
    ps.apply()
    cmap = cmap or st.CMAP
    ids = registry.ordered(campus_ids)

    layer = "mapillary" if source == "mapillary" else "google"
    name = "Mapillary" if source == "mapillary" else "Google"

    # Build one year x month count table per campus first, so the colour
    # scale can be set from the full set before any panel is drawn.
    tables, all_years = {}, set()
    for cid in ids:
        g = points.load_points(cid, layer)
        if g.empty or "year" not in g.columns:
            continue
        yr = pd.to_numeric(g["year"], errors="coerce")
        mo = (pd.to_numeric(g["month"], errors="coerce") if "month" in g.columns
              else pd.to_datetime(g.get("year_month"), errors="coerce",
                                  format="%Y-%m").dt.month)
        d = pd.DataFrame({"year": yr, "month": mo}).dropna()
        if d.empty:
            continue
        d["year"] = d["year"].astype(int)
        d["month"] = d["month"].astype(int)
        counts = d.groupby(["year", "month"]).size()
        tables[cid] = counts
        all_years.update(counts.index.get_level_values("year"))

    if not tables:
        print(f"  ! no {source} temporal data")
        return None

    # A common year axis across every panel: a campus with no records in a
    # given year still shows that year as an all-zero row.
    year_min, year_max = min(all_years), max(all_years)
    years = list(range(year_min, year_max + 1))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    grids = {}
    for cid in ids:                      # every requested campus, not just
        m = np.zeros((len(years), 12))   # ones with data, so a campus with
        if cid in tables:                # zero records still gets a blank
            for (yr, mo), n in tables[cid].items():   # (all-zero) panel
                m[years.index(yr), mo - 1] = n
        grids[cid] = m

    all_counts = np.concatenate([g.ravel() for g in grids.values()])
    if log:
        disp = {cid: np.log10(g + 1) for cid, g in grids.items()}
        vmax = vmax if vmax is not None else np.log10(all_counts.max() + 1)
        vmin = 0
        cbar_label = f"{name} records (log\u2081\u2080 scale)"
    else:
        disp = grids
        vmax = vmax if vmax is not None else all_counts.max()
        vmin = 0
        cbar_label = f"{name} records"

    n = len(ids)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    # Panels are sized to keep individual cells legible even though only the
    # edge panels carry axis labels; tight spacing keeps 40 of them from
    # spreading into an oversized canvas.
    panel_w = cell_size * 12 * 1.28
    panel_h = cell_size * len(years) * 1.22
    width = width or min(ncols * panel_w + 1.0, 16.0)
    fig_h = min(nrows * panel_h + 0.55, 22.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, fig_h))
    axes = np.atleast_1d(axes).ravel()

    im = None
    for k, cid in enumerate(ids):
        ax = axes[k]
        im = ax.pcolormesh(disp[cid], cmap=cmap, vmin=vmin, vmax=vmax,
                           edgecolors="white", linewidth=0.3)
        ax.set_title(registry.display_name(cid), fontsize=9.5, pad=2.6)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, len(years))
        ax.invert_yaxis()

        # Labels only on the outer edge: left column for years, bottom row
        # for months. Interior panels stay label-free, matching the other
        # small-multiple figures in this module.
        row, col = divmod(k, ncols)
        is_left = (col == 0)
        is_bottom = (row == nrows - 1) or (k + ncols >= n)

        if is_left:
            ax.set_yticks(np.arange(len(years)) + 0.5)
            ax.set_yticklabels([str(y) for y in years], fontsize=7.0)
        else:
            ax.set_yticks([])
        if is_bottom:
            ax.set_xticks(np.arange(12) + 0.5)
            ax.set_xticklabels(month_labels, fontsize=7.0, rotation=90)
        else:
            ax.set_xticks([])

        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0, pad=1)

    for ax in axes[n:]:
        ax.set_axis_off()

    # No figure-level title: the colorbar label and panel titles already
    # carry what a suptitle would say, and removing it lets the panel grid
    # use the vertical space instead.
    # Tight spacing between panels: gaps are kept small since no interior
    # panel needs room for its own labels. top is pushed close to 1.0 now
    # that there is no title reserving space above the grid, and bottom is
    # pulled in so the colorbar sits close under the last row rather than
    # floating in a wide empty band.
    fig.subplots_adjust(left=0.055, right=0.99, top=0.97, bottom=0.075,
                        wspace=0.10, hspace=0.18)

    cax = fig.add_axes([0.30, 0.028, 0.40, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=9, labelpad=4)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.outline.set_visible(False)

    stem = save or f"fig8b_{'mly' if source == 'mapillary' else 'ggl'}_calendar"
    maps.save_figure(fig, stem)
    return fig, axes


def fig_local_moran(campus_ids, source: str = "mapillary", ncols=8,
                    permutations: int = 199, save=None, show_area=False):
    """Local Moran's I per cell, one map face per campus.

    Global Moran's I says a campus is clustered; it cannot say where. This
    decomposes it, so the value is per cell and the diverging ramp is centred
    on zero — positive where a cell resembles its neighbours, negative where it
    contradicts them.
    """
    col = "mly_count" if source == "mapillary" else "ggl_count"
    prefix = "mly" if source == "mapillary" else "ggl"
    aug = metrics.local_morans_augmenter(col, prefix, permutations=permutations)

    prepared = maps.prepare(campus_ids, augment=aug)
    vals = np.concatenate([p[0][f"{prefix}_Ii"].to_numpy(float)
                           for p in prepared.values()
                           if f"{prefix}_Ii" in p[0].columns])
    vals = vals[np.isfinite(vals)]
    # Robust limits: a handful of extreme cells would otherwise wash out the
    # ramp for everything else.
    lim = float(np.nanpercentile(np.abs(vals), 98)) if len(vals) else 1.0

    name = "Mapillary" if source == "mapillary" else "Google"
    stem = save or f"fig9_{prefix}_local_moran"
    # prepared is passed back so the permutation test is not paid for twice.
    return maps.small_multiples(
        campus_ids, column=f"{prefix}_Ii", kind="continuous",
        cmap=st.CMAP_DIV, vmin=-lim, vmax=lim, ncols=ncols,
        prepared=prepared, show_area=show_area,
        cbar_label=f"Local Moran's I, {name} density", save=stem)


def fig_moran_quadrants(campus_ids, source: str = "mapillary", ncols=8,
                        permutations: int = 199, save=None, show_area=False):
    """LISA cluster quadrants, one map face per campus.

    Where the value map shows strength, this shows kind. High-High is a
    coverage cluster, Low-Low a coverage gap — and for this project the Low-Low
    clusters are the finding: contiguous interior areas that no source reaches.
    Cells failing the permutation test are drawn near-white and mean no signal,
    not a weak one.
    """
    col = "mly_count" if source == "mapillary" else "ggl_count"
    prefix = "mly" if source == "mapillary" else "ggl"
    aug = metrics.local_morans_augmenter(col, prefix, permutations=permutations)
    stem = save or f"fig9b_{prefix}_moran_quadrants"
    return maps.small_multiples(
        campus_ids, column=f"{prefix}_quadrant", kind="categorical",
        colors=st.QUADRANT_COLORS, order=st.QUADRANT_ORDER,
        labels=st.QUADRANT_LABELS, ncols=ncols, augment=aug,
        show_area=show_area, save=stem)


# --------------------------------------------------------------------------
# 10. Contributors
# --------------------------------------------------------------------------

def fig_contributors(campus_ids, width=None, save="fig10_contributors",
                     log: bool = True, jitter: float = 0.0,
                     height: float = 6.6, dot_size: float = 16,
                     dot_edge: float = 0.35, dot_edge_color: str = "white",
                     dot_alpha: float = 0.9, gini_fontsize: float = 7.5):
    """Who mapped each campus, in three panels on one campus axis.

    (a) how many contributors. (b) one dot per contributor at the number of
    images they uploaded. (c) how unequally those images are distributed.

    The three answer questions the others cannot. (a) alone would call a campus
    well-contributed; (b) shows whether that is twenty people sharing the work
    or one carrying it; (c) reduces (b) to a single comparable number.

    (c) stays in canonical order rather than sorted by value, because it shares
    the axis with the panels above it — a sorted third panel would break the
    column alignment that makes the figure readable. The ranked view of the
    same quantity is in the summary table.

    Dot styling is exposed because the right size depends on how many
    contributors a campus has: ``dot_size`` and ``dot_edge`` are worth tuning
    once you see your own data. ``gini_fontsize`` governs panel (c)'s numbers,
    which sit in a short panel and need to be set larger than the default to
    stay readable.
    """
    ps.apply()
    ids = registry.ordered(campus_ids)
    prof = metrics.contributor_profile(ids)
    if prof.empty:
        print("  ! no contributor data")
        return None
    summ = metrics.contributor_summary(ids).set_index("campus_id")

    # Wider than the text block: 40 categories with rotated labels need the
    # room. Crop or scale at layout time.
    width = width or ps.FULL_W * 1.55
    fig, axes = plt.subplots(3, 1, figsize=(width, height), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.5, 0.9]})
    x = np.arange(len(ids))
    cols = registry.colors(ids)

    # -- (a) how many contributors ----------------------------------------
    ax = axes[0]
    counts = [int(summ.loc[c, "n_contributors"]) if c in summ.index else 0
              for c in ids]
    ax.bar(x, counts, color=cols, width=0.74, linewidth=0)
    ax.set_ylabel("Contributors")
    ax.set_xlim(-0.8, len(ids) - 0.2)
    # Contributors are people: integer ticks, not 15.0.
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    ps.finish(ax)
    ps.panel_letter(fig, ax, "a", dx=-0.045, dy=1.05)

    # -- (b) how much each contributed ------------------------------------
    ax2 = axes[1]
    rng = np.random.default_rng(0)
    for i, cid in enumerate(ids):
        g = prof[prof["campus_id"] == cid]
        if g.empty:
            continue
        # Dots sit on the campus tick by default, so each campus reads as one
        # clean vertical series and heights compare straight across.
        xs = (np.full(len(g), float(i)) if not jitter
              else i + rng.uniform(-jitter, jitter, len(g)))
        ax2.scatter(xs, g["n_images"], s=dot_size, color=registry.color(cid),
                    alpha=dot_alpha, edgecolor=dot_edge_color,
                    linewidth=dot_edge, zorder=3)

    if log:
        ax2.set_yscale("log")
        ax2.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax2.yaxis.set_minor_locator(mticker.NullLocator())
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):d}" if v >= 1 else ""))
    ax2.set_ylabel("Images per contributor")
    ps.finish(ax2)
    ps.panel_letter(fig, ax2, "b", dx=-0.045, dy=1.03)

    # -- (c) how unequally ------------------------------------------------
    ax3 = axes[2]
    gini = [summ.loc[c, "gini"] if c in summ.index else np.nan for c in ids]
    ax3.bar(x, gini, color=cols, width=0.74, linewidth=0)
    ax3.set_ylabel("Gini")
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0, 0.5, 1.0])
    # This panel is short, so its tick labels come out smaller than the ones
    # above it unless set explicitly. Readable numbers matter more here than
    # matching the other panels exactly: the Gini value is the point.
    ax3.tick_params(axis="y", labelsize=gini_fontsize)
    ax3.set_xticks(x)
    ax3.set_xticklabels(registry.display_names(ids), fontsize=5.0)
    ps.rotate_xlabels(ax3, 90)
    ps.finish(ax3)
    ps.panel_letter(fig, ax3, "c", dx=-0.045, dy=1.06)
    ax3.text(0.995, 0.94, "0 = evenly shared \u00b7 1 = one person",
             transform=ax3.transAxes, ha="right", va="top",
             fontsize=gini_fontsize, color="#6f6f6f")

    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, axes


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def build_all(campus_ids, ncols=8, maup_sizes=None, show_area=False,
              skip_maup=False, skip_morans=False):
    """Every figure in the plan, in order. Returns {name: (fig, ax)}."""
    out = {}
    steps = [
        ("fig1_coverage", lambda: fig_coverage(campus_ids)),
        ("fig2_agreement_maps", lambda: fig_agreement_maps(campus_ids, ncols=ncols, show_area=show_area)),
        ("fig2b_agreement_composition", lambda: fig_agreement_composition(campus_ids)),
        ("fig3_decay", lambda: fig_decay(campus_ids)),
        ("fig4_ggl_depth", lambda: fig_temporal_depth(campus_ids, "google", ncols, show_area=show_area)),
        ("fig4_mly_depth", lambda: fig_temporal_depth(campus_ids, "mapillary", ncols, show_area=show_area)),
        ("fig4b_depth_diff", lambda: fig_depth_diff(campus_ids, ncols=ncols, show_area=show_area)),
        ("fig5_temporal", lambda: fig_temporal_signature(campus_ids)),
        ("fig7_mly_counts_log", lambda: fig_count_maps(campus_ids, "mapillary", ncols, show_area=show_area, log=True)),
        ("fig7_mly_counts_raw", lambda: fig_count_maps(campus_ids, "mapillary", ncols, show_area=show_area, log=False)),
        ("fig7_ggl_counts_log", lambda: fig_count_maps(campus_ids, "google", ncols, show_area=show_area, log=True)),
        ("fig7_ggl_counts_raw", lambda: fig_count_maps(campus_ids, "google", ncols, show_area=show_area, log=False)),
        ("fig7c_totals", lambda: fig_count_bars(campus_ids)),
        ("fig8_mly_annual", lambda: fig_annual_bars(campus_ids, "mapillary", ncols)),
        ("fig8_ggl_annual", lambda: fig_annual_bars(campus_ids, "google", ncols)),
        ("fig9_mly_local_moran", lambda: fig_local_moran(campus_ids, "mapillary", ncols, show_area=show_area)),
        ("fig9_ggl_local_moran", lambda: fig_local_moran(campus_ids, "google", ncols, show_area=show_area)),
        ("fig9b_mly_quadrants", lambda: fig_moran_quadrants(campus_ids, "mapillary", ncols, show_area=show_area)),
        ("fig9b_ggl_quadrants", lambda: fig_moran_quadrants(campus_ids, "google", ncols, show_area=show_area)),
        ("fig10_contributors", lambda: fig_contributors(campus_ids)),
    ]
    if not skip_maup:
        steps.append(("figS1_maup",
                      lambda: fig_maup(campus_ids,
                                       sizes=maup_sizes or config.MAUP_SIZES)))
    if not skip_morans:
        steps.append(("figS2_morans", lambda: fig_autocorrelation(campus_ids)))

    for name, fn in steps:
        print(f"[{name}]")
        try:
            out[name] = fn()
        except Exception as exc:                              # noqa: BLE001
            print(f"  !! failed: {type(exc).__name__}: {exc}")
        plt.close("all")
    return out
