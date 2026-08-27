"""Figures.

One function per analysis in the plan. Each returns ``(fig, ax)`` and writes
PDF plus PNG when given ``save``.

Style rules that apply throughout: authored at final printed width, no grid,
direct labelling in preference to legends, one sequential ramp, and the
diverging ramp only where zero is a real midpoint.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from campus_svi import config, registry
from campus_svi.analysis import maps, metrics, paperstyle as ps
from campus_svi.analysis import style as st


# --------------------------------------------------------------------------
# 1. Coverage ratio by campus
# --------------------------------------------------------------------------

def fig_coverage(campus_ids, save="fig1_coverage", sort: bool = True):
    """Horizontal paired bars: Mapillary against Google, one row per campus.

    Horizontal because 40 campus names will not fit as rotated x labels. The
    left spine is dropped: the campus names *are* the axis.
    """
    ps.apply()
    df = metrics.coverage_table(campus_ids)
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
                       sort_by_coverage=True, show_area=False):
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
    """Coverage against normalised depth into the campus.

    Every campus is a grey line; the mean is the focal series. With 40 lines,
    individual identity is not readable and should not pretend to be — the
    spread is the message, and ``highlight`` traces the few worth naming.
    """
    ps.apply()
    prof = metrics.decay_profile(campus_ids, normalize=True)
    if prof.empty:
        return None
    key = {"mly": "mly", "ggl": "ggl", "either": "either"}[source]

    fig, ax = plt.subplots(figsize=(ps.HALF_W, 2.4))
    for _, g in prof.groupby("campus_id"):
        g = g.sort_values("bin_mid")
        ax.plot(g["bin_mid"], g[key], color="#cfcfcf", lw=0.55, zorder=1)
    for cid in (highlight or []):
        g = prof[prof["campus_id"] == cid].sort_values("bin_mid")
        if not g.empty:
            ax.plot(g["bin_mid"], g[key], color=ps.MUTED[1], lw=1.2, zorder=3)
            ax.annotate(registry.display_name(cid),
                        (g["bin_mid"].iloc[-1], g[key].iloc[-1]),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=5.8, color=ps.MUTED[1], va="center")

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
    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, ax


def fig_openness(campus_ids, save="fig3b_openness", source="either"):
    """The decay slope per campus — the road-free openness index.

    Given its own figure because 40 campus labels need vertical room that a
    side panel cannot supply without colliding.
    """
    ps.apply()
    sl = metrics.decay_slope(campus_ids, source=source)
    if sl.empty:
        return None
    sl = sl.sort_values("openness_slope")

    n = len(sl)
    fig, ax = plt.subplots(figsize=(ps.HALF_W, max(2.4, 0.15 * n + 0.9)))
    y = np.arange(n)
    # Focal marks the enclosed campuses — the ones the argument is about.
    cols = [ps.FOCAL if v < 0 else "#c2c2c2" for v in sl["openness_slope"]]
    ax.barh(y, sl["openness_slope"], height=0.7, color=cols)
    ax.axvline(0, color="#2b2b2b", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(registry.display_names(sl["campus_id"]), fontsize=5.8)
    ax.set_xlabel("Openness slope\n(negative = coverage decays inward)")
    ax.set_ylim(-0.7, n - 0.3)
    ps.finish(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    if save:
        maps.save_figure(fig, save)
    return fig, ax


# --------------------------------------------------------------------------
# 4. Temporal depth
# --------------------------------------------------------------------------

def fig_temporal_depth(campus_ids, save="fig4_temporal_depth", ncols=8,
                       show_area=False):
    """Distinct Google capture years per cell, as small multiples."""
    return maps.small_multiples(
        campus_ids, column="ggl_n_years", kind="continuous", cmap=st.CMAP,
        ncols=ncols, show_area=show_area,
        cbar_label="Distinct Google capture years per cell", save=save)


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
        ("fig3b_openness", lambda: fig_openness(campus_ids)),
        ("fig4_temporal_depth", lambda: fig_temporal_depth(campus_ids, ncols=ncols, show_area=show_area)),
        ("fig4b_depth_diff", lambda: fig_depth_diff(campus_ids, ncols=ncols, show_area=show_area)),
        ("fig5_temporal", lambda: fig_temporal_signature(campus_ids)),
        ("fig6_programme", lambda: fig_programme(campus_ids)),
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
