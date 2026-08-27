"""Study area figures.

Two maps, kept in their own module because they are the only figures that
touch a basemap and the only ones drawn in geographic rather than grid space.

**Overview.** Indonesia at national extent — which is strongly horizontal, so
the figure is wide and short. Campus centroids in institutional colour, with
labels pushed out to the margins and joined by leader lines. Labels are stacked
along the left and right edges sorted by latitude rather than placed next to
their point: at 40 campuses, several of them minutes apart in Java, any
place-beside-the-point scheme collides. Stacking guarantees no overlap by
construction and the leader lines carry the association.

Basemap is deliberately plain. A satellite background at national extent adds
texture that competes with forty coloured points and tells the reader nothing
about where imagery exists.

**Per-campus panels.** Satellite imagery, with everything outside the boundary
covered by a translucent scrim and the interior left clear. That inverts the
usual highlight-the-study-area convention for a reason: the argument is about
what is inside the fence, so the inside is what stays legible while context
remains visible enough to place it.

Imagery needs ``contextily`` and network access. Without either, panels fall
back to a plain fill and say so rather than failing.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from campus_svi import boundaries, registry
from campus_svi.analysis import maps, paperstyle as ps

#: Indonesia, trimmed to the archipelago. Wider than tall by roughly 3:1,
#: which is what makes a landscape figure the honest shape for this map.
INDONESIA_BOUNDS = (94.5, -11.5, 141.5, 6.5)     # west, south, east, north

WEB_MERCATOR = "EPSG:3857"


def _has_contextily():
    try:
        import contextily  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Overview map
# --------------------------------------------------------------------------

def _label_slots(n: int, y0: float, y1: float):
    """Evenly spaced vertical slots for stacked labels."""
    if n == 1:
        return np.array([(y0 + y1) / 2])
    return np.linspace(y1, y0, n)


def fig_overview(campus_ids, save="fig0_study_area", width=None,
                 height: float = None, bounds=None, label_pad: float = 0.16,
                 point_size: float = 20, basemap: bool = False,
                 min_label_gap: float = 0.115):
    """National overview: campus centroids with leader-line labels.

    ``label_pad`` is the share of the axis width reserved for labels on each
    side. Raise it if names are clipped; lower it to give the map more room.
    """
    ps.apply()
    width = width or ps.FULL_W
    w, s, e, n = bounds or INDONESIA_BOUNDS

    reg = registry.build(campus_ids, include_cells=False)
    reg = reg.dropna(subset=["centroid_lat", "centroid_lon"])
    if reg.empty:
        raise ValueError("No campus centroids available.")
    reg = reg.set_index("campus_id").loc[
        registry.ordered(reg["campus_id"])].reset_index()

    # Height is driven by the label column, not the map: with 40 names split
    # across two columns, the taller constraint is fitting ~20 lines legibly.
    per_side = math.ceil(len(reg) / 2)
    height = height or max(3.4, per_side * min_label_gap + 0.6)
    fig, ax = plt.subplots(figsize=(width, height))

    # Draw the map in the middle, with margins reserved for label columns.
    span = e - w
    ax.set_xlim(w - span * label_pad, e + span * label_pad)
    ax.set_ylim(s, n)

    if basemap and _has_contextily():
        try:
            import contextily as cx
            cx.add_basemap(ax, crs="EPSG:4326",
                           source=cx.providers.CartoDB.PositronNoLabels,
                           attribution_size=4)
        except Exception as exc:                              # noqa: BLE001
            print(f"  ! basemap unavailable: {type(exc).__name__}")

    # Split into two label columns by longitude *rank*, not by the midpoint
    # of the map. Indonesian universities cluster heavily in Java, so a
    # midpoint split puts nearly all 40 labels in one column and leaves the
    # other empty. Ranking balances the columns whatever the distribution.
    ordered_lon = reg.sort_values("centroid_lon")
    half_n = math.ceil(len(ordered_lon) / 2)
    left = ordered_lon.iloc[:half_n].sort_values("centroid_lat")
    right = ordered_lon.iloc[half_n:].sort_values("centroid_lat")

    x_left = w - span * label_pad * 0.92
    x_right = e + span * label_pad * 0.92

    for side, df, xlab, ha in (("l", left, x_left, "right"),
                               ("r", right, x_right, "left")):
        slots = _label_slots(len(df), s + (n - s) * 0.03, n - (n - s) * 0.03)
        for (_, row), ys in zip(df.iterrows(), slots[::-1]):
            colr = registry.color(row["campus_id"])
            ax.plot([row["centroid_lon"], xlab], [row["centroid_lat"], ys],
                    color=colr, lw=0.35, alpha=0.55, zorder=2,
                    solid_capstyle="round")
            ax.text(xlab + (-0.25 if ha == "right" else 0.25), ys,
                    registry.display_name(row["campus_id"]),
                    ha=ha, va="center", fontsize=4.8, color="#2b2b2b",
                    zorder=4)

    ax.scatter(reg["centroid_lon"], reg["centroid_lat"],
               s=point_size, c=registry.colors(reg["campus_id"]),
               edgecolor="white", linewidth=0.4, zorder=5)

    ax.set_aspect("equal")
    ax.set_axis_off()

    # Latitude/longitude are meaningless as decoration here; a scale bar is
    # not, so the reader can judge distance between sites.
    _geographic_scale_bar(ax, w, e, s, n)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    if save:
        maps.save_figure(fig, save)
    return fig, ax


def _geographic_scale_bar(ax, w, e, s, n, y_frac: float = 0.06):
    """Approximate scale bar for a lat/lon axis, at the map's mean latitude."""
    lat_mid = (s + n) / 2
    km_per_deg = 111.32 * math.cos(math.radians(lat_mid))
    for bar_km in (2000, 1000, 500):
        deg = bar_km / km_per_deg
        if deg < (e - w) * 0.35:
            break
    x0 = w + (e - w) * 0.02
    y = s + (n - s) * y_frac
    ax.plot([x0, x0 + deg], [y, y], color="#2b2b2b", lw=1.0,
            solid_capstyle="butt", zorder=6)
    for xx in (x0, x0 + deg):
        ax.plot([xx, xx], [y - (n - s) * 0.012, y + (n - s) * 0.012],
                color="#2b2b2b", lw=1.0, zorder=6)
    ax.text(x0 + deg / 2, y + (n - s) * 0.02, f"{bar_km:g} km",
            ha="center", va="bottom", fontsize=5.2, color="#2b2b2b", zorder=6)


# --------------------------------------------------------------------------
# Per-campus imagery panels
# --------------------------------------------------------------------------

def fig_campus_panels(campus_ids, save="fig0b_campus_boundaries", ncols=8,
                      width=None, imagery: bool = True, scrim: float = 0.62,
                      scrim_color: str = "white", margin: float = 1.22,
                      zoom: int = None):
    """Boundary of each campus over satellite imagery.

    Everything outside the boundary is covered by a translucent scrim and the
    interior left clear, so the campus reads as a window onto the imagery. The
    outline is in institutional colour, matching the overview map.

    ``scrim`` is the opacity of the mask: higher isolates the campus more
    firmly, lower keeps more surrounding context. ``margin`` sets how much
    context is in frame at all.

    Each panel is fitted to its own campus, so scale differs between them and
    each carries its own bar — the same contract as the analysis maps.
    """
    from shapely.geometry import box as shapely_box

    ps.apply()
    width = width or ps.FULL_W
    ids = registry.ordered(campus_ids)

    use_imagery = imagery and _has_contextily()
    if imagery and not use_imagery:
        print("  ! contextily not installed — panels drawn without imagery.")
        print("    pip install contextily")

    n = len(ids)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    panel = width / ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width, nrows * panel * 1.24 + 0.5))
    axes = np.atleast_1d(axes).ravel()
    imagery_ok = False          # only claim imagery in the caption if it loaded

    for k, cid in enumerate(ids):
        ax = axes[k]
        try:
            bnd = boundaries.load(cid).to_crs(WEB_MERCATOR)
        except Exception as exc:                              # noqa: BLE001
            ax.set_axis_off()
            print(f"  ! {cid}: {type(exc).__name__}")
            continue

        poly = bnd.geometry.iloc[0]
        minx, miny, maxx, maxy = poly.bounds
        cx_, cy_ = (minx + maxx) / 2, (miny + maxy) / 2
        half = max(maxx - minx, maxy - miny) / 2 * margin
        x0, x1 = cx_ - half, cx_ + half
        y0, y1 = cy_ - half, cy_ + half

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")

        if use_imagery:
            try:
                import contextily as cx
                # contextily wants "auto" or an int; None is rejected.
                kw = {"zoom": zoom} if zoom is not None else {"zoom": "auto"}
                cx.add_basemap(ax, crs=WEB_MERCATOR,
                               source=cx.providers.Esri.WorldImagery,
                               attribution=False, **kw)
                imagery_ok = True
            except Exception as exc:                          # noqa: BLE001
                ax.set_facecolor("#e9e9e9")
                if k == 0:
                    print(f"  ! imagery fetch failed: {type(exc).__name__}: {exc}")
        else:
            ax.set_facecolor("#ededed")

        # The scrim is the panel rectangle minus the campus: one polygon with
        # a hole, so the interior is genuinely untouched rather than covered
        # by a second, lighter layer.
        frame = shapely_box(x0, y0, x1, y1)
        mask = frame.difference(poly)
        if not mask.is_empty:
            import geopandas as gpd
            gpd.GeoSeries([mask], crs=WEB_MERCATOR).plot(
                ax=ax, color=scrim_color, alpha=scrim, linewidth=0, zorder=3)

        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color=registry.color(cid), lw=0.9, zorder=4)

        ax.set_axis_off()
        ax.set_title(registry.display_name(cid), fontsize=5.6, pad=1.8,
                     color="#2b2b2b")
        maps.panel_scale_bar(ax, half)

    for ax in axes[n:]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.005, right=0.995, top=0.972, bottom=0.03,
                        wspace=0.06, hspace=0.30)
    # Report what actually rendered, not what was requested: a caption
    # crediting imagery that failed to load would be simply false.
    note = ("Esri World Imagery" if imagery_ok else "imagery unavailable")
    fig.text(0.995, 0.006,
             f"Each panel at its own scale \u00b7 {note}",
             ha="right", va="bottom", fontsize=5.0, color="#6f6f6f",
             style="italic")

    if save:
        maps.save_figure(fig, save)
    return fig, axes


def fig_study_area(campus_ids, ncols=8, imagery: bool = True):
    """Both study-area figures, in order."""
    out = {}
    print("[fig0_study_area]")
    out["overview"] = fig_overview(campus_ids)
    print("[fig0b_campus_boundaries]")
    out["panels"] = fig_campus_panels(campus_ids, ncols=ncols, imagery=imagery)
    return out
