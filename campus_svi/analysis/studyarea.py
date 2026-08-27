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
import matplotlib.patheffects as pe

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

#: Basemap providers, tried in order. CartoDB is deliberately absent: its
#: tiles now return an "API KEY REQUIRED" watermark, and xyzservices still
#: reports requires_token() as False, so the metadata cannot be trusted here.
#: Esri and OpenStreetMap serve these layers without a key.
PLAIN_PROVIDERS = ["Esri.WorldGrayCanvas", "Esri.WorldTopoMap",
                   "OpenStreetMap.Mapnik"]
IMAGERY_PROVIDERS = ["Esri.WorldImagery"]


def _provider(path: str):
    import xyzservices.providers as xp
    obj = xp
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _add_basemap(ax, crs, providers, zoom=None):
    """Try each provider until one renders. Returns the name that worked."""
    import contextily as cx

    for name in providers:
        try:
            kw = {"zoom": zoom} if zoom is not None else {"zoom": "auto"}
            cx.add_basemap(ax, crs=crs, source=_provider(name),
                           attribution=False, **kw)
            return name
        except Exception:                                     # noqa: BLE001
            continue
    return None


def _text_size_data(ax, text, fontsize):
    """Approximate label width and height in data units.

    Rendered extents need a draw pass, which is expensive inside a placement
    loop. A character-count estimate is close enough to keep labels apart,
    and erring slightly large is the safe direction.
    """
    fig = ax.figure
    pos = ax.get_position()
    w_pt = pos.width * fig.get_figwidth() * 72
    h_pt = pos.height * fig.get_figheight() * 72
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    per_x = (x1 - x0) / max(w_pt, 1)
    per_y = (y1 - y0) / max(h_pt, 1)
    return (len(text) * fontsize * 0.52 * per_x,
            fontsize * 1.18 * per_y, per_x, per_y)


def _overlaps(a, b, pad_x=0.0, pad_y=0.0):
    return not (a[2] + pad_x < b[0] or b[2] + pad_x < a[0] or
                a[3] + pad_y < b[1] or b[3] + pad_y < a[1])


def fig_overview(campus_ids, save="fig0_study_area", width=None,
                 height: float = None, bounds=None, point_size: float = 20,
                 basemap: bool = True, fontsize: float = 5.2,
                 leader_min_pt: float = 7.0, pad_frac: float = 0.10,
                 label_pad_pt: float = 2.6, zoom_to_points: bool = True):
    """National overview: campus centroids labelled in place.

    Labels sit **inside** the map, placed by a greedy search: for each campus
    the candidate positions around its point are tried nearest-first, and the
    first that collides with no already-placed label is taken. A leader line is
    drawn only when the label ended up far enough away to need one, so the
    uncrowded campuses in Sumatra and Sulawesi get a clean label against their
    point while the Java cluster fans outward.

    This is better than stacking labels in the margins, which guarantees no
    overlap but spends half the figure width on a column of names and draws
    forty long leader lines across the map.

    The basemap is plain rather than satellite: texture at national extent
    competes with forty coloured points and says nothing about where imagery
    exists.
    """
    ps.apply()
    width = width or ps.FULL_W

    reg = registry.build(campus_ids, include_cells=False)
    reg = reg.dropna(subset=["centroid_lat", "centroid_lon"])
    if reg.empty:
        raise ValueError("No campus centroids available.")
    reg = reg.set_index("campus_id").loc[
        registry.ordered(reg["campus_id"])].reset_index()

    if bounds is not None:
        w, s, e, n = bounds
    elif zoom_to_points:
        # Frame the campuses, not the country. Most of Indonesia's east is
        # empty of study sites, and showing it spends the figure's width on
        # sea while the Java cluster stays illegible.
        lon0, lon1 = reg["centroid_lon"].min(), reg["centroid_lon"].max()
        lat0, lat1 = reg["centroid_lat"].min(), reg["centroid_lat"].max()
        dx = max(lon1 - lon0, 1.0) * pad_frac
        dy = max(lat1 - lat0, 1.0) * pad_frac
        # Extra room on the sides for labels, which extend horizontally.
        w, e = lon0 - dx * 2.2, lon1 + dx * 2.2
        s, n = lat0 - dy * 1.4, lat1 + dy * 1.6
    else:
        w, s, e, n = INDONESIA_BOUNDS

    # Height follows the framed aspect ratio, so the map is not stretched.
    aspect = (n - s) / max(e - w, 1e-9)
    height = height or float(np.clip(width * aspect * 1.05, 2.6, 8.0))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)

    used = None
    if basemap and _has_contextily():
        used = _add_basemap(ax, "EPSG:4326", PLAIN_PROVIDERS)
        if used is None:
            print("  ! no basemap provider reachable — drawing without one")

    ax.set_xlim(w, e)
    ax.set_ylim(s, n)

    # Points first: labels are placed around them and must not cover them.
    ax.scatter(reg["centroid_lon"], reg["centroid_lat"], s=point_size,
               c=registry.colors(reg["campus_id"]), edgecolor="white",
               linewidth=0.4, zorder=6)

    _, _, per_x, per_y = _text_size_data(ax, "M", fontsize)
    placed: list[tuple] = []
    marker_r = (point_size ** 0.5) * 0.6

    # Candidate rings, nearest first: a label close to its point needs no
    # leader line at all.
    # Horizontal placements first: a label to the left or right of its point
    # reads more cleanly than one above or below, and leaves vertical room for
    # the next campus in a dense cluster.
    radii = [7, 11, 16, 22, 30, 40, 52, 66, 82]
    angles = np.deg2rad([0, 180, 20, -20, 45, -45, 70, -70, 90, -90,
                         135, -135, 160, -160])

    # Densest areas first. A campus in the Java cluster has the fewest viable
    # slots, so it should choose before an isolated one takes a nearby space.
    pts = reg[["centroid_lon", "centroid_lat"]].to_numpy(float)
    dens = []
    for i, p in enumerate(pts):
        d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
        dens.append((np.sort(d)[1:6]).sum())
    order = np.argsort(dens)

    for idx in order:
        row = reg.iloc[idx]
        cid = row["campus_id"]
        label = registry.display_name(cid)
        lw, lh, _, _ = _text_size_data(ax, label, fontsize)
        px, py = float(row["centroid_lon"]), float(row["centroid_lat"])

        best = None
        for r_pt in radii:
            for a in angles:
                dx = np.cos(a) * r_pt * per_x
                dy = np.sin(a) * r_pt * per_y
                cx_, cy_ = px + dx, py + dy
                ha = "left" if dx >= 0 else "right"
                x_lo = cx_ if ha == "left" else cx_ - lw
                rect = (x_lo, cy_ - lh / 2, x_lo + lw, cy_ + lh / 2)
                if not (w < rect[0] and rect[2] < e and
                        s < rect[1] and rect[3] < n):
                    continue
                # label_pad_pt keeps a visible gap between neighbouring
                # labels rather than letting them touch.
                if any(_overlaps(rect, o, per_x * label_pad_pt,
                                 per_y * label_pad_pt * 0.45)
                       for o in placed):
                    continue
                # Do not sit on top of any campus point.
                if np.any((pts[:, 0] > rect[0] - marker_r * per_x) &
                          (pts[:, 0] < rect[2] + marker_r * per_x) &
                          (pts[:, 1] > rect[1] - marker_r * per_y) &
                          (pts[:, 1] < rect[3] + marker_r * per_y)):
                    continue
                best = (cx_, cy_, ha, rect, r_pt)
                break
            if best:
                break

        if best is None:      # nowhere clean: place at the far ring anyway
            dx = radii[-1] * per_x
            cx_, cy_, ha = px + dx, py, "left"
            best = (cx_, cy_, ha, (cx_, cy_ - lh / 2, cx_ + lw, cy_ + lh / 2),
                    radii[-1])

        cx_, cy_, ha, rect, r_pt = best
        placed.append(rect)
        colr = registry.color(cid)
        if r_pt >= leader_min_pt:
            ax.plot([px, cx_ - (0.4 * per_x if ha == "left" else -0.4 * per_x)],
                    [py, cy_], color=colr, lw=0.35, alpha=0.7, zorder=4,
                    solid_capstyle="round")
        ax.text(cx_, cy_, label, ha=ha, va="center", fontsize=fontsize,
                color="#1f1f1f", zorder=7,
                path_effects=[pe.withStroke(linewidth=1.1, foreground="white")])

    _geographic_scale_bar(ax, w, e, s, n)
    if used:
        fig.text(0.995, 0.008, used.replace(".", " "), ha="right", va="bottom",
                 fontsize=4.4, color="#7a7a7a", style="italic")

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
                             figsize=(width, nrows * panel * 1.14 + 0.5))
    axes = np.atleast_1d(axes).ravel()
    imagery_ok = False          # only claim imagery in the caption if it loaded
    warned = False

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
            got = _add_basemap(ax, WEB_MERCATOR, IMAGERY_PROVIDERS, zoom=zoom)
            if got:
                imagery_ok = True
            else:
                ax.set_facecolor("#e9e9e9")
                if not warned:
                    print("  ! imagery unreachable — panels drawn plain")
                    warned = True
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
                        wspace=0.06, hspace=0.16)
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
