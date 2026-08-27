"""Small-multiple campus maps.

Each panel is **fitted to its own campus boundary**, so every campus fills its
frame regardless of size. That makes internal structure legible on a 0.2 km2
campus and a 4 km2 one alike, which is what matters when the subject is where
coverage falls inside a boundary.

The cost is that scale differs between panels, so each panel carries **its own
scale bar**. A single shared bar would be false here. The bars are deliberately
small — a short rule and a number under each panel — because forty of them must
not dominate the figure.

What panel size no longer encodes is campus extent: a small campus and a large
one look alike. The per-panel bars carry that, and ``area_km2`` from the
registry can be printed under each name with ``show_area=True``.

Two implementation notes. Forty campuses span several UTM zones, so each is
projected to its own local UTM and translated to put its centroid at the
origin; panels are drawn in metres from centre, with no common CRS to distort
anything. And at a 20 m grid a single campus can carry thousands of cells, so
the cell layer is rasterised per panel while boundaries, text and scale bars
stay vector — visually identical in print, but a PDF that opens.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from campus_svi import boundaries, config, registry
from campus_svi.analysis import metrics, paperstyle as ps
from campus_svi.analysis import style as st


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _localise(gdf, poly_utm, crs_m):
    """Local UTM, translated so the campus centroid sits at (0, 0)."""
    from shapely.affinity import translate

    c = poly_utm.centroid
    g = gdf.to_crs(crs_m).copy()
    g["geometry"] = g.geometry.apply(lambda geom: translate(geom, -c.x, -c.y))
    return g, translate(poly_utm, -c.x, -c.y)


def prepare(campus_ids, with_distance: bool = False):
    """Load and localise each campus once. Returns {campus_id: (cells, poly)}."""
    out = {}
    for cid in campus_ids:
        try:
            cells = metrics.load_cells(cid, with_distance=with_distance)
        except FileNotFoundError:
            continue
        bnd = boundaries.load(cid)
        crs_m = boundaries.utm_crs(bnd)
        poly = bnd.to_crs(crs_m).geometry.iloc[0]
        out[cid] = _localise(cells, poly, crs_m)
    return out


def _half_extent(poly) -> float:
    minx, miny, maxx, maxy = poly.bounds
    return max(maxx - minx, maxy - miny) / 2


# --------------------------------------------------------------------------
# Per-panel scale bar
# --------------------------------------------------------------------------

def _nice_length(window_m: float) -> float:
    """A round bar length around a third of the panel width."""
    target = window_m * 0.66
    for step in (25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000):
        if step >= target:
            return step
    return 5000


def panel_scale_bar(ax, half_window_m: float, y_frac: float = -0.055):
    """A compact scale bar for one panel, in that panel's own metres.

    Placed *below* the axes rather than inside it. Panels are fitted to their
    campus, so the map fills the frame and there is no reliable empty corner —
    an inside bar collides with the data on some campus every time. Below the
    axes it is always clear, and it reads as a caption to the panel.

    Bar and label sit on one line to keep the row gap small: forty of these
    must not become a band of furniture across the figure.
    """
    bar_m = _nice_length(half_window_m)
    frac = bar_m / (2 * half_window_m)
    if frac > 0.5:                           # too wide to leave room for text
        bar_m /= 2
        frac = bar_m / (2 * half_window_m)

    x0, x1 = 0.0, frac
    ax.plot([x0, x1], [y_frac, y_frac], transform=ax.transAxes,
            color="#2b2b2b", lw=0.7, solid_capstyle="butt",
            clip_on=False, zorder=10)
    for xx in (x0, x1):
        ax.plot([xx, xx], [y_frac - 0.016, y_frac + 0.016],
                transform=ax.transAxes, color="#2b2b2b", lw=0.7,
                clip_on=False, zorder=10)
    label = f"{bar_m/1000:g} km" if bar_m >= 1000 else f"{bar_m:g} m"
    ax.text(x1 + 0.035, y_frac, label, transform=ax.transAxes,
            ha="left", va="center", fontsize=4.8, color="#2b2b2b",
            clip_on=False, zorder=10)


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

def small_multiples(campus_ids, column: str, kind: str = "continuous",
                    ncols: int = 8, width: float = None,
                    cmap: str = None, vmin=None, vmax=None,
                    colors: dict = None, order: list = None,
                    labels: dict = None, title: str = "", cbar_label: str = "",
                    panel_letters: bool = False, show_boundary: bool = True,
                    show_area: bool = False, margin: float = 1.06,
                    rasterize: bool = True, sort_by=None, save: str = None):
    """One map face per campus, each fitted to its own boundary.

    kind
        ``"continuous"``  colour cells by a numeric column via one ramp
        ``"categorical"`` colour cells by a class column via ``colors``

    Returns ``(fig, axes)``.
    """
    ps.apply()
    width = width or ps.FULL_W
    cmap = cmap or st.CMAP

    prepared = prepare(campus_ids)
    ids = [c for c in campus_ids if c in prepared]
    if not ids:
        raise ValueError("No campuses with cell data.")
    if sort_by is not None:
        rank = {c: i for i, c in enumerate(sort_by)}
        ids.sort(key=lambda c: rank.get(c, 1e9))

    areas = {}
    if show_area:
        try:
            reg = registry.load().set_index("campus_id")["area_km2"].to_dict()
            areas = reg
        except Exception:                                     # noqa: BLE001
            areas = {}

    n = len(ids)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    panel = width / ncols
    # Extra height per row carries the name, the optional area, and the bar.
    # Row height carries the panel, its title, and the scale bar beneath it.
    fig_h = nrows * panel * 1.26 + (0.9 if kind else 0.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, fig_h))
    axes = np.atleast_1d(axes).ravel()

    if kind == "continuous":
        chunks = [prepared[c][0][column].to_numpy(float) for c in ids
                  if column in prepared[c][0].columns]
        vals = np.concatenate(chunks) if chunks else np.array([])
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            raise ValueError(f"No finite values for '{column}'.")
        vmin = float(vals.min()) if vmin is None else vmin
        vmax = float(vals.max()) if vmax is None else vmax
        if vmin == vmax:
            vmax = vmin + 1e-9

    for k, cid in enumerate(ids):
        ax = axes[k]
        cells, poly = prepared[cid]

        if kind == "continuous":
            if column in cells.columns and cells[column].notna().any():
                cells.plot(ax=ax, column=column, cmap=cmap, vmin=vmin, vmax=vmax,
                           linewidth=0, antialiased=False, rasterized=rasterize,
                           missing_kwds={"color": "#f4f4f4"})
            else:
                cells.plot(ax=ax, color="#f4f4f4", linewidth=0,
                           antialiased=False, rasterized=rasterize)
        elif column in cells.columns:
            for cls in (order or sorted(cells[column].dropna().unique())):
                sub = cells[cells[column] == cls]
                if not sub.empty:
                    sub.plot(ax=ax, color=(colors or {}).get(cls, "#cccccc"),
                             linewidth=0, antialiased=False,
                             rasterized=rasterize)

        if show_boundary:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color="#2b2b2b", lw=0.4, zorder=5)

        # Fit the panel to this campus, with a small margin.
        h = _half_extent(poly) * margin
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_aspect("equal")
        ax.set_axis_off()

        name = registry.display_name(cid)
        if show_area and cid in areas and areas[cid] == areas[cid]:
            name = f"{name}  ({areas[cid]:.2f} km\u00b2)"
        ax.set_title(name, fontsize=5.8, pad=1.8, color="#2b2b2b")

        # Each panel is at its own scale, so each carries its own bar.
        panel_scale_bar(ax, h)

        if panel_letters and k < 26:
            ps.panel_letter(fig, ax, chr(97 + k), dx=-0.02, dy=1.0)

    for ax in axes[n:]:
        ax.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=8, y=0.995)

    fig.subplots_adjust(left=0.005, right=0.995,
                        top=0.955 if title else 0.972,
                        bottom=0.075, wspace=0.06, hspace=0.30)

    # One colorbar or legend for the whole figure, never per panel.
    if kind == "continuous":
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cax = fig.add_axes([0.10, 0.038, 0.30, 0.011])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(cbar_label or column, fontsize=6, labelpad=2)
        cb.ax.tick_params(labelsize=5.6, length=2, pad=1.5)
        cb.outline.set_visible(False)
        cb.set_ticks(np.linspace(vmin, vmax, 4))
    elif kind == "categorical":
        handles = [Patch(facecolor=(colors or {}).get(c, "#ccc"), edgecolor="none",
                         label=(labels or {}).get(c, c.replace("_", " ")))
                   for c in (order or [])]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.005),
                   ncol=min(4, len(handles)), frameon=False, fontsize=6,
                   handlelength=1.1, handleheight=1.0, columnspacing=1.4,
                   borderpad=0)

    fig.text(0.995, 0.012, "Each panel at its own scale", ha="right",
             va="bottom", fontsize=5.2, color="#6f6f6f", style="italic")

    if save:
        save_figure(fig, save)
    return fig, axes


def save_figure(fig, stem: str, outdir=None, dpi: int = 400):
    """PDF for the manuscript, PNG for review.

    dpi governs the rasterised cell layer inside the PDF; text and boundaries
    stay vector regardless.
    """
    outdir = Path(outdir) if outdir else config.DATA_DIR / "analysis" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None: explicit subplots_adjust margins would otherwise be
    # overridden by "tight", which also shifts figure-coordinate annotations.
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches=None, dpi=dpi)
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches=None)
    print(f"  -> figures/{stem}.pdf, figures/{stem}.png")
    return outdir / f"{stem}.pdf"
