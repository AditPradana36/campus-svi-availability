"""
paperstyle.py
-------------
Reusable matplotlib/seaborn styling that reproduces the figure conventions
observed in Nature-style benchmark papers:

  * humanist sans (Arial-metric), figure text SMALLER than body text
  * despine(trim=True) -> spines clipped to outermost ticks
  * no gridlines, no panel background, outward ticks
  * bold lowercase panel letters placed in figure coords
  * muted categorical palette with one dark saturated focal series
  * vector output, authored at final printed size

Usage:
    import paperstyle as ps
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.6))
    ...
    ps.finish(ax)                    # despine + trim
    ps.panel_letter(fig, ax, "a")
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------
# Canvas sizes: author at FINAL PRINTED SIZE, never scale in LaTeX.
# --------------------------------------------------------------------------
FULL_W = 7.0      # inches, full text width (two-column span)
HALF_W = 3.4      # inches, single column
COL_W = 4.6       # inches, 2/3 width

# --------------------------------------------------------------------------
# Type scale. Body text is ~10pt; figure text sits at 0.6-0.8x that.
# --------------------------------------------------------------------------
FS_TICK = 6.5
FS_LABEL = 7.5
FS_TITLE = 8.0
FS_LETTER = 9.5
FS_LEGEND = 6.0

# --------------------------------------------------------------------------
# Palette. Baselines muted; focal series dark + saturated so it wins on
# VALUE, not hue novelty -> survives grayscale and most CVD.
# --------------------------------------------------------------------------
MUTED = [
    "#9fd4c0",  # pale mint
    "#c3b49a",  # warm tan
    "#8a7358",  # mid brown
    "#9aa4cd",  # periwinkle
    "#4a4a73",  # navy-purple
    "#8ecae0",  # sky blue
    "#f2a58c",  # salmon
    "#3f8f7d",  # teal
]
FOCAL = "#8c2f2f"  # dark brick red -- reserved for "ours"

# A restrained alternative: greys for context, one accent, one focal.
MINIMAL = ["#c9c9c9", "#a8a8a8", "#878787", "#666666", "#3f8f7d"]

ALPHA_OBS = 0.40   # individual observations
LW_SPINE = 0.7
LW_ERR = 1.1
MS_OBS = 3.2
MS_MEAN = 5.0


def apply(font="Liberation Sans"):
    """Install the rcParams. Call once, before creating figures."""
    sns.set_theme(style="ticks")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.linewidth": LW_SPINE,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.titlelocation": "center",
        "axes.titlepad": 4,
        "axes.labelpad": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.major.width": LW_SPINE,
        "ytick.major.width": LW_SPINE,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed as TrueType -> text stays selectable
        "ps.fonttype": 42,
    })


def finish(ax, trim=True, offset=None):
    """Despine + trim: clip remaining spines to the outermost ticks.

    NOTE: seaborn's trim=True drops ticks that fall outside the DATA range,
    which silently wipes the labels on any categorical axis (forest plots,
    bar charts, anything with fixed tick positions). We snapshot the ticks
    and restore them afterwards.
    """
    xt, xl = ax.get_xticks(), [t.get_text() for t in ax.get_xticklabels()]
    yt, yl = ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    sns.despine(ax=ax, top=True, right=True, trim=trim, offset=offset)

    if trim:
        if len(ax.get_xticks()) == 0 and len(xt):
            ax.set_xticks(xt)
            if any(xl):
                ax.set_xticklabels(xl)
        if len(ax.get_yticks()) == 0 and len(yt):
            ax.set_yticks(yt)
            if any(yl):
                ax.set_yticklabels(yl)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def panel_letter(fig, ax, letter, dx=-0.085, dy=1.06):
    """Bold lowercase letter in AXES coords, nudged outside the axes.

    Placed relative to the axes rather than the figure so it stays put
    when y-label widths differ between panels.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=FS_LETTER, fontweight="bold",
            ha="left", va="bottom")


def sparse_yticks(ax, n=5):
    """Five or six ticks is plenty. Nobody needs eleven."""
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=n, prune=None))


def rotate_xlabels(ax, rotation=45, ha="right"):
    for lab in ax.get_xticklabels():
        lab.set_rotation(rotation)
        lab.set_ha(ha)
        lab.set_rotation_mode("anchor")


# --------------------------------------------------------------------------
# Heatmap conventions
# --------------------------------------------------------------------------
# Sequential for magnitudes on a common floor; diverging ONLY when zero is a
# meaningful midpoint (deltas, residuals, log-ratios). Both are perceptually
# uniform and CVD-safe -- do not substitute jet/rainbow.
CMAP_SEQ = "magma"
CMAP_SEQ_ALT = "viridis"
CMAP_DIV = "RdBu_r"

FS_ANNOT = 5.8      # in-cell numerals: smaller than tick labels
LW_CELL = 0.5       # hairline gap between cells
CELL_EDGE = "white"


def heatmap_axes(ax, frame=False):
    """Heatmaps want no trimmed spines -- the cell grid IS the frame.

    Ticks are removed entirely: cell boundaries already delimit categories,
    so tick marks are redundant ink.
    """
    for s in ax.spines.values():
        s.set_visible(frame)
        if frame:
            s.set_linewidth(LW_SPINE)
    ax.tick_params(length=0, pad=2)


def annot_color(value, vmin, vmax, cmap, threshold=0.55):
    """Pick black or white per cell so numerals stay legible.

    Uses relative luminance of the actual cell color rather than a fixed
    midpoint, which fails on asymmetric or diverging ramps.
    """
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    r, g, b, _ = mpl.colormaps[cmap](norm(value))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < threshold else "#1a1a1a"


def slim_colorbar(fig, mappable, ax, label="", orientation="vertical",
                  size=0.030, pad=0.02, n_ticks=5):
    """Thin colorbar with sparse ticks and a hairline outline removed.

    Vertical bars sit flush to the panel; horizontal ones are for maps where
    dead space inside the panel can absorb them.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    side = "right" if orientation == "vertical" else "bottom"
    cax = div.append_axes(side, size=f"{size * 100:.1f}%", pad=pad)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.8, width=LW_SPINE, labelsize=FS_TICK, pad=1.5)
    cb.set_label(label, fontsize=FS_LABEL, labelpad=3)
    cb.locator = mpl.ticker.MaxNLocator(nbins=n_ticks)
    cb.update_ticks()
    return cb
