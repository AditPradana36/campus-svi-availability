"""Sample image download — for looking at what the metadata describes.

The rest of this project is metadata only, deliberately: coverage can be
measured from positions and dates without ever fetching a picture. This module
is the exception, for pulling a handful of images to check that a cell flagged
as covered really does contain usable streetscape, or to illustrate a finding.

It is not a bulk collector. Downloading a campus at full resolution would take
hours, produce gigabytes, and is not needed for anything in the analysis.
Keep the counts small.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from campus_svi import config, points, registry


def outdir(campus_id: str = None) -> Path:
    d = config.DATA_DIR / "samples"
    if campus_id:
        d = d / campus_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Selecting what to download
# --------------------------------------------------------------------------

def pick(campus_id: str, source: str = "mapillary", grid_id: str = None,
         n: int = 4, seed: int = 0) -> pd.DataFrame:
    """Choose sample records from a campus, optionally from one grid cell.

    Returns the metadata rows, so you can see dates and contributors before
    deciding to fetch anything.
    """
    layer = "mapillary" if source == "mapillary" else "google"
    gdf = points.load_points(campus_id, layer)
    if gdf.empty:
        return pd.DataFrame()

    if grid_id:
        # Points carry no grid_id — assignment happens at analysis time — so
        # join against the grid here rather than assuming a stored column.
        import geopandas as gpd

        from campus_svi import grids
        grid = grids.load_grid(campus_id)
        cell = grid[grid["grid_id"] == grid_id]
        if cell.empty:
            raise ValueError(f"No grid cell '{grid_id}' in {campus_id}.")
        gdf = gpd.sjoin(gdf.to_crs(grid.crs), cell[["grid_id", "geometry"]],
                        how="inner", predicate="within")
        gdf = gdf.drop(columns=["index_right"], errors="ignore")

    if gdf.empty:
        return pd.DataFrame()
    if n and len(gdf) > n:
        gdf = gdf.sample(min(n, len(gdf)), random_state=seed)
    return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))


def cells_with_coverage(campus_id: str, source: str = "mapillary",
                        min_count: int = 3, n: int = 10) -> pd.DataFrame:
    """Grid cells that actually hold records, as candidates to sample from."""
    from campus_svi import cells as cellsmod

    col = "mly_count" if source == "mapillary" else "ggl_count"
    c = cellsmod.load_cells(campus_id)
    c = c[c[col] >= min_count].sort_values(col, ascending=False)
    return pd.DataFrame(c.drop(columns="geometry")[["grid_id", col]].head(n))


# --------------------------------------------------------------------------
# Mapillary
# --------------------------------------------------------------------------

MLY_RESOLUTIONS = ["thumb_256_url", "thumb_1024_url", "thumb_2048_url",
                   "thumb_original_url"]


def download_mapillary(image_ids, campus_id: str = None,
                       resolution: str = "thumb_1024_url",
                       verbose: bool = True) -> list[Path]:
    """Fetch Mapillary images by id.

    The image URL is not in the stored metadata and is not stable, so it is
    requested per image at download time. ``thumb_original_url`` needs a token
    with the right scope; 1024 or 2048 is plenty for visual checking.
    """
    if resolution not in MLY_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {MLY_RESOLUTIONS}")
    token = config.require_mapillary_token()
    d = outdir(campus_id)
    saved = []
    ids = list(image_ids)
    if verbose and len(ids) > 20:
        print(f"  note: {len(ids)} images requested — this module is for samples.")

    for iid in ids:
        try:
            meta = requests.get(
                f"https://graph.mapillary.com/{iid}",
                params={"fields": resolution},
                headers={"Authorization": f"OAuth {token}"},
                timeout=60).json()
            url = meta.get(resolution)
            if not url:
                if verbose:
                    print(f"  ! {iid}: no {resolution} returned")
                continue
            img = requests.get(url, timeout=120)
            img.raise_for_status()
            p = d / f"mly_{iid}.jpg"
            p.write_bytes(img.content)
            saved.append(p)
            if verbose:
                print(f"  {p.name}  ({len(img.content)/1024:.0f} KB)")
        except Exception as exc:                              # noqa: BLE001
            if verbose:
                print(f"  ! {iid}: {type(exc).__name__}: {exc}")
    return saved


# --------------------------------------------------------------------------
# Google
# --------------------------------------------------------------------------

def download_google(pano_ids, campus_id: str = None, zoom: int = 3,
                    verbose: bool = True, concurrency: int = 4) -> list[Path]:
    """Fetch Google panoramas by pano id, via streetlevel.

    Uses streetlevel's **async** entry points. The synchronous
    ``download_panorama`` calls ``asyncio.run()`` internally, which raises
    inside a notebook because Colab and Jupyter already run an event loop —
    so every download fails with "cannot be called from a running event loop"
    even though the ids are perfectly good.

    ``zoom`` is the tile pyramid level, not a map zoom: 0 is a thumbnail and 5
    is full resolution, which is a large stitched image and slow. 3 suits a
    visual check.
    """
    import asyncio

    from aiohttp import ClientSession, ClientTimeout
    from streetlevel import streetview

    from campus_svi.mapillary import run_async      # notebook-safe loop helper

    ids = list(pano_ids)
    d = outdir(campus_id)
    saved: list[Path] = []

    if verbose and len(ids) > 12:
        print(f"  note: {len(ids)} panoramas requested. This module is for "
              f"samples — each is a stitched image and zoom {zoom} is not small.")

    async def _run():
        sem = asyncio.Semaphore(concurrency)
        async with ClientSession(timeout=ClientTimeout(total=300)) as session:
            async def one(pid):
                try:
                    async with sem:
                        pano = await streetview.find_panorama_by_id_async(
                            pid, session)
                        if pano is None:
                            if verbose:
                                print(f"  ! {pid}: not found")
                            return
                        p = d / f"ggl_{pid}_z{zoom}.jpg"
                        await streetview.download_panorama_async(
                            pano, str(p), session, zoom=zoom)
                    if p.exists():
                        saved.append(p)
                        if verbose:
                            print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB, "
                                  f"{getattr(pano, 'date', None)})")
                except Exception as exc:                      # noqa: BLE001
                    if verbose:
                        print(f"  ! {pid}: {type(exc).__name__}: {exc}")

            await asyncio.gather(*[one(p) for p in ids])

    run_async(_run())
    return saved


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------

def sample_cell(campus_id: str, grid_id: str, source: str = "mapillary",
                n: int = 4, **kw) -> list[Path]:
    """Pick and download a few records from one grid cell."""
    rows = pick(campus_id, source=source, grid_id=grid_id, n=n)
    if rows.empty:
        print(f"No {source} records in {grid_id}.")
        return []
    # n=0 means "no cap", which on a dense cell can be dozens of downloads.
    # Say what is about to happen rather than quietly starting.
    print(f"{registry.display_name(campus_id)} / {grid_id}: "
          f"downloading {len(rows)} of {source} record(s)"
          + ("  [n=0 means no limit]" if not n else ""))
    if source == "mapillary":
        return download_mapillary(rows["image_id"], campus_id, **kw)
    return download_google(rows["pano_id"], campus_id, **kw)


def contact_sheet(paths, ncols: int = 4, width: float = 7.0, save=None):
    """Lay downloaded images out in a grid, for a quick look at all of them."""
    import math

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        print("Nothing to show.")
        return None

    ncols = min(ncols, len(paths))
    nrows = math.ceil(len(paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width, width / ncols * nrows * 0.62))
    axes = [axes] if nrows * ncols == 1 else list(axes.ravel())

    for ax, p in zip(axes, paths):
        try:
            ax.imshow(mpimg.imread(p))
        except Exception:                                     # noqa: BLE001
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center",
                    fontsize=7, transform=ax.transAxes)
        ax.set_title(p.stem[:34], fontsize=5)
        ax.set_axis_off()
    for ax in axes[len(paths):]:
        ax.set_axis_off()

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
        print(f"-> {save}")
    return fig
