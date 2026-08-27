"""
campus_svi.analysis — analysis extension
========================================

Reads the acquisition deliverables (``data/points/``, ``data/cells/``) and
produces tables and publication figures. Fetches nothing.

    from campus_svi.analysis import metrics, figures
    ids = metrics.available()
    metrics.write_tables(ids)
    figures.build_all(ids)

Layout:
    metrics.py     numbers only — every figure value is also a table
    maps.py        small-multiple map engine, common scale, one scale bar
    figures.py     the six core figures plus robustness panels
    style.py       colour contract shared across every figure
    paperstyle.py  house style module
"""

from campus_svi.analysis import paperstyle, style  # noqa: F401

__all__ = ["metrics", "maps", "figures", "style", "paperstyle"]
