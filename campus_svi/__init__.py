"""
campus_svi — street-view imagery availability acquisition
=========================================================

Acquisition only. Produces two deliverables per campus:

    data/points/{campus}_points.gpkg    every image/panorama inside the
                                        boundary, with full metadata
                                        (layers: mapillary, google)
    data/cells/{campus}_cells.gpkg      those points aggregated onto the
                                        analysis grid

Analysis is deliberately not part of this repo; it reads these outputs.

Stages
------
    1. grids.build_grid       boundary -> analysis grid (+ Mapillary seeds)
    2. mapillary.fetch_campus adaptive quadtree over resumable seed boxes
    3. google.fetch_campus    coverage tiles, then per-position enrichment
    4. points.build_points    dedup + reclip to the true boundary
    5. cells.build_cells      spatial join onto the grid

Or all of it:

    from campus_svi import pipeline
    pipeline.run_campus("ui_main")
    pipeline.run_all()
"""

__version__ = "1.0.0"

from campus_svi import config  # noqa: F401

__all__ = ["config", "boundaries", "grids", "mapillary", "google",
           "points", "cells", "pipeline", "checkpoint"]
