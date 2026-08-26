# Campus SVI — acquisition

Street-view imagery **availability** data for Indonesian university campuses: crowdsourced (Mapillary) and proprietary (Google) coverage, collected on a regular grid.

**Metadata only.** No imagery is downloaded. **Acquisition only** — no analysis lives here.

---

## What it produces

Two deliverables per campus, both on Drive:

```
data/points/{campus}_points.gpkg    every image / panorama inside the boundary,
                                    full metadata, layers: mapillary, google
data/cells/{campus}_cells.gpkg      those points aggregated onto the grid
data/cells/all_campuses_cells.csv   every campus stacked, for analysis
```

Analysis reads these. Nothing in this repo computes coverage ratios, decay curves, or figures.

---

## Quick start

**Colab** — open `notebooks/01_acquisition.ipynb`, which mounts Drive, runs one campus, then the batch.

**Command line:**

```bash
pip install -r requirements.txt
export MAPILLARY_TOKEN="MLY|..."          # the only credential needed

python scripts/run.py --list              # what's in boundaries/
python scripts/run.py --campus ui_main    # one campus, all stages
python scripts/run.py --all               # every campus
python scripts/status.py                  # where everything stands
```

Every stage is checkpointed. Re-running resumes; it never restarts.

---

## Setup

Boundaries live on Drive, one GeoPackage per campus, named by campus id:

```
/content/drive/MyDrive/campus-svi-availability/boundaries/
    ipb.gpkg  itb_ganesha.gpkg  itb_jatinangor.gpkg  its.gpkg  ugm.gpkg
    ui_main.gpkg  um.gpkg  unair_b.gpkg  unair_c.gpkg  unand.gpkg  ...
```

The filename stem becomes `campus_id` throughout, so a university with several sites simply gets several files. Shapefile and GeoJSON work too. Invalid geometry is repaired, multi-part boundaries dissolved, a missing CRS assumed to be EPSG:4326 with a warning.

Point the pipeline elsewhere with `config.set_root(...)` or `CAMPUS_SVI_ROOT`.

**Google needs no credentials.** `streetlevel` wraps Google's internal endpoints — no API key, no billing. They are undocumented and can change without notice, so pin the version you publish with.

---

## How it works

```
boundary
   ├─ 1. grid        square cells in local UTM + coarse Mapillary seed boxes
   ├─ 2. mapillary   adaptive quadtree inside each seed box     ─┐ async,
   ├─ 3. google      coverage tiles, then per-position enrich   ─┘ checkpointed
   ├─ 4. points      dedup + reclip to the true boundary   -> deliverable
   └─ 5. cells       spatial join onto the grid            -> deliverable
```

### Mapillary: adaptive subdivision

A whole-campus bbox is **refused outright** — `HTTP 500: "Please reduce the amount of data you're asking for"` — rather than truncated silently. That refusal is a usable signal and drives the design:

1. The campus is cut into **seed boxes** (`MLY_SEED_SIZE_M`, default 500 m), each an atomic checkpointed unit.
2. Inside a seed, a **quadtree** subdivides any box the API refuses. Refusal depends on *fields × limit*, not area alone, so a refused box is first retried at a lower page limit.
3. The page limit **ratchets down and is shared** across the run. Once the API has rejected 500, no later box rediscovers that by halving from 500 again — in testing this cut a campus from 28,358 requests to 6,274 with identical results. The settled value is reported at the end of each campus.
4. A sampled **verification split** (`MLY_VERIFY_FRACTION`, default 0.1) covers the other failure mode: a box that succeeded but was silently capped.

On verification: never test a result count against the limit you requested. The API's effective cap can be lower, in which case the check never fires and subdivision silently stops. Verification instead splits the box and asks whether the quadrants jointly return more.

### Google: two stages

| Stage | Call | Returns |
|---|---|---|
| A | `get_coverage_tile_async` | Every panorama on a zoom-17 tile (~304 m at Indonesian latitudes): id, position, elevation, orientation, links. **No dates.** |
| B | `find_panorama_by_id_async` | Capture date, `source`, `copyright_message`, `uploader`, street name, and the `historical` list of earlier panoramas. |

Stage A is an area query, structurally like a Mapillary bbox search, and reaches panoramas a radius search never surfaces. It carries geometry only — confirmed by dumping a raw tile response, where each entry is just `[[2, pano_id], null, [[lat, lon], [elevation], [heading, pitch, roll]]]`.

Stage B is the only source of temporal depth. Cost is one request per *position*, not per panorama: each response also returns that position's dated historical captures, so a position with five past captures yields six records from one request. Failures retry in their own rounds.

### Points and cells

Deduplication and reclipping are structural, not error handling. Mapillary images on a box seam are returned by both neighbours; Google tiles overlap, and a historical list can name a pano another tile already returned. Neither fetch unit respects the campus outline — tiles are ~304 m squares that overrun it wholesale — so points are clipped against the original boundary polygon.

Cells are assigned by **spatial join**, never by whichever fetch unit returned the point. Google panoramas are road-snapped and can land in a neighbouring cell, so fetch-time attribution would be wrong.

---

## Output schema

**Points — `mapillary` layer:** `image_id`, `lat`/`lon`, `computed_lat`/`computed_lon`, `captured_at` (Unix ms), `captured_date`, `year`, `year_month`, `compass_angle`, `computed_compass_angle`, `altitude`, `camera_type`, `is_pano`, `sequence_id`, `creator_id`, `creator_username`, `organization_id`.

**Points — `google` layer:** `pano_id`, `lat`/`lon`, `date`, `year`, `month`, `day`, `year_month`, `upload_year`, `capture_source`, `copyright_message`, `uploader`, `street_name`, `country_code`, `elevation`, `heading`/`pitch`/`roll`, `is_third_party`, `is_historical`, `parent_pano_id`, `n_neighbors`.

Two fields worth knowing:

- **`capture_source`** — the capture programme. `launch` is car coverage snapped to roads; **`scout` is trekker or tripod coverage that is not**; `innerspace` is Business View. This turns the road-snapping caveat into something measurable.
- **`day`** — present for third-party uploads only. Official Google coverage is month-level, so bin temporal analysis to year or month, or the two subsets are not comparable. Mapillary timestamps are millisecond-precise, which widens the gap further.

**Cells:** grid geometry plus `mly_count`, `mly_n_sequences`, `mly_n_creators`, `mly_n_orgs`, `mly_n_years`, `mly_year_min`/`max`, `mly_n_months`, `mly_pano_ratio`, `ggl_count`, `ggl_n_positions`, `ggl_n_historical`, `ggl_n_years`, `ggl_year_min`/`max`, `ggl_third_party_ratio`, `ggl_launch_ratio`, `ggl_scout_ratio`, `ggl_innerspace_ratio`, `ggl_mean_elevation`, plus `mly_coverage`, `ggl_coverage`, `either_coverage`, `agreement` (`both` / `mapillary_only` / `google_only` / `neither`) and `depth_diff`.

---

## Configuration

All in `campus_svi/config.py`.

| Setting | Default | Effect |
|---|---|---|
| `CELL_SIZE_M` | 100 | Analysis resolution. Does not change Google request count (tile-based) |
| `MLY_SEED_SIZE_M` | 500 | Atomic Mapillary fetch unit; also the resume granularity |
| `MLY_PAGE_LIMIT` | 500 | Starting page size; ratchets down on refusal |
| `MLY_MAX_DEPTH` | 10 | Quadtree ceiling inside a seed box |
| `MLY_VERIFY_FRACTION` | 0.1 | Share of boxes given the truncation check; 1.0 verifies all |
| `MLY_CONCURRENCY` / `GOOGLE_CONCURRENCY` | 8 | Lower this first if requests start failing |
| `GOOGLE_ENRICH` | `True` | `False` keeps tile geometry only, no dates |

---

## Troubleshooting

**`depth_exhausted` in the Mapillary summary** — the quadtree hit `MLY_MAX_DEPTH` with boxes still being refused, so some data may be missing. Raise it and re-run; finished seeds are skipped.

**Requests failing in bulk** — lower `MLY_CONCURRENCY` / `GOOGLE_CONCURRENCY` before raising the sleep values.

**A source shows 0% coverage** — usually a token problem or a boundary in the wrong place, not absent imagery. Check `status.py` first.

**A campus failed mid-batch** — re-run it. Every stage resumes from its checkpoint, and `run.py --all` skips completed work.

---

## Provenance

Record the `streetlevel` version and collection date with the data. These are undocumented endpoints, and the version is part of the provenance rather than housekeeping.

## License

MIT.
