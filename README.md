# Crowdsourced vs. proprietary street view coverage around Indonesian university campuses — Repository

Mohammad Raditia Pradana<sup>a,b,∗</sup>, Jarot Mulyo Semedi<sup>a,b</sup>

<sup>a</sup> SPARC (Spatial Modeling & Analysis Research Cluster), Universitas Indonesia, Depok, 16424, West Java, Indonesia
<sup>b</sup> Department of Geography, Faculty of Mathematics and Natural Sciences, Universitas Indonesia, Depok, 16424, West Java, Indonesia

> **Note:** This repository's codebase was developed with the assistance of [Claude Code](https://claude.com/claude-code), Anthropic's AI coding assistant.

---

Street-view imagery **availability** data for Indonesian university campuses: crowdsourced (Mapillary) and proprietary (Google) coverage, collected on a regular grid.

**Metadata only.** No imagery is downloaded.

Two stages: **acquisition** (`campus_svi/`) collects the data, **analysis** (`campus_svi/analysis/`) turns it into tables and publication figures. The analysis layer fetches nothing — it reads the acquisition outputs, so you can re-analyse without re-collecting.

📦 **[Dataset on Hugging Face](https://huggingface.co/datasets/rpradana36/campus-svi-availability)** — the published `data/points/` and `data/cells/` deliverables described below.

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

**Colab** — `notebooks/01_acquisition.ipynb` collects the data; `notebooks/02_analysis.ipynb` produces the tables and figures. They are deliberately separate: acquisition is a long network-bound run you do once, analysis is fast and iterative.

**Command line:**

```bash
pip install -r requirements.txt
export MAPILLARY_TOKEN="MLY|..."          # the only credential needed

python scripts/run.py --list              # what's in boundaries/
python scripts/run.py --campus ui_main    # one campus, all stages
python scripts/run.py --all               # every campus
python scripts/status.py                  # where everything stands

python scripts/make_registry.py           # campus registry CSV
python scripts/analyse.py                 # all tables and figures
python scripts/analyse.py --tables-only
python scripts/analyse.py --ncols 5 --skip-maup
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

## Campus registry

`data/reference/campus_registry.csv`, built by `scripts/make_registry.py`.

| Column | Source |
|---|---|
| `campus_id` | boundary filename stem — the key used throughout |
| `display_name`, `full_name` | lookup in `campus_svi/registry.py` |
| `city`, `province`, `island` | lookup — **hand-entered** |
| `centroid_lat`, `centroid_lon` | computed in the campus's UTM zone, returned as WGS84 |
| `area_km2`, `perimeter_km`, `utm_epsg` | computed from the boundary |
| `n_cells`, `cell_size_m` | from the cell table |
| `verified` | always written `False` — set it True once you have checked the row |

`campus_id` is a slug: stable, lowercase, safe in paths, and not what should appear on a figure. `display_name` is. Multi-site universities keep the institution's own designation — UNESA Campus 1 and 2, UNAIR Campus B and C — rather than place names, since those are the official labels.

---

## Campuses

40 campuses across Java, Sumatra, Kalimantan, Sulawesi and Bali, defined in `campus_svi/registry.py` and boundary files under `boundaries/`.

| Slug (`campus_id`) | Display name | Full name | City | Province | Island |
|---|---|---|---|---|---|
| `ipb` | IPB | IPB University (Dramaga) | Bogor | West Java | Java |
| `itb_ganesha` | ITB Ganesha | Institut Teknologi Bandung — Ganesha | Bandung | West Java | Java |
| `itb_jatinangor` | ITB Jatinangor | Institut Teknologi Bandung — Jatinangor | Sumedang | West Java | Java |
| `itera` | ITERA | Institut Teknologi Sumatera | South Lampung | Lampung | Sumatra |
| `its` | ITS | Institut Teknologi Sepuluh Nopember | Surabaya | East Java | Java |
| `telkom_uni` | Telkom University | Telkom University | Bandung Regency | West Java | Java |
| `ugm` | UGM | Universitas Gadjah Mada | Sleman | DI Yogyakarta | Java |
| `ui_main` | UI | Universitas Indonesia — Depok | Depok | West Java | Java |
| `uii_kaliurang` | UII | Universitas Islam Indonesia — Kaliurang | Sleman | DI Yogyakarta | Java |
| `um` | UM | Universitas Negeri Malang | Malang | East Java | Java |
| `umy` | UMY | Universitas Muhammadiyah Yogyakarta | Bantul | DI Yogyakarta | Java |
| `unair_b` | UNAIR Campus B | Universitas Airlangga — Campus B | Surabaya | East Java | Java |
| `unair_c` | UNAIR Campus C | Universitas Airlangga — Campus C | Surabaya | East Java | Java |
| `unand` | UNAND | Universitas Andalas | Padang | West Sumatra | Sumatra |
| `unbraw` | UB | Universitas Brawijaya | Malang | East Java | Java |
| `undip` | UNDIP | Universitas Diponegoro — Tembalang | Semarang | Central Java | Java |
| `unej` | UNEJ | Universitas Jember | Jember | East Java | Java |
| `unesa_1` | UNESA Campus 1 | Universitas Negeri Surabaya — Campus 1 | Surabaya | East Java | Java |
| `unesa_2` | UNESA Campus 2 | Universitas Negeri Surabaya — Campus 2 | Surabaya | East Java | Java |
| `unhas` | UNHAS | Universitas Hasanuddin | Makassar | South Sulawesi | Sulawesi |
| `unila` | UNILA | Universitas Lampung | Bandar Lampung | Lampung | Sumatra |
| `unimed` | UNIMED | Universitas Negeri Medan | Deli Serdang | North Sumatra | Sumatra |
| `unj` | UNJ | Universitas Negeri Jakarta | East Jakarta | DKI Jakarta | Java |
| `unja` | UNJA | Universitas Jambi | Muaro Jambi | Jambi | Sumatra |
| `unlam` | ULM | Universitas Lambung Mangkurat | Banjarbaru | South Kalimantan | Kalimantan |
| `unmul` | UNMUL | Universitas Mulawarman | Samarinda | East Kalimantan | Kalimantan |
| `unnes` | UNNES | Universitas Negeri Semarang | Semarang | Central Java | Java |
| `unp` | UNP | Universitas Negeri Padang | Padang | West Sumatra | Sumatra |
| `unpad` | UNPAD | Universitas Padjadjaran — Jatinangor | Sumedang | West Java | Java |
| `unri` | UNRI | Universitas Riau | Pekanbaru | Riau | Sumatra |
| `uns` | UNS | Universitas Sebelas Maret | Surakarta | Central Java | Java |
| `unsoed` | UNSOED | Universitas Jenderal Soedirman | Banyumas | Central Java | Java |
| `unsrat` | UNSRAT | Universitas Sam Ratulangi | Manado | North Sulawesi | Sulawesi |
| `unsri_indralaya` | UNSRI Indralaya | Universitas Sriwijaya — Indralaya | Ogan Ilir | South Sumatra | Sumatra |
| `untad` | UNTAD | Universitas Tadulako | Palu | Central Sulawesi | Sulawesi |
| `untan` | UNTAN | Universitas Tanjungpura | Pontianak | West Kalimantan | Kalimantan |
| `unud_jimbaran` | UDAYANA Jimbaran | Universitas Udayana — Jimbaran | Badung | Bali | Bali |
| `uny` | UNY | Universitas Negeri Yogyakarta | Yogyakarta | DI Yogyakarta | Java |
| `upi` | UPI | Universitas Pendidikan Indonesia | Bandung | West Java | Java |
| `usu` | USU | Universitas Sumatera Utara | Medan | North Sumatra | Sumatra |


---

## Analysis

`campus_svi/analysis/` reads `data/cells/` and `data/points/` and writes to `data/analysis/`.

| Module | Role |
|---|---|
| `registry.py` | Display names, city/province, centroid — builds `campus_registry.csv` |
| `metrics.py` | Numbers only — every figure value is also written as a CSV |
| `maps.py` | Small-multiple map engine |
| `figures.py` | The six core figures plus robustness panels |
| `style.py` | Colour contract shared across every figure |
| `paperstyle.py` | House style: final-size authoring, ~6pt type, trimmed spines, no grid |

### Figures

| File | Analysis |
|---|---|
| `fig1_coverage` | Coverage ratio per source per campus |
| `fig2_agreement_maps` | Cell-level source agreement, small multiples |
| `fig2b_agreement_composition` | Agreement class composition per campus |
| `fig3_decay` | Coverage against depth into campus |
| `fig3b_openness` | Decay slope per campus — the openness index |
| `fig4_temporal_depth` | Distinct Google capture years per cell |
| `fig4b_depth_diff` | Mapillary minus Google capture years |
| `fig5_temporal` | Annual volume, monthly series, burstiness |
| `fig6_programme` | Google capture programme composition |
| `figS1_maup` | Coverage at 20/50/100 m cells |
| `figS2_morans` | Moran's I per campus |

### Maps: per-panel zoom, per-panel scale bar

Each panel is **fitted to its own campus boundary**, so every campus fills its frame regardless of size. That keeps internal structure legible on a 0.2 km² campus and a 4 km² one alike, which is what matters when the subject is where coverage falls inside a boundary.

Scale therefore differs between panels, so **each panel carries its own scale bar**, placed below the frame — panels are filled edge to edge, so there is no reliable empty corner inside, and an interior bar collides with the data on some campus every time. A single shared bar would be false here.

What panel size no longer encodes is campus extent: a small campus and a large one look alike. The bars carry that, and `show_area=True` prints each campus's area beside its name.

Panels carry no axes, ticks, or per-panel legends — one colorbar and one legend serve the whole figure. At 40 faces, anything repeated 40 times is noise.

Forty campuses span several UTM zones, so each is projected to its own local UTM and translated to put its centroid at the origin; panels are drawn in metres from centre, with no common CRS to distort anything. At 20 m a campus can carry thousands of cells, so the cell layer is rasterised per panel while boundaries, text and bars stay vector — identical in print, but a PDF that opens.

### Two things the analysis needs that acquisition does not store

**Distance to boundary** (`metrics.add_boundary_distance`) is computed at analysis time from the boundary polygon. Normalised by each campus's own maximum, so 0 is the perimeter and 1 the deepest interior point, which makes campuses of different size comparable.

**Moran's I** (`metrics.morans_i`) is computed on the lattice directly from row/column adjacency, with a permutation test. No spatial-weights library, and no ambiguity about what counts as adjacent.

### Restyling

Colours live in `analysis/style.py`, the house style in `analysis/paperstyle.py`. Change them there and every figure follows. Do not override a colour in a single figure — the set stops reading as one document.

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
| `CELL_SIZE_M` | 20 | Analysis resolution. Does not change Google request count (tile-based) |
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
