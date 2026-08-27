"""Campus registry — identity, location, and display naming.

The pipeline keys everything on ``campus_id`` (the boundary filename stem).
That is deliberately a slug: lowercase, underscored, stable, safe in paths.
It is not what should appear on a figure.

This module maps each id to a properly cased display name and its
administrative location, and computes centroid and area from the boundary
files themselves rather than from any external source.

Multi-site universities keep the institution's own designation: UNESA numbers
its sites Campus 1 and 2, UNAIR letters its sites B and C, and ITB, UNPAD,
UNSRI and UDAYANA name theirs after the place. Those are the official labels,
so they are used verbatim rather than translated into place names.

Every row carries ``verified = False`` until you have checked it. City and
province are hand-entered here and are the one part of this file that is not
derived from your data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# campus_id -> (display_name, full_name, city, province, island)
REGISTRY: dict[str, tuple[str, str, str, str, str]] = {
    "ipb": ("IPB", "IPB University (Dramaga)",
            "Bogor", "West Java", "Java"),
    "itb_ganesha": ("ITB Ganesha", "Institut Teknologi Bandung — Ganesha",
                    "Bandung", "West Java", "Java"),
    "itb_jatinangor": ("ITB Jatinangor", "Institut Teknologi Bandung — Jatinangor",
                       "Sumedang", "West Java", "Java"),
    "itera": ("ITERA", "Institut Teknologi Sumatera",
              "South Lampung", "Lampung", "Sumatra"),
    "its": ("ITS", "Institut Teknologi Sepuluh Nopember",
            "Surabaya", "East Java", "Java"),
    "telkom_uni": ("Telkom University", "Telkom University",
                   "Bandung Regency", "West Java", "Java"),
    "ugm": ("UGM", "Universitas Gadjah Mada",
            "Sleman", "DI Yogyakarta", "Java"),
    "ui_main": ("UI", "Universitas Indonesia — Depok",
                "Depok", "West Java", "Java"),
    "uii_kaliurang": ("UII", "Universitas Islam Indonesia — Kaliurang",
                      "Sleman", "DI Yogyakarta", "Java"),
    "um": ("UM", "Universitas Negeri Malang",
           "Malang", "East Java", "Java"),
    "umy": ("UMY", "Universitas Muhammadiyah Yogyakarta",
            "Bantul", "DI Yogyakarta", "Java"),
    "unair_b": ("UNAIR Campus B", "Universitas Airlangga — Campus B",
                "Surabaya", "East Java", "Java"),
    "unair_c": ("UNAIR Campus C", "Universitas Airlangga — Campus C",
                "Surabaya", "East Java", "Java"),
    "unand": ("UNAND", "Universitas Andalas",
              "Padang", "West Sumatra", "Sumatra"),
    "unbraw": ("UB", "Universitas Brawijaya",
               "Malang", "East Java", "Java"),
    "undip": ("UNDIP", "Universitas Diponegoro — Tembalang",
              "Semarang", "Central Java", "Java"),
    "unej": ("UNEJ", "Universitas Jember",
             "Jember", "East Java", "Java"),
    "unesa_1": ("UNESA Campus 1", "Universitas Negeri Surabaya — Campus 1",
                "Surabaya", "East Java", "Java"),
    "unesa_2": ("UNESA Campus 2", "Universitas Negeri Surabaya — Campus 2",
                "Surabaya", "East Java", "Java"),
    "unhas": ("UNHAS", "Universitas Hasanuddin",
              "Makassar", "South Sulawesi", "Sulawesi"),
    "unila": ("UNILA", "Universitas Lampung",
              "Bandar Lampung", "Lampung", "Sumatra"),
    "unimed": ("UNIMED", "Universitas Negeri Medan",
               "Deli Serdang", "North Sumatra", "Sumatra"),
    "unj": ("UNJ", "Universitas Negeri Jakarta",
            "East Jakarta", "DKI Jakarta", "Java"),
    "unja": ("UNJA", "Universitas Jambi",
             "Muaro Jambi", "Jambi", "Sumatra"),
    "unlam": ("ULM", "Universitas Lambung Mangkurat",
              "Banjarbaru", "South Kalimantan", "Kalimantan"),
    "unmul": ("UNMUL", "Universitas Mulawarman",
              "Samarinda", "East Kalimantan", "Kalimantan"),
    "unnes": ("UNNES", "Universitas Negeri Semarang",
              "Semarang", "Central Java", "Java"),
    "unp": ("UNP", "Universitas Negeri Padang",
            "Padang", "West Sumatra", "Sumatra"),
    "unpad": ("UNPAD", "Universitas Padjadjaran — Jatinangor",
              "Sumedang", "West Java", "Java"),
    "unri": ("UNRI", "Universitas Riau",
             "Pekanbaru", "Riau", "Sumatra"),
    "uns": ("UNS", "Universitas Sebelas Maret",
            "Surakarta", "Central Java", "Java"),
    "unsoed": ("UNSOED", "Universitas Jenderal Soedirman",
               "Banyumas", "Central Java", "Java"),
    "unsrat": ("UNSRAT", "Universitas Sam Ratulangi",
               "Manado", "North Sulawesi", "Sulawesi"),
    "unsri_indralaya": ("UNSRI Indralaya", "Universitas Sriwijaya — Indralaya",
                        "Ogan Ilir", "South Sumatra", "Sumatra"),
    "untad": ("UNTAD", "Universitas Tadulako",
              "Palu", "Central Sulawesi", "Sulawesi"),
    "untan": ("UNTAN", "Universitas Tanjungpura",
              "Pontianak", "West Kalimantan", "Kalimantan"),
    "unud_jimbaran": ("UDAYANA Jimbaran", "Universitas Udayana — Jimbaran",
                      "Badung", "Bali", "Bali"),
    "uny": ("UNY", "Universitas Negeri Yogyakarta",
            "Yogyakarta", "DI Yogyakarta", "Java"),
    "upi": ("UPI", "Universitas Pendidikan Indonesia",
            "Bandung", "West Java", "Java"),
    "usu": ("USU", "Universitas Sumatera Utara",
            "Medan", "North Sumatra", "Sumatra"),
}

# --------------------------------------------------------------------------
# Institutional colours
# --------------------------------------------------------------------------

#: Sites of the same institution share a colour: ITB Ganesha and Jatinangor,
#: UNAIR Campus B and C, UNESA Campus 1 and 2 are one university each, and
#: colouring them apart would imply otherwise.
INSTITUTION = {
    "itb_ganesha": "itb", "itb_jatinangor": "itb",
    "unair_b": "unair", "unair_c": "unair",
    "unesa_1": "unesa", "unesa_2": "unesa",
    "ui_main": "ui", "uii_kaliurang": "uii",
    "unsri_indralaya": "unsri", "unud_jimbaran": "unud",
    "telkom_uni": "telkom",
}

#: Approximate institutional colours, for linking a campus across figures.
#: HAND-ENTERED AND UNVERIFIED — treat as a starting point and correct any you
#: know to be wrong; nothing downstream depends on the exact value.
#:
#: Note that 40 institutional colours cannot all be visually distinct, and
#: several universities share a similar blue or green. Colour here links a
#: point to its panel; the labels carry identity. Do not ask a reader to
#: identify a campus from colour alone.
INSTITUTION_COLORS = {
    "ipb": "#1F6F3D", "itb": "#005CA9", "itera": "#E4572E", "its": "#0E4C92",
    "telkom": "#E30613", "ugm": "#C8A415", "ui": "#FFD100", "uii": "#003A70",
    "um": "#1B6B4C", "umy": "#2E7D5B", "unair": "#003F87", "unand": "#1E7B45",
    "unbraw": "#14487F", "undip": "#003366", "unej": "#4B8F29",
    "unesa": "#1560BD", "unhas": "#C8102E", "unila": "#1F7A4D",
    "unimed": "#0F7B3F", "unj": "#007A33", "unja": "#E36414",
    "unlam": "#00539B", "unmul": "#F2A900", "unnes": "#1B8A5A",
    "unp": "#1C4E80", "unpad": "#C99700", "unri": "#0B6E4F", "uns": "#1F4E9C",
    "unsoed": "#2E6B8A", "unsrat": "#B22222", "unsri": "#F2C300",
    "untad": "#1E824C", "untan": "#005B96", "unud": "#A31621",
    "uny": "#2A6DB0", "upi": "#0B3C8C", "usu": "#007A3D",
}

_FALLBACK = "#8a8a8a"


def institution(campus_id: str) -> str:
    """Institution key for a campus site."""
    return INSTITUTION.get(campus_id, campus_id)


def color(campus_id: str) -> str:
    """Institutional colour for a campus, shared across its sites."""
    return INSTITUTION_COLORS.get(institution(campus_id), _FALLBACK)


def colors(campus_ids) -> list[str]:
    return [color(c) for c in campus_ids]


# Ids whose official abbreviation differs from the slug enough to be worth a
# second look when proofreading the registry.
RENAMED = {"unbraw": "UB", "unlam": "ULM", "unud_jimbaran": "UDAYANA"}


#: Canonical presentation order — the order REGISTRY is written in, which
#: begins with IPB. Every figure and table uses it so a reader can carry a
#: campus's position from one panel to the next without re-reading labels.
#: Sorting by a metric would make each figure internally tidy but mutually
#: incomparable, which is the more expensive mistake across a set this size.
CANONICAL_ORDER = list(REGISTRY.keys())


def ordered(campus_ids) -> list[str]:
    """Campus ids in canonical order; anything unregistered goes last, sorted."""
    rank = {c: i for i, c in enumerate(CANONICAL_ORDER)}
    known = [c for c in campus_ids if c in rank]
    rest = sorted(c for c in campus_ids if c not in rank)
    return sorted(known, key=lambda c: rank[c]) + rest


def display_name(campus_id: str) -> str:
    """Presentation name for a campus id, for figures and tables.

    Falls back to an upper-cased slug so an unregistered campus still renders
    something sensible rather than raising mid-figure.
    """
    e = REGISTRY.get(campus_id)
    return e[0] if e else campus_id.replace("_", " ").upper()


def display_names(campus_ids) -> list[str]:
    return [display_name(c) for c in campus_ids]


def entry(campus_id: str) -> dict:
    e = REGISTRY.get(campus_id)
    if not e:
        return {"campus_id": campus_id, "display_name": display_name(campus_id),
                "full_name": None, "city": None, "province": None,
                "island": None, "in_registry": False}
    return {"campus_id": campus_id, "display_name": e[0], "full_name": e[1],
            "city": e[2], "province": e[3], "island": e[4], "in_registry": True}


# --------------------------------------------------------------------------
# Building the CSV
# --------------------------------------------------------------------------

def build(campus_ids=None, include_cells: bool = True) -> pd.DataFrame:
    """Registry table with centroid and area derived from the boundary files.

    Centroid is computed in the campus's own UTM zone and converted back to
    WGS84, which is more accurate than taking a centroid in degrees.
    """
    from campus_svi import boundaries, cells as cellsmod

    ids = campus_ids or boundaries.list_campuses()
    rows = []
    for cid in ids:
        row = entry(cid)
        try:
            bnd = boundaries.load(cid)
            crs_m = boundaries.utm_crs(bnd)
            poly_m = bnd.to_crs(crs_m).geometry.iloc[0]
            cent = bnd.to_crs(crs_m).geometry.centroid.to_crs("EPSG:4326").iloc[0]
            row["centroid_lat"] = round(cent.y, 6)
            row["centroid_lon"] = round(cent.x, 6)
            row["area_km2"] = round(poly_m.area / 1e6, 4)
            row["perimeter_km"] = round(poly_m.length / 1e3, 4)
            row["utm_epsg"] = (crs_m.to_epsg() if hasattr(crs_m, "to_epsg")
                               else str(crs_m))
        except Exception as exc:                              # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"

        if include_cells:
            try:
                c = cellsmod.load_cells(cid)
                row["n_cells"] = len(c)
                row["cell_size_m"] = int(c["cell_size_m"].iloc[0]) if len(c) else None
            except Exception:                                 # noqa: BLE001
                row["n_cells"] = None
                row["cell_size_m"] = None

        # City and province are hand-entered, not derived. Check them once.
        row["verified"] = False
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["campus_id", "display_name", "full_name", "city", "province",
            "island", "centroid_lat", "centroid_lon", "area_km2",
            "perimeter_km", "utm_epsg", "n_cells", "cell_size_m",
            "in_registry", "verified"]
    return df[[c for c in cols if c in df.columns]]


def path() -> Path:
    from campus_svi import config
    return config.DATA_DIR / "reference" / "campus_registry.csv"


def write(campus_ids=None, verbose: bool = True) -> Path:
    df = build(campus_ids)
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    if verbose:
        print(f"[registry] {len(df)} campuses -> {p}")
        missing = df[~df["in_registry"]]["campus_id"].tolist()
        if missing:
            print(f"  ! not in REGISTRY (add them to registry.py): {missing}")
        print("  city/province are hand-entered — check them, then set "
              "verified = True")
    return p


def load() -> pd.DataFrame:
    p = path()
    if not p.exists():
        raise FileNotFoundError("Run registry.write() first.")
    return pd.read_csv(p)
