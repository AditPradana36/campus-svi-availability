#!/usr/bin/env python
"""Build the campus registry CSV from the boundary files.

Centroid, area, perimeter and UTM zone are derived from your own boundaries.
Display name, city and province come from the lookup in campus_svi/registry.py
and are hand-entered — every row is written with verified = False so you can
check them in one pass.

    python scripts/make_registry.py
    python scripts/make_registry.py --root /content/drive/MyDrive/campus-svi-availability
"""
import argparse

from campus_svi import config, registry


def main():
    ap = argparse.ArgumentParser(description="Build the campus registry CSV.")
    ap.add_argument("--campus", nargs="+")
    ap.add_argument("--root", default=None)
    ap.add_argument("--no-cells", action="store_true",
                    help="Skip cell counts (use before acquisition has run)")
    args = ap.parse_args()

    if args.root:
        config.set_root(args.root)
    config.ensure_dirs()

    df = registry.build(args.campus, include_cells=not args.no_cells)
    p = registry.path()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

    print(f"[registry] {len(df)} campuses -> {p}\n")
    print(df[["campus_id", "display_name", "city", "province",
              "centroid_lat", "centroid_lon", "area_km2"]].to_string(index=False))

    missing = df[~df["in_registry"]]["campus_id"].tolist()
    if missing:
        print(f"\n! not in REGISTRY — add them to campus_svi/registry.py: {missing}")
    print("\nCity and province are hand-entered. Check them, then set "
          "verified = True in the CSV.")
    if registry.RENAMED:
        print("Worth a second look (abbreviation differs from the slug): "
              + ", ".join(f"{k} -> {v}" for k, v in registry.RENAMED.items()))


if __name__ == "__main__":
    main()
