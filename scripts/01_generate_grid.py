#!/usr/bin/env python
"""Step 1 — build the analysis grid for one campus.

    python scripts/01_generate_grid.py --campus ui
    python scripts/01_generate_grid.py --campus itb --cell-size 50
"""
import argparse

from campus_svi import boundaries, config, grids


def main():
    ap = argparse.ArgumentParser(description="Generate the analysis grid for one campus.")
    ap.add_argument("--campus", help="Campus id, i.e. the boundary filename stem (ui, itb, ...)")
    ap.add_argument("--cell-size", type=float, default=None, help="Cell edge length in metres")
    ap.add_argument("--min-overlap", type=float, default=None,
                    help="Drop cells overlapping the boundary by less than this fraction")
    ap.add_argument("--list", action="store_true", help="List available campuses and exit")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.list or not args.campus:
        found = boundaries.list_campuses()
        print("Campuses found in boundaries/:")
        for c in found:
            print(f"  {c}")
        if not found:
            print("  (none — drop ui.shp / itb.gpkg / ... into boundaries/)")
        return

    campus = args.campus.lower()
    grid = grids.build_grid(campus, cell_size_m=args.cell_size, min_overlap=args.min_overlap)
    path = grids.save_grid(grid, campus)

    size = grid["cell_size_m"].iloc[0]
    print(f"[grid/{campus}] {len(grid)} cells at {size:g} m")
    print(f"  campus area inside grid: {grid['area_inside_m2'].sum() / 1e6:.2f} km2")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
