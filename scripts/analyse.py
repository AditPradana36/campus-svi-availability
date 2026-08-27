#!/usr/bin/env python
"""Build analysis tables and figures from acquired data.

    python scripts/analyse.py                    # everything, all campuses
    python scripts/analyse.py --tables-only
    python scripts/analyse.py --campus ui_main itb_ganesha
    python scripts/analyse.py --ncols 6 --skip-maup
"""
import argparse

import matplotlib
matplotlib.use("Agg")

from campus_svi import config, registry
from campus_svi.analysis import figures, metrics


def main():
    ap = argparse.ArgumentParser(description="Analyse campus SVI availability.")
    ap.add_argument("--campus", nargs="+", help="Defaults to every campus with cell data")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ncols", type=int, default=8, help="Columns in small-multiple maps")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--skip-maup", action="store_true",
                    help="MAUP re-grids every campus at 3 sizes; slowest step")
    ap.add_argument("--skip-morans", action="store_true")
    ap.add_argument("--show-area", action="store_true",
                    help="Print each campus's area under its panel name")
    ap.add_argument("--no-registry", action="store_true")
    args = ap.parse_args()

    if args.root:
        config.set_root(args.root)

    ids = metrics.available(args.campus)
    if not ids:
        print("No campuses with cell data. Run acquisition first.")
        return
    print(f"{len(ids)} campuses: {', '.join(ids)}\n")

    if not args.no_registry:
        registry.write(ids)
        print()

    if not args.figures_only:
        print("tables")
        metrics.write_tables(ids)
        print()

    if not args.tables_only:
        figures.build_all(ids, ncols=args.ncols, show_area=args.show_area,
                          skip_maup=args.skip_maup, skip_morans=args.skip_morans)

    print(f"\noutputs under {config.DATA_DIR / 'analysis'}")


if __name__ == "__main__":
    main()
