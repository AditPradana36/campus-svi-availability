#!/usr/bin/env python
"""Step 6 — statistics and figures.

    python scripts/06_analysis.py --campus ui
    python scripts/06_analysis.py --compare ui itb its
"""
import argparse

import matplotlib
matplotlib.use("Agg")

from campus_svi import analysis


def main():
    ap = argparse.ArgumentParser(description="Analyse SVI availability.")
    ap.add_argument("--campus")
    ap.add_argument("--compare", nargs="+", metavar="CAMPUS",
                    help="Cross-campus coverage comparison figure")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if args.campus:
        analysis.run_campus(args.campus.lower(), figures=not args.no_figures)
    if args.compare:
        analysis.plot_coverage_bars([c.lower() for c in args.compare])


if __name__ == "__main__":
    main()
