#!/usr/bin/env python
"""Step 5 — per-cell wide table with agreement classes.

    python scripts/05_unify.py --campus ui
    python scripts/05_unify.py --combine ui itb its
"""
import argparse

from campus_svi import unify


def main():
    ap = argparse.ArgumentParser(description="Unify per-cell metrics for one campus.")
    ap.add_argument("--campus")
    ap.add_argument("--combine", nargs="+", metavar="CAMPUS",
                    help="Stack already-unified campuses into one table")
    args = ap.parse_args()

    if args.campus:
        unify.unify_campus(args.campus.lower())
    if args.combine:
        df = unify.combine([c.lower() for c in args.combine])
        print(f"[combine] {len(df)} cells across {df['campus_id'].nunique()} campuses")


if __name__ == "__main__":
    main()
