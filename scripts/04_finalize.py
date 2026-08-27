#!/usr/bin/env python
"""Step 4 — deduplicate and reclip to the true campus boundary.

The gate between raw and delivery-ready. Nothing downstream reads data/raw/.

    python scripts/04_finalize.py --campus ui
"""
import argparse

from campus_svi import finalize


def main():
    ap = argparse.ArgumentParser(description="Dedup + reclip one campus.")
    ap.add_argument("--campus", required=True)
    args = ap.parse_args()
    finalize.finalize_campus(args.campus.lower())


if __name__ == "__main__":
    main()
