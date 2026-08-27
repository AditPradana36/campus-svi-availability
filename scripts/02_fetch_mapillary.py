#!/usr/bin/env python
"""Step 2 — fetch Mapillary image metadata, one campus at a time.

Resumable: re-running skips cells already logged as done.

    export MAPILLARY_TOKEN="MLY|..."
    python scripts/02_fetch_mapillary.py --campus ui
    python scripts/02_fetch_mapillary.py --campus ui --limit-cells 20   # smoke test
"""
import argparse

from campus_svi import mapillary


def main():
    ap = argparse.ArgumentParser(description="Fetch Mapillary metadata for one campus.")
    ap.add_argument("--campus", required=True)
    ap.add_argument("--limit-cells", type=int, default=None,
                    help="Process at most N pending cells this run")
    ap.add_argument("--sleep", type=float, default=None, help="Seconds between cells")
    ap.add_argument("--no-retry-failed", action="store_true",
                    help="Skip cells previously marked failed instead of retrying them")
    args = ap.parse_args()

    mapillary.fetch_campus(
        args.campus.lower(),
        limit_cells=args.limit_cells,
        retry_failed=not args.no_retry_failed,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
