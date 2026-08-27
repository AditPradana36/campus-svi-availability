#!/usr/bin/env python
"""Step 3 — fetch Google pano metadata via streetlevel (async), one campus at a time.

No API key required. Two resumable stages:
  A  coverage tiles at zoom 17  — spatial census of current coverage
  B  per-panorama enrichment    — date, capture source, copyright, historical

    python scripts/03_fetch_google.py --campus ui
    python scripts/03_fetch_google.py --campus ui --limit-tiles 3   # smoke test
    python scripts/03_fetch_google.py --campus ui --no-enrich       # geometry only
"""
import argparse

from campus_svi import config, google


def main():
    ap = argparse.ArgumentParser(
        description="Fetch Google SVI metadata for one campus via streetlevel."
    )
    ap.add_argument("--campus", required=True)
    ap.add_argument("--no-enrich", action="store_true",
                    help="Skip stage B; keep tile geometry only")
    ap.add_argument("--no-historical", action="store_true",
                    help="Skip expanding each panorama's historical list")
    ap.add_argument("--limit-tiles", type=int, default=None,
                    help="Process at most N tiles this run (smoke test)")
    ap.add_argument("--limit-panos", type=int, default=None,
                    help="Enrich at most N panoramas this run")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Concurrent in-flight requests (default 8)")
    ap.add_argument("--sleep", type=float, default=None)
    ap.add_argument("--tiles-only", action="store_true",
                    help="Alias for --no-enrich")
    args = ap.parse_args()

    print("library:", google.check_library())

    if args.concurrency:
        config.GOOGLE_CONCURRENCY = args.concurrency
    if args.sleep is not None:
        config.GOOGLE_SLEEP = args.sleep
    if args.no_historical:
        config.GOOGLE_INCLUDE_HISTORICAL = False

    google.fetch_campus(
        args.campus.lower(),
        enrich=not (args.no_enrich or args.tiles_only),
        limit_tiles=args.limit_tiles,
        limit_panos=args.limit_panos,
    )


if __name__ == "__main__":
    main()
