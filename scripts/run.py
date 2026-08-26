#!/usr/bin/env python
"""Acquire SVI data for one campus, several, or all of them.

    python scripts/run.py --campus ui_main
    python scripts/run.py --all
    python scripts/run.py --campus itb_ganesha unpad undip
    python scripts/run.py --all --stages grid mapillary     # partial run
    python scripts/run.py --list                            # what's available

Every stage is checkpointed, so re-running resumes rather than restarting.
"""
import argparse

from campus_svi import boundaries, config, pipeline


def main():
    ap = argparse.ArgumentParser(description="Acquire campus SVI point and cell data.")
    ap.add_argument("--campus", nargs="+", help="One or more campus ids")
    ap.add_argument("--all", action="store_true", help="Every campus in boundaries/")
    ap.add_argument("--list", action="store_true", help="List campuses and exit")
    ap.add_argument("--stages", nargs="+", default=list(pipeline.STAGES),
                    choices=list(pipeline.STAGES))
    ap.add_argument("--cell-size", type=float, default=None, help="Grid cell metres")
    ap.add_argument("--seed-size", type=float, default=None,
                    help="Mapillary seed box metres")
    ap.add_argument("--no-enrich", action="store_true",
                    help="Skip Google stage B (no dates, tile geometry only)")
    ap.add_argument("--root", default=None, help="Project root on Drive")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    if args.root:
        config.set_root(args.root)
    config.ensure_dirs()

    found = boundaries.list_campuses()
    if args.list or not (args.campus or args.all):
        print(f"boundaries: {config.BOUNDARY_DIR}")
        print(f"{len(found)} campuses:")
        for c in found:
            print(" ", c)
        return

    targets = found if args.all else [c.lower() for c in args.campus]
    missing = [c for c in targets if c not in found]
    if missing:
        print(f"! not in boundaries/: {', '.join(missing)}")
        targets = [c for c in targets if c in found]
    if not targets:
        return

    kw = dict(stages=tuple(args.stages), cell_size_m=args.cell_size,
              seed_size_m=args.seed_size,
              enrich=None if not args.no_enrich else False)

    if len(targets) == 1:
        pipeline.run_campus(targets[0], **kw)
    else:
        df = pipeline.run_all(targets, stop_on_error=args.stop_on_error, **kw)
        print()
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
