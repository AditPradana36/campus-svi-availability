#!/usr/bin/env python
"""Show acquisition progress for every campus.

    python scripts/status.py
    python scripts/status.py --root /content/drive/MyDrive/campus-svi-availability
"""
import argparse

from campus_svi import cells, config, pipeline


def main():
    ap = argparse.ArgumentParser(description="Acquisition status by campus.")
    ap.add_argument("--campus", nargs="+")
    ap.add_argument("--root", default=None)
    ap.add_argument("--combine", action="store_true",
                    help="Also write all_campuses_cells.csv")
    args = ap.parse_args()

    if args.root:
        config.set_root(args.root)

    df = pipeline.status(args.campus)
    print(f"boundaries: {config.BOUNDARY_DIR}\n")
    print(df.to_string(index=False))

    done = int(df["cells"].sum())
    print(f"\n{done}/{len(df)} campuses have cell data")

    if args.combine:
        print()
        cells.combine(df["campus_id"].tolist())


if __name__ == "__main__":
    main()
