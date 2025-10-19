#!/usr/bin/env python3
"""
Build a per-episode shedding table from eval.py CSV and write a compact summary.

Columns in -> eval.py CSV: episode, reward, steps, served_total, served_crit, served_imp, ...
Writes:
  - *_shed_table.csv: episode, reward, steps, served_total, shed_frac, (shed_nodes)
  - prints summary lines for the thesis.
"""

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV from eval.py --save_csv ...")
    ap.add_argument("--out", default=None, help="Output CSV (default: <stem>_shed_table.csv)")
    ap.add_argument("--nodes", type=int, default=None, help="Total feeder nodes, e.g. 33/145/300/1888")
    ap.add_argument("--min-steps", type=int, default=4, help="Filter: keep episodes with steps >= this")
    args = ap.parse_args()

    p = Path(args.csv)
    df = pd.read_csv(p)

    need = ["episode", "reward", "steps", "served_total", "served_crit", "served_imp"]
    if any(c not in df.columns for c in need):
        raise SystemExit(f"Missing required columns in {p.name}: need {need}")

    df["shed_frac"] = 1.0 - df["served_total"]
    if args.nodes:
        df["shed_nodes"] = df["shed_frac"] * float(args.nodes)

    # filter out instant fails
    dff = df[df["steps"] >= args.min_steps].copy()

    out = Path(args.out) if args.out else p.with_name(p.stem + "_shed_table.csv")
    cols = ["episode", "reward", "steps", "served_total", "shed_frac"] + (["shed_nodes"] if args.nodes else [])
    dff[cols].to_csv(out, index=False)
    print(f"Wrote table: {out}")

    # thesis-friendly summary
    def s(d):
        base = (
            f"n={len(d)}, mean served_total={d['served_total'].mean():.3f}, "
            f"mean shed_frac={d['shed_frac'].mean():.3f}"
        )
        if args.nodes:
            base += f", mean shed_nodes={d['shed_nodes'].mean():.1f}/{args.nodes}"
        return base

    print("UNFILTERED:", s(df))
    print(f"FILTERED (steps≥{args.min_steps}):", s(dff))

if __name__ == "__main__":
    main()
