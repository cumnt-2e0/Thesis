#!/usr/bin/env python3
"""
Per-episode shedding trajectory from eval_per_step CSV.

Adds a 'shed_fraction' trace = 1 - served_total.
Shades fault-live spans, marks PF failures, similar to plot_served_traj.py.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV from eval_per_step.py --out ...")
    ap.add_argument("--episode", type=int, default=None, help="Episode to plot (default: first)")
    ap.add_argument("--out", default=None, help="Output PNG (default: <stem>.epX.shed.png)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ep = args.episode or int(df["episode"].min())
    d = df[df["episode"] == ep].copy()
    if d.empty:
        raise SystemExit(f"Episode {ep} not found in {args.csv}")

    # derive shedding
    if "served_total" not in d.columns:
        raise SystemExit("CSV missing 'served_total' column from eval_per_step logger")
    d["shed_fraction"] = 1.0 - d["served_total"]

    # main plot
    plt.figure(figsize=(12,6))
    plt.plot(d["step"], d["served_total"], label="Total served", lw=2)
    plt.plot(d["step"], d["served_crit"],  label="Critical served", lw=2)
    plt.plot(d["step"], d["served_imp"],   label="Important served", lw=2)
    plt.plot(d["step"], d["shed_fraction"], label="Shed fraction", lw=2, linestyle=":")

    plt.ylim(0, 1.02)
    plt.xlabel("Step"); plt.ylabel("Fraction")
    plt.title(f"Shed & Served Trajectories\n{Path(args.csv).name} (episode {ep})")
    plt.grid(True, alpha=0.3)

    # shade fault-live
    if "fault_live" in d.columns:
        live = d["fault_live"].astype(bool).values
        steps = d["step"].values
        if live.any():
            starts = np.where((live) & (~np.r_[False, live[:-1]]))[0]
            ends   = np.where((live) & (~np.r_[live[1:], False]))[0]
            for s,e in zip(starts, ends):
                plt.axvspan(steps[s]-0.5, steps[e]+0.5, color="red", alpha=0.08)

    # PF fail markers
    if "pf_success" in d.columns:
        pf_bad = ~d["pf_success"].astype(bool)
        if pf_bad.any():
            plt.scatter(d.loc[pf_bad, "step"], [0.0]*pf_bad.sum(), c="k", s=25, label="PF failed", zorder=5)

    # switch toggles (optional visual tick near bottom)
    if "toggled_switch" in d.columns:
        toggled = d["toggled_switch"].fillna(-1).astype(int) >= 0
        if toggled.any():
            xs = d.loc[toggled, "step"].values
            ys = np.full_like(xs, 0.02, dtype=float)
            markerline, stemlines, baseline = plt.stem(xs, ys, linefmt="C7-", markerfmt="C7o", basefmt=" ")
            plt.setp(stemlines, linewidth=1, alpha=0.7)
            plt.setp(markerline, markersize=4)
            ymin, ymax = plt.ylim()
            plt.ylim(min(0.0, ymin), ymax)

    plt.legend(loc="lower right")
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(f".ep{ep}.shed.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"Saved plot: {out}")

if __name__ == "__main__":
    main()
