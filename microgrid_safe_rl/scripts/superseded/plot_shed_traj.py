#!/usr/bin/env python3
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
    if "served_total" not in d.columns:
        raise SystemExit("CSV missing 'served_total' column from eval_per_step logger")

    d["shed_fraction"] = 1.0 - d["served_total"]

    plt.figure(figsize=(12,6))
    plt.plot(d["step"], d["served_total"], label="Total served", lw=2)
    if "served_crit" in d: plt.plot(d["step"], d["served_crit"],  label="Critical served", lw=2)
    if "served_imp"  in d: plt.plot(d["step"], d["served_imp"],   label="Important served", lw=2)
    plt.plot(d["step"], d["shed_fraction"], label="Shed fraction", lw=2, linestyle=":")

    plt.ylim(0, 1.02)
    plt.xlabel("Step"); plt.ylabel("Fraction")
    plt.title(f"Shed & Served Trajectories\n{Path(args.csv).name} (episode {ep})")
    plt.grid(True, alpha=0.3)

    # Shade fault-live spans
    if "fault_live" in d.columns:
        live = d["fault_live"].astype(bool).values
        steps = d["step"].values
        if live.any():
            starts = np.where(live & (~np.r_[False, live[:-1]]))[0]
            ends   = np.where(live & (~np.r_[live[1:], False]))[0]
            for s,e in zip(starts, ends):
                plt.axvspan(steps[s]-0.5, steps[e]+0.5, color="red", alpha=0.08)

    # PF failure markers
    for col_pf in ["pf_success", "powerflow_success", "pf_ok"]:
        if col_pf in d.columns:
            pf_bad = ~d[col_pf].astype(bool)
            if pf_bad.any():
                plt.scatter(d.loc[pf_bad, "step"], [0.0]*pf_bad.sum(), c="k", s=25, label="PF failed", zorder=5)
            break

    # Switch toggle ticks
    if "toggled_switch" in d.columns:
        toggled = d["toggled_switch"].fillna(-1).astype(int) >= 0
        if toggled.any():
            xs = d.loc[toggled, "step"].values
            ys = np.full_like(xs, 0.02, dtype=float)
            markerline, stemlines, baseline = plt.stem(xs, ys, linefmt="C7-", markerfmt="C7o", basefmt=" ")
            plt.setp(stemlines, linewidth=1, alpha=0.7)
            plt.setp(markerline, markersize=4)
            ymin, ymax = plt.ylim(); plt.ylim(min(0.0, ymin), ymax)

    # Isolation marker
    if "isolation_happened" in d.columns and d["isolation_happened"].any():
        s = int(d.loc[d["isolation_happened"].astype(bool), "step"].iloc[0])
        plt.axvline(s, color="C2", linestyle="--", alpha=0.6, label=f"Isolated @ {s}")

    # DER bound overlay (fraction)
    if "der_bound_frac" in d.columns:
        b = float(d["der_bound_frac"].iloc[0])
        plt.hlines(b, xmin=d["step"].min(), xmax=d["step"].max(), linestyles="--", alpha=0.4, label=f"DER bound (frac={b:.2f})")

    # Alternate MW overlay if available
    if {"served_mw","der_bound_mw"}.issubset(d.columns):
        ax2 = plt.gca().twinx()
        ax2.plot(d["step"], d["served_mw"], alpha=0.35)
        ax2.plot(d["step"], d["der_bound_mw"], alpha=0.35)
        ax2.set_ylabel("MW (served / DER bound)", alpha=0.7)

    # Cascade markers (if present)
    if "cascade_tripped" in d.columns:
        # accept either counts or non-empty lists/strings per step
        has_cascade = d["cascade_tripped"].notna() & (d["cascade_tripped"].astype(str) != "[]")
        if has_cascade.any():
            xs = d.loc[has_cascade, "step"].values
            ys = np.full_like(xs, 0.05, dtype=float)
            markerline, stemlines, baseline = plt.stem(xs, ys, linefmt="C3-", markerfmt="C3^", basefmt=" ")
            plt.setp(stemlines, linewidth=1, alpha=0.5)
            plt.setp(markerline, markersize=5)
            plt.text(xs[0], 0.06, "cascade", color="C3", fontsize=8)

    plt.legend(loc="lower right")
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(f".ep{ep}.shed.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"Saved plot: {out}")

if __name__ == "__main__":
    main()
