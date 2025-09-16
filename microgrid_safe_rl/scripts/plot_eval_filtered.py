#!/usr/bin/env python3
"""
Plot eval results with filtering to exclude invalid (instant-fail) episodes.

Filtering rule:
  - Only keep episodes where `powerflow_success` was True at least once.

Also computes shed statistics:
  - shed_frac = 1 - served_total
  - shed_nodes = shed_frac * N_nodes

Usage:
  python microgrid_safe_rl/scripts/plot_eval_filtered.py results.csv \
    --nodes 145 \
    --out results_filtered.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_col(df, candidates):
    """Find first matching column from candidates (case insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV from mgrl-eval --save_csv ...")
    ap.add_argument("--nodes", type=int, required=True, help="Number of load nodes in feeder (e.g., 145)")
    ap.add_argument("--out", default=None, help="Output PNG (default: alongside CSV)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    # Guess column names
    col_ep = find_col(df, ["episode", "ep"])
    col_r  = find_col(df, ["reward"])
    col_c  = find_col(df, ["served_crit", "serv_crit", "crit_served"])
    col_i  = find_col(df, ["served_imp", "serv_imp", "imp_served"])
    col_t  = find_col(df, ["served_total", "serv_total", "total_served"])
    col_pf = find_col(df, ["powerflow_success", "pf_success", "pf_ok"])

    if col_ep is None or col_r is None or col_t is None:
        raise SystemExit(f"Missing required columns. Found: {list(df.columns)}")

    # Derive per-episode stats
    g = df.groupby(col_ep)
    stats = g.agg({
        col_r: "last",
        col_c: "last" if col_c else "mean",
        col_i: "last" if col_i else "mean",
        col_t: "last",
    }).reset_index()

    stats["shed_frac"] = 1.0 - stats[col_t]
    stats["shed_nodes"] = stats["shed_frac"] * args.nodes

    # --- Filtering: keep only episodes with PF success at least once ---
    if col_pf:
        valid_eps = g[col_pf].any()
        valid_eps = valid_eps[valid_eps].index
        stats_f = stats[stats[col_ep].isin(valid_eps)]
    else:
        print("No PF success column found; skipping filtering")
        stats_f = stats.copy()

    print(f"Loaded: {len(stats)} episodes from {csv_path.name}")
    print(f"Filtered valid: {len(stats_f)} episodes with PF success")

    def report(label, df_sub):
        if len(df_sub) == 0:
            print(f"  {label}: no data")
            return
        print(
            f"  {label}: mean reward={df_sub[col_r].mean():.2f} ± {df_sub[col_r].std():.2f} | "
            f"mean served_total={df_sub[col_t].mean():.3f} | "
            f"mean served_crit={df_sub[col_c].mean() if col_c else np.nan:.3f}, "
            f"served_imp={df_sub[col_i].mean() if col_i else np.nan:.3f} | "
            f"mean shed_frac={df_sub['shed_frac'].mean():.3f} | "
            f"mean shed_nodes={df_sub['shed_nodes'].mean():.1f} / {args.nodes}"
        )

    report("UNFILTERED", stats)
    report("FILTERED", stats_f)

    # --- Plot ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left: unfiltered
    ax[0].plot(stats[col_ep], stats[col_c], label="crit served", lw=2)
    ax[0].plot(stats[col_ep], stats[col_i], label="imp served", lw=2)
    ax[0].plot(stats[col_ep], stats[col_t], label="total served", lw=2)
    ax2 = ax[0].twinx()
    ax2.plot(stats[col_ep], stats[col_r], ls="--", alpha=0.5)
    ax[0].set_title("UNFILTERED")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Served fraction")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="lower right")
    ax2.set_ylabel("Reward (dashed)")

    # Right: filtered
    if len(stats_f):
        ax[1].plot(stats_f[col_ep], stats_f[col_c], label="crit served", lw=2)
        ax[1].plot(stats_f[col_ep], stats_f[col_i], label="imp served", lw=2)
        ax[1].plot(stats_f[col_ep], stats_f[col_t], label="total served", lw=2)
        ax2 = ax[1].twinx()
        ax2.plot(stats_f[col_ep], stats_f[col_r], ls="--", alpha=0.5)
        ax2.set_ylabel("Reward (dashed)")
        ax[1].legend(loc="lower right")
    ax[1].set_title("FILTERED (PF success only)")
    ax[1].set_xlabel("Episode")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.out) if args.out else csv_path.with_name(csv_path.stem + "_filtered.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
