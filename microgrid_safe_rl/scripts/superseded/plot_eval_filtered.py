#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_col(df, candidates):
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
    ap.add_argument("--nodes", type=int, required=True, help="Number of load nodes (e.g., 145)")
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
    col_iso_step = find_col(df, ["isolation_step", "isolated_at"])

    if col_ep is None or col_r is None or col_t is None:
        raise SystemExit(f"Missing required columns. Found: {list(df.columns)}")

    # Dynamic aggregation dict
    agg = {col_r: "last", col_t: "last"}
    if col_c: agg[col_c] = "last"
    if col_i: agg[col_i] = "last"
    if col_pf: agg[col_pf] = "any"
    if col_iso_step: agg[col_iso_step] = "max"

    g = df.groupby(col_ep, sort=True)
    stats = g.agg(agg).reset_index()

    # Shed metrics
    stats["shed_frac"] = 1.0 - stats[col_t]
    stats["shed_nodes"] = stats["shed_frac"] * args.nodes

    # Filtering: keep only episodes that had PF success at least once
    if col_pf:
        stats_f = stats[stats[col_pf].astype(bool)].copy()
    else:
        print("No PF success column found; skipping filtering")
        stats_f = stats.copy()

    print(f"Loaded: {len(stats)} episodes from {csv_path.name}")
    print(f"Filtered valid: {len(stats_f)} episodes with PF success")

    def maybe_mean(df_sub, col):
        return (df_sub[col].mean() if (col and col in df_sub.columns) else np.nan)

    def report(label, df_sub):
        if len(df_sub) == 0:
            print(f"  {label}: no data"); return
        iso_rate = np.nan
        iso_mean = np.nan
        if col_iso_step and col_iso_step in df_sub.columns:
            iso_mask = df_sub[col_iso_step].notna()
            iso_rate = 100.0 * iso_mask.mean()
            iso_mean = df_sub.loc[iso_mask, col_iso_step].mean()
        print(
            f"  {label}: reward={df_sub[col_r].mean():.2f}±{df_sub[col_r].std():.2f} | "
            f"served_total={df_sub[col_t].mean():.3f} | "
            f"served_crit={maybe_mean(df_sub, col_c):.3f} | "
            f"served_imp={maybe_mean(df_sub, col_i):.3f} | "
            f"shed_frac={df_sub['shed_frac'].mean():.3f} | "
            f"shed_nodes={df_sub['shed_nodes'].mean():.1f}/{args.nodes} | "
            f"isolation_rate={iso_rate:.1f}% | isolation_step(mean)={iso_mean:.2f}"
        )

    report("UNFILTERED", stats)
    report("FILTERED", stats_f)

    # --- Plot ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left: unfiltered
    if col_c and col_c in stats: ax[0].plot(stats[col_ep], stats[col_c], label="crit served", lw=2)
    if col_i and col_i in stats: ax[0].plot(stats[col_ep], stats[col_i], label="imp served", lw=2)
    ax[0].plot(stats[col_ep], stats[col_t], label="total served", lw=2)
    ax2 = ax[0].twinx(); ax2.plot(stats[col_ep], stats[col_r], ls="--", alpha=0.5)
    ax[0].set_title("UNFILTERED"); ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Served fraction")
    ax[0].grid(True, alpha=0.3); ax[0].legend(loc="lower right"); ax2.set_ylabel("Reward (dashed)")

    # Right: filtered
    if len(stats_f):
        if col_c and col_c in stats_f: ax[1].plot(stats_f[col_ep], stats_f[col_c], label="crit served", lw=2)
        if col_i and col_i in stats_f: ax[1].plot(stats_f[col_ep], stats_f[col_i], label="imp served", lw=2)
        ax[1].plot(stats_f[col_ep], stats_f[col_t], label="total served", lw=2)
        ax2 = ax[1].twinx(); ax2.plot(stats_f[col_ep], stats_f[col_r], ls="--", alpha=0.5); ax2.set_ylabel("Reward (dashed)")
        ax[1].legend(loc="lower right")
    ax[1].set_title("FILTERED (PF success only)"); ax[1].set_xlabel("Episode"); ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.out) if args.out else csv_path.with_name(csv_path.stem + "_filtered.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved plot: {out}")

if __name__ == "__main__":
    main()
