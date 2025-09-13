#!/usr/bin/env python3
"""
Plot served-fraction metrics from an eval CSV produced by `mgrl-eval --save_csv ...`.

The script is resilient to column naming:
- critical:   served_crit | serv_crit | crit_served
- important:  served_imp  | serv_imp  | imp_served
- total:      served_total| serv_total| total_served
- reward:     reward
- episode:    episode | ep
- DER bound:  der_bound_frac | bound_frac | der_bound

Usage:
  python microgrid_safe_rl/scripts/plot_served.py artifacts/models/case145_eval_very_hard.csv \
    --out artifacts/models/case145_eval_very_hard_served.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_col(df, candidates):
    # exact match first
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV produced by mgrl-eval --save_csv ...")
    ap.add_argument("--out", default=None, help="Output PNG path (default: alongside CSV)")
    ap.add_argument("--rolling", type=int, default=0, help="Rolling mean window for smoothing (0=off)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    # Guess column names
    col_ep = find_col(df, ["episode", "ep"])
    col_c  = find_col(df, ["served_crit", "serv_crit", "crit_served"])
    col_i  = find_col(df, ["served_imp", "serv_imp", "imp_served"])
    col_t  = find_col(df, ["served_total", "serv_total", "total_served"])
    col_r  = find_col(df, ["reward"])
    col_b  = find_col(df, ["der_bound_frac", "bound_frac", "der_bound"])  # NEW: DER bound overlay

    missing = [name for name, col in {
        "episode": col_ep, "crit": col_c, "imp": col_i, "total": col_t
    }.items() if col is None]
    if missing:
        raise SystemExit(
            f"CSV is missing required columns (or recognizable aliases): {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    # Optional smoothing
    if args.rolling and args.rolling > 1:
        for c in [col_c, col_i, col_t]:
            if c is not None:
                df[c] = df[c].rolling(args.rolling, min_periods=1).mean()

    # Build the plot
    plt.figure(figsize=(10, 5))
    plt.plot(df[col_ep], df[col_c], label="Critical served", linewidth=2)
    plt.plot(df[col_ep], df[col_i], label="Important served", linewidth=2)
    plt.plot(df[col_ep], df[col_t], label="Total served", linewidth=2)

    # Overlay DER bound if present
    if col_b is not None:
        # If constant per file, draw a single dashed line; else draw per-episode
        if df[col_b].nunique(dropna=True) <= 1:
            try:
                y = float(df[col_b].dropna().iloc[0])
                plt.axhline(y, linestyle="--", alpha=0.6, label="DER bound")
            except Exception:
                pass
        else:
            plt.plot(df[col_ep], df[col_b], linestyle="--", alpha=0.6, label="DER bound")

    plt.ylim(0, 1.05)
    plt.xlim(df[col_ep].min(), df[col_ep].max())
    plt.xlabel("Episode")
    plt.ylabel("Served fraction")
    plt.title(f"Served fractions vs. Episode\n{csv_path.name}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")

    # Optional: twin axis with reward, if present
    if col_r is not None:
        ax = plt.gca().twinx()
        ax.plot(df[col_ep], df[col_r], alpha=0.3, linestyle="--")
        ax.set_ylabel("Reward (dashed)")
        ax.grid(False)

    out = Path(args.out) if args.out else csv_path.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
