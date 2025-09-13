import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--episode", type=int, default=None,
                    help="Episode number to plot (default: first episode in file)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ep = args.episode or int(df["episode"].min())
    d = df[df["episode"] == ep].copy()

    # Main served-fraction traces
    plt.figure(figsize=(12,6))
    plt.plot(d["step"], d["served_total"], label="Total served", linewidth=2)
    plt.plot(d["step"], d["served_crit"],  label="Critical served", linewidth=2)
    plt.plot(d["step"], d["served_imp"],   label="Important served", linewidth=2)
    plt.ylim(0, 1.02)
    plt.xlabel("Step"); plt.ylabel("Served fraction")
    plt.title(f"Per-step served fractions\n{Path(args.csv).name} (episode {ep})")
    plt.grid(True, alpha=0.3)

    # Shade fault-live spans
    live = d["fault_live"].values.astype(bool)
    if live.any():
        steps = d["step"].values
        # find contiguous live segments
        starts = np.where((live) & (~np.r_[False, live[:-1]]))[0]
        ends   = np.where((live) & (~np.r_[live[1:], False]))[0]
        for s,e in zip(starts, ends):
            plt.axvspan(steps[s]-0.5, steps[e]+0.5, color="red", alpha=0.08)

    # PF fail dots
    pf_bad = ~d["pf_success"].astype(bool)
    if pf_bad.any():
        plt.scatter(d.loc[pf_bad, "step"], [0.0]*pf_bad.sum(), c="k", s=25, label="PF failed", zorder=5)

    # Switch toggle stems (on a tiny secondary scale near the bottom)
    toggled = d["toggled_switch"].fillna(-1).astype(int) >= 0
    if toggled.any():
        xs = d.loc[toggled, "step"].values
        ys = np.full_like(xs, 0.02, dtype=float)
        markerline, stemlines, baseline = plt.stem(xs, ys, linefmt="C7-", markerfmt="C7o", basefmt=" ")
        plt.setp(stemlines, linewidth=1, alpha=0.7)
        plt.setp(markerline, markersize=4)
        # keep the axis limits
        ymin, ymax = plt.ylim()
        plt.ylim(min(0.0, ymin), ymax)

    plt.legend(loc="lower right")
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(f".ep{ep}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot: {out}")

if __name__ == "__main__":
    main()
