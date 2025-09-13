#!/usr/bin/env python3
"""
Plot a feeder topology with bus/line/switch overlays.

- Uses geodata if present; otherwise falls back to a spring layout.
- Colors buses by highest-priority load connected at the bus:
    priority 2 (critical)   -> red
    priority 1 (important)  -> orange
    none / normal           -> gray
- Lines: in-service gray, out-of-service light red; highlight --fault_line in bright red.
- Line switches (et='l'): square at the line midpoint, green=closed, black=open.
- Bus switches (et='b'): small ring next to the bus, green=closed, black=open.

Examples:
  python scripts/plot_feeder.py --env_id case145 --env_cfg env.yaml \
    --out artifacts/nets/case145.png

  python scripts/plot_feeder.py --env_id case145 --env_cfg env.yaml \
    --fault_line 160 --out artifacts/nets/case145_fault160.png
"""

import argparse
from pathlib import Path
import copy

import numpy as np
import matplotlib.pyplot as plt

import pandapower as pp
import pandapower.topology as top
import networkx as nx

from microgrid_safe_rl.envs.factory import make_env
from microgrid_safe_rl.utils.config import load_yaml


def get_positions(net):
    # If geodata present, use it
    if hasattr(net, "bus_geodata") and len(net.bus_geodata):
        pos = {}
        for i, row in net.bus_geodata.iterrows():
            pos[i] = (row.x, row.y)
        return pos

    # Otherwise build a graph and use a spring layout
    g = top.create_nxgraph(net, include_switches=True)
    # layout in a deterministic way for consistent plots
    rng = np.random.default_rng(0)
    seed = int(rng.integers(0, 10_000))
    return nx.spring_layout(g, seed=seed, k=None)  # k auto by networkx


def draw_feeder(ax, net, fault_line=None, title=None, downsample=1):
    pos = get_positions(net)

    # ----- draw lines -----
    # Helper to plot a segment
    def seg(a, b, **kwargs):
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], **kwargs)

    # All lines
    for li, row in net.line.iterrows():
        a, b = int(row.from_bus), int(row.to_bus)
        color = "#c44e52" if not bool(row.in_service) else "#a0a0a0"
        lw = 1.0

        if fault_line is not None and int(li) == int(fault_line):
            color = "#d62728"  # bright red
            lw = 2.5

        seg(a, b, color=color, linewidth=lw, alpha=0.9)

    # ----- draw buses (colored by highest priority at the bus) -----
    pr_by_bus = {int(b): 0 for b in net.bus.index}
    if len(net.load):
        loads = net.load
        if "priority" not in loads.columns:
            # prioritize if missing (env normally adds it)
            from microgrid_safe_rl.augmentation import assign_priorities
            assign_priorities(net)
            loads = net.load

        for _, r in loads.iterrows():
            b = int(r.bus)
            p = int(r.get("priority", 0))
            pr_by_bus[b] = max(pr_by_bus[b], p)

    for b in net.bus.index:
        x, y = pos[b]
        pr = pr_by_bus[int(b)]
        if pr == 2:
            c = "#d62728"    # red critical
            z = 20
        elif pr == 1:
            c = "#ff7f0e"    # orange important
            z = 15
        else:
            c = "#7f7f7f"    # gray normal
            z = 10
        ax.scatter([x], [y], s=18, c=c, edgecolors="white", linewidths=0.4, zorder=z)

    # ----- draw switches -----
    if len(net.switch):
        # Line switches at midpoints
        line_sw = net.switch[net.switch.et == "l"]
        for si, sw in line_sw.iterrows():
            li = int(sw.element)
            if li not in net.line.index:
                continue
            a, b = int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])
            xa, ya = pos[a]; xb, yb = pos[b]
            xm, ym = (xa + xb) / 2.0, (ya + yb) / 2.0
            c = "#2ca02c" if bool(sw.closed) else "#000000"
            ax.scatter([xm], [ym], marker="s", s=16, c=c, edgecolors="white", linewidths=0.3, zorder=30)

        # Bus switches: draw near the bus
        bus_sw = net.switch[net.switch.et == "b"]
        for _, sw in bus_sw.iterrows():
            b = int(sw.bus)
            x, y = pos[b]
            # small offset so it doesn't overlap the bus marker
            x += 0.005; y += 0.005
            c = "#2ca02c" if bool(sw.closed) else "#000000"
            ax.scatter([x], [y], marker="o", s=14, facecolors="none", edgecolors=c, linewidths=1.0, zorder=25)

    # Legend / title
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left")
    # lightweight legend proxy
    ax.plot([], [], color="#a0a0a0", lw=2, label="line (in service)")
    ax.plot([], [], color="#c44e52", lw=2, label="line (out of service)")
    if fault_line is not None:
        ax.plot([], [], color="#d62728", lw=3, label=f"faulted line {fault_line}")
    ax.scatter([], [], s=18, c="#d62728", edgecolors="white", linewidths=0.4, label="bus w/ critical load")
    ax.scatter([], [], s=18, c="#ff7f0e", edgecolors="white", linewidths=0.4, label="bus w/ important load")
    ax.scatter([], [], s=18, c="#7f7f7f", edgecolors="white", linewidths=0.4, label="bus (normal)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", required=True, help="case33 | case145 | case300 | case1888")
    ap.add_argument("--env_cfg", default="env.yaml", help="env cfg (use same as training/eval)")
    ap.add_argument("--scenario_cfg", default=None, help="scenario cfg; default disables disturbance for a clean plot")
    ap.add_argument("--fault_line", type=int, default=None, help="highlight a known faulted line id")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    env_cfg = load_yaml(args.env_cfg)
    scenario = load_yaml(args.scenario_cfg) if args.scenario_cfg else {"enabled": False}

    # Build env, but we plot the *network* (no automatic fault unless scenario enabled)
    env = make_env(args.env_id, env_cfg, scenario)
    # Use pristine snapshot to avoid plotting live-fault behavior
    net = copy.deepcopy(env.net0)

    fig, ax = plt.subplots(figsize=(9, 7))
    title = f"{args.env_id}: buses={len(net.bus)}, lines={len(net.line)}, switches={len(net.switch)}"
    draw_feeder(ax, net, fault_line=args.fault_line, title=title)

    out = Path(args.out) if args.out else Path(f"artifacts/nets/{args.env_id}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=args.dpi)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
