# microgrid_safe_rl/scripts/visualise_env.py

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandapower.networks as pn

from microgrid_safe_rl.feeders.registry import get_feeder
from microgrid_safe_rl.augmentation.common import assign_priorities


def _build_graph(net):
    """Create an undirected graph from lines & trafos (buses as nodes)."""
    G = nx.Graph()
    # ensure all buses exist as isolated nodes too
    for b in net.bus.index.tolist():
        G.add_node(int(b))

    # lines
    if len(net.line):
        for _, row in net.line.iterrows():
            G.add_edge(int(row.from_bus), int(row.to_bus))

    # transformers (treat as simple edges between hv/lv buses)
    for tname in ("trafo", "trafo3w"):
        if hasattr(net, tname) and len(getattr(net, tname)):
            df = getattr(net, tname)
            if tname == "trafo":
                for _, row in df.iterrows():
                    G.add_edge(int(row.hv_bus), int(row.lv_bus))
            else:  # 3-winding: connect all three
                for _, row in df.iterrows():
                    hv, mv, lv = int(row.hv_bus), int(row.mv_bus), int(row.lv_bus)
                    G.add_edge(hv, mv)
                    G.add_edge(mv, lv)
                    G.add_edge(hv, lv)
    return G


def _get_positions(net, G):
    """Geodata layout if present, else spring layout."""
    has_geo = hasattr(net, "bus_geodata") and len(getattr(net, "bus_geodata", [])) > 0
    if has_geo:
        geo = net.bus_geodata
        pos = {int(i): (float(r.x), float(r.y)) for i, r in geo.iterrows()}
        # Some buses may be missing geodata -> place them with spring near center
        missing = [n for n in G.nodes if n not in pos]
        if missing:
            fallback = nx.spring_layout(G.subgraph(missing), seed=42)
            pos.update(fallback)
    else:
        print("[INFO] No geodata found, using spring layout")
        pos = nx.spring_layout(G, seed=42, k=1.5 / max(1, np.sqrt(len(G.nodes))))
    return pos


def _size_from_pmw(p_series, base=40.0, span=140.0):
    """Map p_mw to marker sizes (relative, robust to huge values)."""
    if p_series is None or len(p_series) == 0:
        return np.array([])
    p = np.asarray(p_series, dtype=float)
    if np.allclose(p.max(), p.min()):
        return np.full_like(p, base + span * 0.5)
    norm = (p - p.min()) / (p.max() - p.min())
    return base + span * norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", type=str, help="Feeder ID (e.g., case145)")
    parser.add_argument("--out", type=str, default="grid_layout.png", help="Output image path")
    args = parser.parse_args()

    # Load feeder net (no PF needed for this static viz)
    feeder_fn = get_feeder(args.env_id)
    net = feeder_fn()

    # Ensure priorities exist exactly as used by the environment
    if len(net.load) and "priority" not in net.load.columns:
        assign_priorities(net)

    # Totals for title
    total_load = float(net.load["p_mw"].sum()) if len(net.load) else 0.0
    total_gen = float(net.gen["p_mw"].sum()) if hasattr(net, "gen") and len(net.gen) else 0.0
    total_sgen = float(net.sgen["p_mw"].sum()) if hasattr(net, "sgen") and len(net.sgen) else 0.0
    total_der = total_gen + total_sgen

    # Graph + positions
    G = _build_graph(net)
    pos = _get_positions(net, G)

    # Figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Edges (light)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray", width=0.6, alpha=0.8)

    # Base buses (uniform color/size)
    base_nodes = list(G.nodes)
    nx.draw_networkx_nodes(
        G, pos, nodelist=base_nodes, node_size=18, node_color="#3b3b6d", alpha=0.9, linewidths=0
    )

    # DERs
    if hasattr(net, "gen") and len(net.gen):
        gen_buses = net.gen["bus"].astype(int).tolist()
        gen_sizes = _size_from_pmw(net.gen["p_mw"], base=60, span=180)
        nx.draw_networkx_nodes(
            G, pos, nodelist=gen_buses, node_shape="s", node_color="none",
            edgecolors="#1f77b4", linewidths=1.2, node_size=gen_sizes, label="gen"
        )

    if hasattr(net, "sgen") and len(net.sgen):
        sgen_buses = net.sgen["bus"].astype(int).tolist()
        sgen_sizes = _size_from_pmw(net.sgen["p_mw"], base=60, span=180)
        nx.draw_networkx_nodes(
            G, pos, nodelist=sgen_buses, node_shape="^", node_color="none",
            edgecolors="#2ca02c", linewidths=1.2, node_size=sgen_sizes, label="sgen"
        )

    # Critical / Important load buses using existing priority field
    crit_buses, imp_buses = [], []
    if len(net.load) and "priority" in net.load.columns:
        crit_buses = net.load.loc[net.load["priority"] == 2, "bus"].astype(int).unique().tolist()
        imp_buses = net.load.loc[net.load["priority"] == 1, "bus"].astype(int).unique().tolist()

    if crit_buses:
        nx.draw_networkx_nodes(
            G, pos, nodelist=crit_buses, node_shape="o", node_color="none",
            edgecolors="#d62728", linewidths=1.8, node_size=130, label="critical load bus"
        )
    if imp_buses:
        nx.draw_networkx_nodes(
            G, pos, nodelist=imp_buses, node_shape="o", node_color="none",
            edgecolors="#ff7f0e", linewidths=1.6, node_size=110, label="important load bus"
        )

    # Title & legend
    ax.set_title(f"{args.env_id}: |P_load|={total_load:.1f} MW, |P_DER|={total_der:.1f} MW")
    ax.axis("off")

    # Clean legend: show only once per label
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[INFO] Saved visualisation to {args.out}")


if __name__ == "__main__":
    main()
