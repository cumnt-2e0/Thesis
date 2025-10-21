# scripts/visualise_case33_styled.py
#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandapower.networks as pn

from microgrid_safe_rl.augmentation.case33 import augment_case33
from microgrid_safe_rl.augmentation.case145 import assign_priorities  # public API

# -------------------- styling --------------------
def style_preset(theme: str = "light"):
    if theme == "dark":
        bg = "#0f0f12"; fg = "#e6e6e6"
        regular = "#9aa0a6"; tie = "#ff6e54"
        crit = "#e74c3c"; imp = "#ff9f1c"
        gen_c = "#2b7fff"; slack_c = "#2ecc71"; pv_c = "#ff7f0e"; bess_c = "#8e44ad"
    else:
        bg = "#ffffff"; fg = "#222222"
        regular = "#cfd4da"; tie = "#d55e00"
        crit = "#c21807"; imp = "#ff8c00"
        gen_c = "#1f77b4"; slack_c = "#2ca02c"; pv_c = "#ff7f0e"; bess_c = "#9467bd"
    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg, "axes.edgecolor": bg,
        "text.color": fg, "axes.labelcolor": fg, "xtick.color": fg, "ytick.color": fg,
        "font.size": 10, "savefig.facecolor": bg, "savefig.edgecolor": bg,
    })
    return {"fg": fg, "regular": regular, "tie": tie, "crit": crit, "imp": imp,
            "gen": gen_c, "slack": slack_c, "pv": pv_c, "bess": bess_c}

# -------------------- helpers --------------------
def build_graph(net):
    G = nx.Graph()
    for b in net.bus.index.tolist():
        G.add_node(int(b))
    if len(net.line):
        for idx, row in net.line.iterrows():
            G.add_edge(int(row.from_bus), int(row.to_bus), line_idx=int(idx))
    if hasattr(net, "trafo") and len(net.trafo):
        for _, r in net.trafo.iterrows():
            G.add_edge(int(r.hv_bus), int(r.lv_bus))
    return G

def positions(net, G, seed=42):
    if hasattr(net, "bus_geodata") and isinstance(net.bus_geodata, pd.DataFrame) and not net.bus_geodata.empty:
        geo = net.bus_geodata.reindex(net.bus.index)
        pos = {int(i): (float(geo.at[i, "x"]), float(geo.at[i, "y"]))
               for i in net.bus.index if pd.notna(geo.at[i, "x"]) and pd.notna(geo.at[i, "y"])}
        missing = [n for n in G.nodes if n not in pos]
        if missing:
            k = 1.5 / max(1, np.sqrt(len(missing)))
            pos.update(nx.spring_layout(G.subgraph(missing), seed=seed, k=k))
        return pos
    k = 1.5 / max(1, np.sqrt(len(G.nodes)))
    return nx.spring_layout(G, seed=seed, k=k)

def size_from_power(p, base=60.0, span=180.0):
    if p is None or len(p) == 0:
        return np.array([])
    vals = np.asarray(p, dtype=float)
    if np.allclose(vals.max(), vals.min()):
        return np.full_like(vals, base + span * 0.5)
    norm = (vals - vals.min()) / (vals.max() - vals.min())
    return base + span * norm

def switch_midpoints(net, idx_list):
    xs, ys = [], []
    if not idx_list or "et" not in net.switch.columns:
        return np.array([]), np.array([])
    sw = net.switch.loc[idx_list]
    sw = sw[sw["et"] == "l"]
    if sw.empty:
        return np.array([]), np.array([])
    use_geo = hasattr(net, "bus_geodata") and isinstance(net.bus_geodata, pd.DataFrame) and not net.bus_geodata.empty
    for _, row in sw.iterrows():
        li = int(row["element"])
        if li not in net.line.index:
            continue
        fb = int(net.line.at[li, "from_bus"]); tb = int(net.line.at[li, "to_bus"])
        if use_geo and fb in net.bus_geodata.index and tb in net.bus_geodata.index:
            x1, y1 = float(net.bus_geodata.at[fb, "x"]), float(net.bus_geodata.at[fb, "y"])
            x2, y2 = float(net.bus_geodata.at[tb, "x"]), float(net.bus_geodata.at[tb, "y"])
            xs.append(0.5*(x1+x2)); ys.append(0.5*(y1+y2))
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    return xs[mask], ys[mask]

def legend_pt(size: str) -> int:
    return {"small": 9, "med": 11, "large": 13}.get(size, 11)

# -------------------- priority assignment --------------------
def _fallback_assign_priorities(net, crit_pct: float, imp_pct: float):
    """Fallback if the public API signature differs. Uses total load per bus."""
    if len(net.load) == 0:
        return
    uniq_buses = net.load["bus"].astype(int).unique().tolist()
    n_bus = len(uniq_buses)
    n_crit = max(1, int(round(crit_pct * n_bus)))
    n_imp = max(0, int(round(imp_pct * n_bus)))
    # ensure they fit
    n_crit = min(n_crit, n_bus)
    n_imp = min(n_imp, n_bus - n_crit)

    agg = net.load.groupby("bus")["p_mw"].sum().sort_values(ascending=False)
    crit_buses = list(agg.index[:n_crit])
    imp_buses = [b for b in agg.index if b not in crit_buses][:n_imp]

    net.load["priority"] = 0
    net.load.loc[net.load["bus"].isin(imp_buses), "priority"] = 1
    net.load.loc[net.load["bus"].isin(crit_buses), "priority"] = 2

def apply_priorities_via_api(net, crit_pct: float, imp_pct: float, verbose: bool = True):
    """
    Calls the public API from case145 with percentages.
    Tries common signatures, else falls back to deterministic bus-load based split.
    """
    if len(net.load) == 0:
        return

    # If priority already present and seems valid, keep it.
    if "priority" in net.load.columns and net.load["priority"].notna().any():
        return

    called = False
    # Try common parameter spellings
    try:
        assign_priorities(net, critical_pct=crit_pct, important_pct=imp_pct)  # preferred
        called = True
    except TypeError:
        try:
            assign_priorities(net, crit_pct=crit_pct, imp_pct=imp_pct)
            called = True
        except TypeError:
            try:
                assign_priorities(net, critical_fraction=crit_pct, important_fraction=imp_pct)
                called = True
            except Exception:
                called = False
    except Exception:
        called = False

    if not called:
        _fallback_assign_priorities(net, crit_pct, imp_pct)

    if verbose:
        def tier_sum(pri): 
            if "priority" not in net.load.columns: 
                return 0.0, 0
            m = net.load["priority"] == pri
            return float(net.load.loc[m, "p_mw"].sum()), int(m.sum())
        p2, c2 = tier_sum(2)
        p1, c1 = tier_sum(1)
        p0, c0 = tier_sum(0)
        print("=== PRIORITY SUMMARY ===")
        print(f"  - critical:  count={c2:3d}  p_total={p2:.3f} MW")
        print(f"  - important: count={c1:3d}  p_total={p1:.3f} MW")
        print(f"  - normal:    count={c0:3d}  p_total={p0:.3f} MW")

# -------------------- load & augment --------------------
def load_case33():
    base = pn.case33bw()
    net, _meta = augment_case33(
        base,
        keep_slack=False,
        add_switches=True,
        add_tie_switches=True,
        run_pf_after=False,
        target_pv_frac=0.22,
        target_bess_p_frac=0.10,
        target_bess_e_hours=1.0,
    )
    return net

# -------------------- plotting --------------------
def plot_case33(env_id: str, out: Path, labels: str, theme: str, fmt: str, legend_size: str,
                crit_pct: float, imp_pct: float):
    colors = style_preset(theme)
    net = load_case33()

    # Apply priorities via public API (percentages) with safe fallback
    apply_priorities_via_api(net, crit_pct=crit_pct, imp_pct=imp_pct, verbose=True)

    # normalize flags
    if "is_tie" not in net.line.columns:
        net.line["is_tie"] = False
    net.line["is_tie"] = net.line["is_tie"].where(net.line["is_tie"].notna(), False).astype(bool)
    if hasattr(net, "switch") and len(net.switch):
        for col, default in (("is_controllable", False), ("is_tie", False), ("closed", True)):
            if col not in net.switch.columns:
                net.switch[col] = default
            net.switch[col] = net.switch[col].where(net.switch[col].notna(), default).astype(bool)

    # totals (shown in title)
    total_load = float(net.load["p_mw"].sum()) if len(net.load) else 0.0
    total_gen = float(net.gen["p_mw"].sum()) if hasattr(net, "gen") and len(net.gen) else 0.0
    total_sgen = float(net.sgen["p_mw"].sum()) if hasattr(net, "sgen") and len(net.sgen) else 0.0
    total_storage = float(net.storage["p_mw"].sum()) if hasattr(net, "storage") and len(net.storage) and "p_mw" in net.storage else 0.0

    G = build_graph(net)
    pos = positions(net, G, seed=33)

    fig, ax = plt.subplots(figsize=(12, 8))

    # edges
    regular, ties = [], []
    for u, v, data in G.edges(data=True):
        li = data.get("line_idx")
        if li is None or (li in net.line.index and bool(net.line.at[li, "is_tie"])):
            ties.append((u, v))
        else:
            regular.append((u, v))
    if regular:
        nx.draw_networkx_edges(G, pos, edgelist=regular, ax=ax,
                               edge_color=colors["regular"], width=1.2, alpha=0.95)
    if ties:
        nx.draw_networkx_edges(G, pos, edgelist=ties, ax=ax,
                               edge_color=colors["tie"], width=2.0, style="dashed", alpha=0.95)

    # base buses
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes), node_size=22,
                           node_color="#3b3b6d" if theme == "light" else "#aab4ff", alpha=0.95, linewidths=0)

    # slack + all gens
    slack_bus = None
    gen_buses = []
    if hasattr(net, "gen") and len(net.gen):
        if "slack" in net.gen.columns:
            m = net.gen["slack"].fillna(False).astype(bool)
            if m.any():
                slack_bus = int(net.gen.loc[m, "bus"].iloc[0])
        gen_buses = net.gen["bus"].astype(int).tolist()

    if gen_buses:
        gen_sizes = size_from_power(net.gen["p_mw"], base=60, span=180)
        nx.draw_networkx_nodes(G, pos, nodelist=gen_buses, node_shape="s", node_color="none",
                               edgecolors=colors["gen"], linewidths=1.8, node_size=gen_sizes, label="gen")

    # sgen (PV) + storage (BESS)
    sgen_buses = net.sgen["bus"].astype(int).tolist() if hasattr(net, "sgen") and len(net.sgen) else []
    if sgen_buses:
        sgen_sizes = size_from_power(net.sgen["p_mw"], base=60, span=180)
        nx.draw_networkx_nodes(G, pos, nodelist=sgen_buses, node_shape="*", node_color="none",
                               edgecolors=colors["pv"], linewidths=1.8, node_size=sgen_sizes, label="PV (sgen)")
    stor_buses = net.storage["bus"].astype(int).tolist() if hasattr(net, "storage") and len(net.storage) else []
    if stor_buses:
        p = net.storage["p_mw"] if "p_mw" in net.storage else pd.Series(np.zeros(len(stor_buses)))
        stor_sizes = size_from_power(p, base=60, span=180)
        nx.draw_networkx_nodes(G, pos, nodelist=stor_buses, node_shape="D", node_color="none",
                               edgecolors=colors["bess"], linewidths=1.8, node_size=stor_sizes, label="BESS")

    # priority rings (as assigned above)
    if len(net.load) and "priority" in net.load.columns:
        crit_buses = net.load.loc[net.load["priority"] == 2, "bus"].astype(int).unique().tolist()
        imp_buses = net.load.loc[net.load["priority"] == 1, "bus"].astype(int).unique().tolist()
        if crit_buses:
            nx.draw_networkx_nodes(G, pos, nodelist=crit_buses, node_shape="o", node_color="none",
                                   edgecolors=colors["crit"], linewidths=2.0, node_size=150, label="critical bus")
        if imp_buses:
            nx.draw_networkx_nodes(G, pos, nodelist=imp_buses, node_shape="o", node_color="none",
                                   edgecolors=colors["imp"], linewidths=1.8, node_size=130, label="important bus")

    # switches
    sec_sw_idx = tie_sw_idx = []
    if hasattr(net, "switch") and len(net.switch):
        sec_sw_idx = net.switch.index[(net.switch["et"] == "l") &
                                      (net.switch["is_controllable"]) & (~net.switch["is_tie"]) &
                                      (net.switch["closed"])].tolist()
        tie_sw_idx = net.switch.index[(net.switch["et"] == "l") &
                                      (net.switch["is_controllable"]) & (net.switch["is_tie"]) &
                                      (~net.switch["closed"])].tolist()
    sec_x, sec_y = switch_midpoints(net, sec_sw_idx)
    tie_x, tie_y = switch_midpoints(net, tie_sw_idx)
    if sec_x.size:
        ax.scatter(sec_x, sec_y, s=70, marker="s", facecolors="none",
                   edgecolors=colors["gen"], linewidths=1.8, label="Sectionalizing Sw (NC)")
    if tie_x.size:
        ax.scatter(tie_x, tie_y, s=90, marker="^", facecolors="none",
                   edgecolors=colors["tie"], linewidths=2.0, label="Tie Sw (NO)")

    # labels
    if labels != "none":
        if labels == "assets":
            label_buses = set(gen_buses + sgen_buses + stor_buses + ([slack_bus] if slack_bus is not None else []))
        else:
            label_buses = pos.keys()
        for b in label_buses:
            x, y = pos[b]
            ax.text(x, y + 0.02, str(b), ha="center", va="bottom", fontsize=7, color=colors["fg"])

    # title & legend
    ax.axis("off")
    handles, labels_list = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels_list):
        if l and l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=True, fontsize=legend_pt(legend_size))

    out = out.with_suffix(f".{fmt.lower()}")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=300 if fmt.lower() == "png" else None, bbox_inches="tight")
    print(f"[INFO] Saved: {out}")

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default="case33", help="feeder id (case33)")
    ap.add_argument("--out", default="artifacts/nets/case33_styled", help="output path without extension")
    ap.add_argument("--labels", choices=["none", "assets", "all"], default="assets")
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--format", choices=["png", "svg", "pdf"], default="svg")
    ap.add_argument("--legend", choices=["small", "med", "large"], default="large")
    # priority percentages (case33 test implied ~4/32 and ~7/32)
    ap.add_argument("--crit_pct", type=float, default=0.125, help="fraction of buses marked critical (by API)")
    ap.add_argument("--imp_pct", type=float, default=0.22, help="fraction of buses marked important (by API)")
    args = ap.parse_args()
    plot_case33(args.env_id, Path(args.out),
                labels=args.labels, theme=args.theme, fmt=args.format,
                legend_size=args.legend, crit_pct=args.crit_pct, imp_pct=args.imp_pct)

if __name__ == "__main__":
    main()
