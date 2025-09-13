# microgrid_safe_rl/augmentation/common.py
import numpy as np
import pandapower as pp
import pandas as pd

def ensure_switches_for_lines(net):
    # one sectionalizing switch at from_bus for each line (normally closed)
    for li in net.line.index:
        fb = int(net.line.at[li, "from_bus"])
        exists = ((net.switch.et == "l") & (net.switch.element == li)).any() if len(net.switch) else False
        if not exists:
            pp.create_switch(net, bus=fb, element=int(li), et="l", closed=True, name=f"SEC_SW_L{li}")

def ensure_ders(net, n=2, p_each=0.2):
    if len(net.bus) < 2 or n <= 0:
        return
    buses = list(net.bus.index)
    idxs = buses[-(n+1):-1] if len(buses) > (n+1) else buses[-n:]
    for b in idxs:
        pp.create_sgen(net, bus=int(b), p_mw=p_each, q_mvar=0.0, name=f"PV@{b}")

def ensure_bess(net, bus_idx=2, max_e=0.8, soc=60.0):
    if hasattr(net, "storage") and len(net.storage):
        return
    b = int(bus_idx) if bus_idx < len(net.bus) else int(len(net.bus)//2)
    pp.create_storage(net, bus=b, p_mw=0.0, max_e_mwh=max_e, soc_percent=soc, name="BESS")

def assign_priorities(net, top_pct=0.05, next_pct=0.10):
    if "p_base_mw" not in net.load.columns:
        net.load["p_base_mw"] = net.load["p_mw"].astype(float)
    load_bus = net.load.groupby("bus")["p_base_mw"].sum()
    if load_bus.empty:
        net.load["priority"] = 0
        return
    n_buses = len(load_bus)
    s = load_bus.sort_values(ascending=False)
    top_n = max(1, int(np.ceil(top_pct * n_buses)))
    next_n = max(1, int(np.ceil(next_pct * n_buses)))
    crit_buses = set(s.index[:top_n]); imp_buses = set(s.index[top_n: top_n+next_n])
    net.load["priority"] = [
        2 if int(row.bus) in crit_buses else (1 if int(row.bus) in imp_buses else 0)
        for _, row in net.load.iterrows()
    ]

def _add_tie_line(net, a, b, length_km=0.2, r_ohm_per_km=0.4, x_ohm_per_km=0.3):
    # avoid duplicate edges
    existing = (((net.line["from_bus"] == a) & (net.line["to_bus"] == b)) |
                ((net.line["from_bus"] == b) & (net.line["to_bus"] == a)))
    if existing.any():
        return
    li = pp.create_line_from_parameters(
        net, from_bus=int(a), to_bus=int(b), length_km=length_km,
        r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km,
        c_nf_per_km=0.0, max_i_ka=0.4, name=f"TIE_{a}_{b}"
    )
    # normally-open switch at 'a' so policy can close to reconfigure
    pp.create_switch(net, bus=int(a), element=int(li), et="l", closed=False, name=f"TIE_SW_{a}_{b}")

def _add_case33_ties(net):
    # simple heuristic ties for case33 to enable backfeeding
    buses = list(net.bus.index)
    if len(buses) < 8:
        return
    pairs = [
        (buses[-1], buses[len(buses)//2]),
        (buses[-3], buses[max(1, len(buses)//3)]),
        (buses[-5], buses[max(1, len(buses)//4)]),
    ]
    for a, b in pairs:
        if a != b:
            _add_tie_line(net, a, b)

def augment_full(net, feeder: str | None = None):
    feeder = (feeder or "").lower()
    ensure_switches_for_lines(net)

    if feeder == "case33":
        _add_case33_ties(net)          # give the small feeder useful switching
        ensure_ders(net, n=2, p_each=0.2)
        ensure_bess(net, bus_idx=2, max_e=0.8, soc=60.0)
    else:
        # big transmission-like cases: do NOT add synthetic DER/BESS or extra ties
        pass

    assign_priorities(net)
    return net
