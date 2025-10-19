import math
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandapower as pp

try:
    from pandapower.topology import connected_components
except Exception:
    connected_components = None

LOG = logging.getLogger("EnvFactory:AUG.CASE145")
LOG.setLevel(logging.INFO)

DEFAULTS = {
    # tiering
    "tier_percent_critical": 0.10,   # top 10% of buses by total P
    "tier_percent_important": 0.20,  # next 20%
    "reward_weights": {"tier1": 8.0, "tier2": 3.0, "tier3": 1.0},
    "log_tier_summary": True,

    # powerflow / energizing
    "pf_sanity_check": True,
    "energize": True,                     # <— turn on topology energizing at reset
    "close_line_switches": True,          # close line-to-bus switches
    "close_bus_bus_ties": True,           # close bus-bus ties (reconnect islands)
    "set_lines_in_service": True,         # set line.in_service=True
    "set_trafos_in_service": True,        # set trafo.in_service=True
    "disable_oltc_for_pf": True,          # disable OLTC during reset PF (until taps are defined)
}

# ---------- helpers ----------

def _fix_trafo_metadata(net):
    """Resolve pandapower tap column dtype warning and optionally disable OLTC until taps are defined."""
    if hasattr(net, "trafo") and len(net.trafo):
        # Drop or coerce legacy/foreign tap columns
        for col in ["tap_neutral", "tap_min", "tap_max", "tap_pos"]:
            if col in net.trafo.columns:
                try:
                    net.trafo.drop(columns=[col], inplace=True)
                    LOG.info("AUG.CASE145: dropped trafo column '%s' (avoids dtype warning).", col)
                except Exception:
                    pass

        if "oltc" in net.trafo.columns and DEFAULTS.get("disable_oltc_for_pf", True):
            try:
                net.trafo["oltc"] = False
                LOG.info("AUG.CASE145: temporarily disabled OLTC for PF stability.")
            except Exception:
                pass


def _total_bus_p(net):
    n_bus = len(net.bus)
    p = np.zeros(n_bus, dtype=float)
    if hasattr(net, "load") and len(net.load):
        for _, row in net.load.iterrows():
            if not row.get("in_service", True):
                continue
            b = int(row["bus"])
            p[b] += float(row.get("p_mw", 0.0))
    return p


def _assign_tiers_by_bus(net, pct_crit: float, pct_imp: float):
    """
    Compute per-bus total active power, rank buses, and assign both a string tier
    and an integer priority (0=critical, 1=important, 2=other) to each *load* row.
    """
    p_bus = _total_bus_p(net)
    order = np.argsort(-p_bus)  # descending
    n_bus = len(order)
    n_crit = max(1, int(math.ceil(pct_crit * n_bus))) if n_bus else 0
    n_imp = int(math.ceil(pct_imp * n_bus)) if n_bus else 0

    crit_buses = set(order[:n_crit])
    imp_buses = set(order[n_crit:n_crit + n_imp])
    norm_buses = set(order[n_crit + n_imp:])

    if hasattr(net, "load") and len(net.load):
        tiers = []
        prios = []
        for _, row in net.load.iterrows():
            b = int(row["bus"])
            if b in crit_buses:
                tiers.append("critical")
                prios.append(0)  # 0 = critical
            elif b in imp_buses:
                tiers.append("important")
                prios.append(1)  # 1 = important
            else:
                tiers.append("normal")
                prios.append(2)  # 2 = other
        net.load["tier"] = tiers
        net.load["priority"] = prios

    return {
        "p_bus": p_bus,
        "crit_buses": sorted(crit_buses),
        "imp_buses": sorted(imp_buses),
        "norm_buses": sorted(norm_buses),
    }


def _normalize_weights(w: Dict[str, float]):
    vals = np.array([w.get("tier1", 0.0), w.get("tier2", 0.0), w.get("tier3", 0.0)], dtype=float)
    s = float(np.sum(vals))
    if s <= 0:
        return {"tier1": 0.34, "tier2": 0.33, "tier3": 0.33}
    vals = vals / s
    return {"tier1": float(vals[0]), "tier2": float(vals[1]), "tier3": float(vals[2])}


def _count_live_lines(net) -> Tuple[int, list]:
    """
    Lines that are both in_service AND whose terminal buses are energized
    (based on res_bus vm_pu after a successful PF).
    """
    if not hasattr(net, "line") or not len(net.line):
        return 0, []

    if not hasattr(net, "res_bus") or "vm_pu" not in net.res_bus:
        # fall back to in_service count
        idx = net.line.index[net.line.get("in_service", True) == True]
        return int(len(idx)), list(idx)

    energized_buses = set(net.res_bus.index[net.res_bus["vm_pu"] > 0])
    live = []
    for idx, row in net.line.iterrows():
        if not row.get("in_service", True):
            continue
        fb = int(row["from_bus"])
        tb = int(row["to_bus"])
        if fb in energized_buses and tb in energized_buses:
            live.append(idx)
    return len(live), live


def _energize_topology(
    net,
    set_lines_in_service=True,
    set_trafos_in_service=True,
    close_line_switches=True,
    close_bus_bus_ties=True,
):
    """
    Make more of the physical network 'live' by ensuring equipment is in_service
    and relevant switches are closed, then run a PF to energize buses.
    """
    # 0) Fix trafo metadata / disable OLTC until taps are properly defined
    _fix_trafo_metadata(net)

    # 1) in_service flags
    if set_lines_in_service and hasattr(net, "line") and len(net.line):
        try:
            net.line["in_service"] = True
        except Exception:
            pass

    if set_trafos_in_service and hasattr(net, "trafo") and len(net.trafo):
        try:
            net.trafo["in_service"] = True
        except Exception:
            pass

    if hasattr(net, "load") and len(net.load):
        # keep loads on (you can choose to be selective later)
        try:
            net.load["in_service"] = True
        except Exception:
            pass

    if hasattr(net, "gen") and len(net.gen):
        # keep generators in service (adjust P limits elsewhere if needed)
        try:
            net.gen["in_service"] = True
        except Exception:
            pass

    # 2) switches
    if hasattr(net, "switch") and len(net.switch) and "closed" in net.switch.columns and "et" in net.switch.columns:
        if close_line_switches:
            mask_l = (net.switch["et"] == "l")
            net.switch.loc[mask_l, "closed"] = True
        if close_bus_bus_ties:
            mask_b = (net.switch["et"] == "b")
            net.switch.loc[mask_b, "closed"] = True

    # 3) initial PF to energize buses
    try:
        pp.runpp(net, algorithm="nr", init="auto", tolerance_mva=1e-8,
                 enforce_q_lims=True, max_iteration=80, numba=True)
    except Exception as e:
        LOG.warning("AUG.CASE145: PF after energize_topology failed: %s", e)

    # 4) log current live line count
    n_live, _ = _count_live_lines(net)
    LOG.info("AUG.CASE145: energize_topology → live_lines=%d / total=%d",
             n_live, len(net.line) if hasattr(net, "line") else 0)
 

def _pf_sanity(net):
    if not pp:
        LOG.warning("pandapower not available; skipping PF sanity check.")
        return {"ok": True, "note": "pp missing"}

    ok = True
    try:
        pp.runpp(net, algorithm="nr", init="results", tolerance_mva=1e-8,
                 enforce_q_lims=True, max_iteration=80, numba=True)
    except Exception as e:
        LOG.warning("PF sanity check failed: %s", e)
        ok = False

    max_loading = None
    if hasattr(net, "res_line") and len(net.res_line) and "loading_percent" in net.res_line:
        max_loading = float(np.nanmax(net.res_line["loading_percent"].values))

    # energized live lines (not just in_service)
    live_lines, _ = _count_live_lines(net)

    served_frac = 1.0
    if hasattr(net, "res_load") and len(net.res_load) and hasattr(net, "load") and len(net.load):
        served = float(np.nansum(net.res_load.get("p_mw", 0.0)))
        total = float(np.nansum(net.load.get("p_mw", 0.0)))
        served_frac = served / total if total > 0 else 1.0

    return {"ok": ok, "served_frac": served_frac, "max_line_load_pct": max_loading, "live_lines": live_lines}


def _tier_summary(net, p_bus, weights_norm):
    tiers = {"critical": {"count": 0, "p_total": 0.0},
             "important": {"count": 0, "p_total": 0.0},
             "normal": {"count": 0, "p_total": 0.0}}
    if hasattr(net, "load") and len(net.load):
        for _, row in net.load.iterrows():
            t = row.get("tier", "normal")
            tiers[t]["count"] += 1
            tiers[t]["p_total"] += float(row.get("p_mw", 0.0))

    top_bus_idx = list(np.argsort(-p_bus)[:10])
    bus_lines = []
    for b in top_bus_idx:
        p = p_bus[b]
        if p <= 0:
            continue
        mix = {}
        if hasattr(net, "load") and len(net.load):
            sub = net.load[net.load["bus"] == b]
            if len(sub):
                for t in ("critical", "important", "normal"):
                    c = int(np.sum(sub["tier"] == t)) if "tier" in sub else 0
                    if c > 0:
                        mix[t] = 1 if c == len(sub) else c
        bus_lines.append((b, p, mix))

    LOG.info("=== PRIORITY SUMMARY ===")
    LOG.info("tiers (count): %s", {k: v["count"] for k, v in tiers.items()})
    LOG.info("weight sum (should be 1.0): %.6f", weights_norm["tier1"] + weights_norm["tier2"] + weights_norm["tier3"])
    LOG.info("loads by tier (count, total MW):")
    LOG.info("  - critical  count=%4d  p_total=%.3f MW", tiers["critical"]["count"], tiers["critical"]["p_total"])
    LOG.info("  - important count=%4d  p_total=%.3f MW", tiers["important"]["count"], tiers["important"]["p_total"])
    LOG.info("  - normal    count=%4d  p_total=%.3f MW", tiers["normal"]["count"], tiers["normal"]["p_total"])

    if bus_lines:
        LOG.info("\nTOP 10 BUS TOTAL LOAD (MW) & local tier mix:")
        for b, p, mix in bus_lines:
            LOG.info("  - bus %4d (%d): %.3f MW  mix=%s", b, b, p, mix if mix else "{}")


# ---------- public API (expected by env/factory) ----------

def assign_priorities(
    net,
    critical_top_pct: float = DEFAULTS["tier_percent_critical"],
    important_next_pct: float = DEFAULTS["tier_percent_important"],
    reward_weights: Dict[str, float] = None,
    log_summary: bool = True,
) -> Dict[str, Any]:
    """
    Writes net.load['tier'] (str) and net.load['priority'] (int: 0/1/2) and returns a metadata dict.
    """
    rw = reward_weights or DEFAULTS["reward_weights"]
    tiers_meta = _assign_tiers_by_bus(net, critical_top_pct, important_next_pct)
    weights_norm = _normalize_weights(rw)
    if log_summary:
        _tier_summary(net, tiers_meta["p_bus"], weights_norm)
    return {
        "pct_critical": float(critical_top_pct),
        "pct_important": float(important_next_pct),
        "weights_norm": weights_norm,
        "crit_buses": tiers_meta["crit_buses"],
        "imp_buses": tiers_meta["imp_buses"],
        "norm_buses": tiers_meta["norm_buses"],
    }


def augment_case145(net, options: Dict[str, Any] = None, **kwargs) -> Tuple[object, Dict[str, Any]]:
    """
    Lightweight augmenter: energize topology; no coverage/global stress; keep equipment states.
    Returns (net, info).
    Accepts either an 'options' dict or loose kwargs for convenience.
    """
    # Merge options + kwargs and layer defaults
    opts_in = dict(options or {})
    opts_in.update(kwargs or {})
    opts = {**DEFAULTS, **opts_in}

    # 0) Topology energizing (before any PF sanity so it reflects the new baseline)
    if opts.get("energize", True):
        _energize_topology(
            net,
            set_lines_in_service=bool(opts.get("set_lines_in_service", True)),
            set_trafos_in_service=bool(opts.get("set_trafos_in_service", True)),
            close_line_switches=bool(opts.get("close_line_switches", True)),
            close_bus_bus_ties=bool(opts.get("close_bus_bus_ties", True)),
        )

    # 1) Priorities (writes 'tier' and 'priority')
    pri_meta = assign_priorities(
        net,
        critical_top_pct=float(opts.get("tier_percent_critical", DEFAULTS["tier_percent_critical"])),
        important_next_pct=float(opts.get("tier_percent_important", DEFAULTS["tier_percent_important"])),
        reward_weights=opts.get("reward_weights", DEFAULTS["reward_weights"]),
        log_summary=bool(opts.get("log_tier_summary", True)),
    )

    # 2) Optional PF sanity (reflects energized topology)
    pf_meta = _pf_sanity(net) if opts.get("pf_sanity_check", True) else {"ok": None}

    # 3) Pack info (coverage removed entirely)
    info = {
        "pf_sanity": pf_meta,
        "priority": pri_meta,
        "notes": "Coverage pass removed; topology energized at reset; tiers assigned by bus-load percentiles.",
    }

    net["aug_tag"] = "AUG.CASE145.v1"
    net.line["aug_line_tag"] = "AUG"

    # breadcrumb
    try:
        if not hasattr(net, "_aug"):
            net._aug = {}
        net._aug["case145"] = info
    except Exception:
        pass

    LOG.info("AUG.OK: topology energized; coverage disabled.")
    return net, info
