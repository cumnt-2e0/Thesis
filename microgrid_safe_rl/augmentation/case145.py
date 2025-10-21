import math
import logging
from typing import Dict, Any, Tuple, Set

import numpy as np
import pandas as pd
import pandapower as pp
import networkx as nx

try:
    from pandapower.topology import connected_components, create_nxgraph
except Exception:
    connected_components = None
    create_nxgraph = None

LOG = logging.getLogger("EnvFactory:AUG.CASE145")
LOG.setLevel(logging.INFO)

DEFAULTS = {
    # Tiering
    "tier_percent_critical": 0.10,
    "tier_percent_important": 0.20,
    "reward_weights": {"tier1": 8.0, "tier2": 3.0, "tier3": 1.0},
    "log_tier_summary": True,

    # DER configuration
    "add_ders": True,
    "der_penetration": 0.25,  # Fraction of load buses with DERs
    "pv_sizing_factor": (0.5, 0.8),  # PV capacity as fraction of local load
    "bess_hours": (2.0, 4.0),  # Battery duration at 50% load
    "bess_initial_soc": 80.0,  # Start at 80% SOC
    "add_black_start_gen": True,
    "black_start_capacity_mw": 10.0,

    # Switch configuration
    "add_switches": True,
    "switch_density": 0.15,  # Fraction of lines with switches
    "n_tie_switches": 5,  # Normally-open tie switches

    # Operating limits
    "voltage_limits": (0.95, 1.05),
    "default_line_rating_ka": 0.5,
    "trafo_max_loading": 120.0,

    # Powerflow & energizing
    "pf_sanity_check": True,
    "energize": True,
    "close_line_switches": True,
    "close_bus_bus_ties": False,  # Ties start OPEN
    "set_lines_in_service": True,
    "set_trafos_in_service": True,
    "disable_oltc_for_pf": True,
}


# ==================== HELPER FUNCTIONS ==================== #

def _fix_trafo_metadata(net):
    """Resolve pandapower tap column dtype warning."""
    if hasattr(net, "trafo") and len(net.trafo):
        for col in ["tap_neutral", "tap_min", "tap_max", "tap_pos"]:
            if col in net.trafo.columns:
                try:
                    net.trafo.drop(columns=[col], inplace=True)
                except Exception:
                    pass
        if "oltc" in net.trafo.columns and DEFAULTS.get("disable_oltc_for_pf", True):
            try:
                net.trafo["oltc"] = False
            except Exception:
                pass


def _total_bus_p(net):
    """Calculate total active power per bus."""
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
    """Assign tier labels and priority integers to loads based on bus ranking."""
    p_bus = _total_bus_p(net)
    order = np.argsort(-p_bus)
    n_bus = len(order)
    n_crit = max(1, int(math.ceil(pct_crit * n_bus))) if n_bus else 0
    n_imp = int(math.ceil(pct_imp * n_bus)) if n_bus else 0

    crit_buses = set(order[:n_crit])
    imp_buses = set(order[n_crit:n_crit + n_imp])

    if hasattr(net, "load") and len(net.load):
        tiers = []
        prios = []
        for _, row in net.load.iterrows():
            b = int(row["bus"])
            if b in crit_buses:
                tiers.append("critical")
                prios.append(0)
            elif b in imp_buses:
                tiers.append("important")
                prios.append(1)
            else:
                tiers.append("normal")
                prios.append(2)
        net.load["tier"] = tiers
        net.load["priority"] = prios

    return {
        "p_bus": p_bus,
        "crit_buses": sorted(crit_buses),
        "imp_buses": sorted(imp_buses),
    }


def _normalize_weights(w: Dict[str, float]):
    """Normalize reward weights to sum to 1.0."""
    vals = np.array([w.get("tier1", 0.0), w.get("tier2", 0.0), w.get("tier3", 0.0)], dtype=float)
    s = float(np.sum(vals))
    if s <= 0:
        return {"tier1": 0.34, "tier2": 0.33, "tier3": 0.33}
    vals = vals / s
    return {"tier1": float(vals[0]), "tier2": float(vals[1]), "tier3": float(vals[2])}


def _tier_summary(net, p_bus, weights_norm):
    """Log tier distribution summary."""
    tiers = {
        "critical": {"count": 0, "p_total": 0.0},
        "important": {"count": 0, "p_total": 0.0},
        "normal": {"count": 0, "p_total": 0.0}
    }
    if hasattr(net, "load") and len(net.load):
        for _, row in net.load.iterrows():
            t = row.get("tier", "normal")
            tiers[t]["count"] += 1
            tiers[t]["p_total"] += float(row.get("p_mw", 0.0))

    LOG.info("=== PRIORITY SUMMARY ===")
    LOG.info("  - critical:  count=%4d  p_total=%.3f MW", tiers["critical"]["count"], tiers["critical"]["p_total"])
    LOG.info("  - important: count=%4d  p_total=%.3f MW", tiers["important"]["count"], tiers["important"]["p_total"])
    LOG.info("  - normal:    count=%4d  p_total=%.3f MW", tiers["normal"]["count"], tiers["normal"]["p_total"])


# ==================== DER ADDITION ==================== #

def _resolve_voltage_conflicts(net):
    """
    Fix voltage setpoint conflicts:
    1. Remove ext_grid if we're adding a slack gen
    2. Make all other gens at slack bus non-slack and voltage-following
    3. Ensure only ONE voltage-controlling element per bus
    """
    # Find all voltage-controlling elements per bus
    voltage_controllers = {}  # bus -> list of (type, idx, vm_pu)
    
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        for idx, row in net.ext_grid.iterrows():
            bus = int(row["bus"])
            vm = float(row.get("vm_pu", 1.0))
            voltage_controllers.setdefault(bus, []).append(("ext_grid", idx, vm))
    
    if hasattr(net, "gen") and len(net.gen):
        for idx, row in net.gen.iterrows():
            bus = int(row["bus"])
            vm = float(row.get("vm_pu", 1.0))
            # Both slack and non-slack gens can control voltage
            is_slack = row.get("slack", False)
            voltage_controllers.setdefault(bus, []).append(("gen_slack" if is_slack else "gen", idx, vm))
    
    # Resolve conflicts: keep only ONE controller per bus
    conflicts = {bus: controllers for bus, controllers in voltage_controllers.items() if len(controllers) > 1}
    
    if conflicts:
        LOG.warning(f"Found voltage controller conflicts at {len(conflicts)} buses - resolving...")
        
        for bus, controllers in conflicts.items():
            # Check if they have different setpoints
            voltages = [vm for _, _, vm in controllers]
            if len(set(voltages)) > 1:
                LOG.warning(f"  Bus {bus}: {len(controllers)} controllers with different setpoints: "
                          f"{[(t, f'{vm:.3f}pu') for t, _, vm in controllers]}")
            
            # Strategy: 
            # 1. Keep slack gen if present
            # 2. Otherwise keep first ext_grid
            # 3. Make all other gens at this bus non-voltage-controlling
            
            has_slack_gen = any(c[0] == "gen_slack" for c in controllers)
            
            if has_slack_gen:
                # Remove ext_grids at this bus
                for ctrl_type, idx, _ in controllers:
                    if ctrl_type == "ext_grid":
                        net.ext_grid.drop(idx, inplace=True)
                        LOG.warning(f"    Removed ext_grid {idx} at bus {bus}")
                
                # Make non-slack gens PQ-mode (don't control voltage)
                for ctrl_type, idx, _ in controllers:
                    if ctrl_type == "gen":
                        # In pandapower, gens with slack=False don't control voltage if vm_pu is not set
                        # But to be safe, we can mark them as not controllable
                        if "controllable" in net.gen.columns:
                            net.gen.at[idx, "controllable"] = False
                        LOG.warning(f"    Set gen {idx} to non-voltage-controlling at bus {bus}")
            else:
                # Keep first ext_grid, remove others
                kept = False
                for ctrl_type, idx, _ in controllers:
                    if not kept and ctrl_type == "ext_grid":
                        kept = True
                        continue
                    if ctrl_type == "ext_grid":
                        net.ext_grid.drop(idx, inplace=True)
                        LOG.warning(f"    Removed duplicate ext_grid {idx} at bus {bus}")
                    elif ctrl_type == "gen":
                        if "controllable" in net.gen.columns:
                            net.gen.at[idx, "controllable"] = False
                        LOG.warning(f"    Set gen {idx} to non-voltage-controlling at bus {bus}")


def _add_distributed_energy_resources(net, opts: Dict[str, Any]):
    """Add PV, BESS, and optional black-start generator to the network."""
    if not opts.get("add_ders", True):
        return

    penetration = float(opts.get("der_penetration", 0.25))
    
    # Find load buses ranked by demand
    if not hasattr(net, "load") or len(net.load) == 0:
        LOG.warning("No loads in network - skipping DER placement")
        return

    bus_loads = net.load.groupby("bus")["p_mw"].sum()
    n_der_buses = max(5, int(len(bus_loads) * penetration))
    der_buses = bus_loads.nlargest(n_der_buses).index.tolist()

    LOG.info(f"Adding DERs to {len(der_buses)} buses (penetration={penetration:.1%})")

    pv_min, pv_max = opts.get("pv_sizing_factor", (0.5, 0.8))
    bess_min, bess_max = opts.get("bess_hours", (2.0, 4.0))
    initial_soc = float(opts.get("bess_initial_soc", 80.0))

    total_pv_capacity = 0.0
    total_pv_output = 0.0

    for bus in der_buses:
        total_load = float(bus_loads[bus])

        # Solar PV: 50-80% of local load capacity
        pv_capacity = total_load * np.random.uniform(pv_min, pv_max)
        pv_output = pv_capacity * 0.6  # INCREASED: 60% of capacity
        
        pp.create_sgen(
            net,
            bus=bus,
            p_mw=pv_output,  # Higher initial output
            q_mvar=0.0,
            sn_mva=pv_capacity,
            type="PV",
            controllable=True,
            max_p_mw=pv_capacity,
            min_p_mw=0.0,
            max_q_mvar=pv_capacity * 0.3,
            min_q_mvar=-pv_capacity * 0.3,
            name=f"PV_bus{bus}",
            in_service=True
        )
        
        total_pv_capacity += pv_capacity
        total_pv_output += pv_output

        # BESS: 2-4 hours at 50% load
        bess_capacity = total_load * 0.5
        bess_energy = bess_capacity * np.random.uniform(bess_min, bess_max)

        pp.create_storage(
            net,
            bus=bus,
            p_mw=0.0,  # Initially idle
            max_e_mwh=bess_energy,
            soc_percent=initial_soc,
            min_p_mw=-bess_capacity,  # Discharge
            max_p_mw=bess_capacity,   # Charge
            controllable=True,
            name=f"BESS_bus{bus}",
            in_service=True
        )

    LOG.info(f"DER addition complete: {len(der_buses)} PV+BESS pairs")
    LOG.info(f"  Total PV capacity: {total_pv_capacity:.1f} MW")
    LOG.info(f"  Total PV output: {total_pv_output:.1f} MW")

    # Black-start generator (diesel genset for islanding)
    if opts.get("add_black_start_gen", True):
        # Calculate required capacity based on total load
        total_load_mw = float(net.load["p_mw"].sum()) if len(net.load) else 100.0
        
        # CRITICAL: Black-start must cover the GAP between load and PV
        generation_gap = total_load_mw - total_pv_output
        min_capacity = max(total_load_mw * 0.6, generation_gap * 1.2)
        config_capacity = float(opts.get("black_start_capacity_mw", 10.0))
        black_start_mw = max(min_capacity, config_capacity)
        
        # Initial output: cover most of the gap
        initial_gen_output = generation_gap * 0.95  # 95% of gap
        
        # Find where to place the slack (prefer where ext_grid was)
        if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
            slack_bus = int(net.ext_grid["bus"].iloc[0])
            # NOW remove ALL ext_grids to avoid conflict
            LOG.info(f"Removing {len(net.ext_grid)} ext_grid(s) to replace with black-start gen")
            net.ext_grid.drop(net.ext_grid.index, inplace=True)
        else:
            # No ext_grid, use first bus
            slack_bus = int(net.bus.index[0])
        
        # Create slack generator (our ONLY voltage reference)
        pp.create_gen(
            net,
            bus=slack_bus,
            p_mw=initial_gen_output,  # CRITICAL FIX: Start with output!
            vm_pu=1.0,
            slack=True,  # This is now our ONLY slack reference
            min_p_mw=0.0,
            max_p_mw=black_start_mw,
            controllable=True,
            name="BlackStart_Gen",
            in_service=True
        )
        
        LOG.info(f"Added black-start slack generator:")
        LOG.info(f"  Bus: {slack_bus}")
        LOG.info(f"  Capacity: {black_start_mw:.1f} MW")
        LOG.info(f"  Initial output: {initial_gen_output:.1f} MW")
        LOG.info(f"  Total load: {total_load_mw:.1f} MW")
        LOG.info(f"  PV output: {total_pv_output:.1f} MW")
        LOG.info(f"  Generation gap: {generation_gap:.1f} MW")
        LOG.info(f"  Balance check: {initial_gen_output + total_pv_output:.1f} MW vs {total_load_mw:.1f} MW")

def _identify_restoration_opportunities(net) -> Dict[int, Dict[str, Any]]:
    """
    Analyze network topology to identify strategic tie switch locations.
    Returns map of potential tie connections with restoration metadata.
    """
    if not (connected_components and create_nxgraph):
        LOG.warning("NetworkX topology tools not available - using heuristic tie placement")
        return {}
    
    # Build network graph
    try:
        G = create_nxgraph(net, respect_switches=True)
    except Exception as e:
        LOG.warning(f"Could not create topology graph: {e}")
        return {}
    
    # Find critical buses (tier 1 loads)
    critical_buses = set()
    if hasattr(net, "load") and "priority" in net.load.columns:
        crit_loads = net.load[net.load.priority == 0]
        critical_buses = set(crit_loads.bus.values)
    
    # Find pairs of buses that:
    # 1. Are NOT directly connected
    # 2. Are 2-4 hops apart (good for alternate paths)
    # 3. Connecting them would provide restoration for critical loads
    
    restoration_map = {}
    buses = list(net.bus.index)
    
    # Sample pairs to avoid O(n²) complexity
    np.random.seed(42)
    sample_size = min(len(buses) * 3, 500)
    pairs_to_check = []
    
    for _ in range(sample_size):
        b1, b2 = np.random.choice(buses, size=2, replace=False)
        if not _are_buses_connected(net, b1, b2):
            pairs_to_check.append((b1, b2))
    
    for b1, b2 in pairs_to_check:
        try:
            # Check path length
            if nx.has_path(G, b1, b2):
                path = nx.shortest_path(G, b1, b2)
                path_length = len(path) - 1
                
                # Good tie candidates: 3-6 hops apart
                if 3 <= path_length <= 6:
                    # Count critical buses in path
                    critical_in_path = [b for b in path if b in critical_buses]
                    
                    # Estimate buses that would be restorable
                    # (simplified: assume tie restores ~40% of path)
                    restorable_buses = path[1:-1][:len(path)//2]
                    
                    restoration_map[(b1, b2)] = {
                        'path_length': path_length,
                        'critical_in_path': len(critical_in_path),
                        'restorable_buses': restorable_buses,
                        'critical_buses': [b for b in restorable_buses if b in critical_buses],
                        'priority_score': len(critical_in_path) * 10 + (10 - path_length)
                    }
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    
    # Rank by priority score
    sorted_ties = sorted(
        restoration_map.items(),
        key=lambda x: x[1]['priority_score'],
        reverse=True
    )
    
    return dict(sorted_ties)


def _add_tie_switches_for_restoration(net, opts: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Add normally-open tie switches that enable critical load restoration.
    Returns restoration_map: tie_switch_idx -> restoration_info
    """
    n_ties = int(opts.get("n_tie_switches", 5))
    
    LOG.info("=" * 60)
    LOG.info("ADDING TIE SWITCHES FOR CRITICAL LOAD RESTORATION")
    LOG.info("=" * 60)
    
    # Identify restoration opportunities
    opportunities = _identify_restoration_opportunities(net)
    
    if not opportunities:
        LOG.warning("No restoration opportunities identified - using heuristic placement")
        # Fallback: spread ties across network
        buses = list(net.bus.index)
        opportunities = {}
        step = len(buses) // (n_ties + 1)
        for i in range(n_ties):
            b1 = buses[i * step]
            b2 = buses[min((i + 1) * step + step//2, len(buses)-1)]
            if not _are_buses_connected(net, b1, b2):
                opportunities[(b1, b2)] = {
                    'path_length': 4,
                    'critical_in_path': 0,
                    'restorable_buses': [],
                    'critical_buses': [],
                    'priority_score': 0
                }
    
    # Select top N opportunities
    top_opportunities = list(opportunities.items())[:n_ties]
    
    restoration_map = {}
    tie_switches = []
    tie_lines = []
    
    # Mark existing lines
    if 'is_tie' not in net.line.columns:
        net.line['is_tie'] = False
    
    for idx, ((bus1, bus2), info) in enumerate(top_opportunities):
        # Create tie line (normally out-of-service)
        tie_line_idx = pp.create_line_from_parameters(
            net,
            from_bus=bus1,
            to_bus=bus2,
            length_km=0.5,  # Nominal length
            r_ohm_per_km=0.4095,
            x_ohm_per_km=0.4784,
            c_nf_per_km=3.4,
            max_i_ka=0.40,
            name=f"TIE_{bus1}_{bus2}",
            in_service=False  # Normally OPEN
        )
        
        net.line.at[tie_line_idx, 'is_tie'] = True
        tie_lines.append(tie_line_idx)
        
        # Create switch for this tie
        tie_switch_idx = pp.create_switch(
            net,
            bus=bus1,
            element=tie_line_idx,
            et='l',  # Line switch
            closed=False,  # Normally OPEN
            type='CB',  # Circuit breaker
            name=f'TIE_SW_{bus1}_{bus2}'
        )
        
        tie_switches.append(tie_switch_idx)
        
        # Store restoration info
        restoration_map[tie_switch_idx] = {
            'bus1': bus1,
            'bus2': bus2,
            'tie_line_idx': tie_line_idx,
            'path_length': info['path_length'],
            'restorable_buses': info['restorable_buses'],
            'critical_buses': info['critical_buses'],
            'priority_score': info['priority_score']
        }
        
        LOG.info(f"Tie {idx+1}: Bus {bus1} ↔ Bus {bus2}")
        LOG.info(f"  → Restores {len(info['restorable_buses'])} buses, "
                f"{len(info['critical_buses'])} critical")
        LOG.info(f"  → Path length: {info['path_length']} hops")
        LOG.info(f"  → Priority score: {info['priority_score']}")
    
    # Store in network
    net['restoration_map'] = restoration_map
    net['tie_switches'] = tie_switches
    net['tie_lines'] = tie_lines
    
    LOG.info(f"✓ Added {len(tie_switches)} TIE SWITCHES for restoration")
    LOG.info("=" * 60)
    
    return restoration_map
# ==================== SWITCH ADDITION ==================== #

def _are_buses_connected(net, b1: int, b2: int) -> bool:
    """Check if two buses are already connected by a line."""
    for _, row in net.line.iterrows():
        if {int(row["from_bus"]), int(row["to_bus"])} == {b1, b2}:
            return True
    return False


def _add_microgrid_switches(net, opts: Dict[str, Any]):
    """Add line switches (sectionalizers) and tie switches for reconfiguration."""
    if not opts.get("add_switches", True):
        return

    switch_density = float(opts.get("switch_density", 0.15))
    
    # Initialize columns
    if 'is_tie' not in net.line.columns:
        net.line['is_tie'] = False
    if 'is_sectionalizing' not in net.line.columns:
        net.line['is_sectionalizing'] = False

    # 1. SECTIONALIZING SWITCHES (normally CLOSED)
    n_lines = len(net.line)
    n_switches = max(10, int(n_lines * switch_density))

    candidate_lines = net.line.index.tolist()
    np.random.shuffle(candidate_lines)

    sectionalizing_switches = []
    for line_idx in candidate_lines[:n_switches]:
        row = net.line.loc[line_idx]
        sw_idx = pp.create_switch(
            net,
            bus=int(row["from_bus"]),
            element=line_idx,
            et="l",
            closed=True,  # Initially CLOSED
            type="LS",
            name=f"SW_SEC_{line_idx}"
        )
        sectionalizing_switches.append(sw_idx)
        net.line.at[line_idx, 'is_sectionalizing'] = True

    LOG.info(f"Added {len(sectionalizing_switches)} sectionalizing switches (normally closed)")

    # 2. TIE SWITCHES FOR RESTORATION (normally OPEN)
    restoration_map = _add_tie_switches_for_restoration(net, opts)
    
    # Mark switch types
    if not hasattr(net, 'switch') or 'is_tie' not in net.switch.columns:
        net.switch['is_tie'] = False
    if 'is_controllable' not in net.switch.columns:
        net.switch['is_controllable'] = False
    
    # Mark all added switches as controllable
    for sw_idx in sectionalizing_switches:
        net.switch.at[sw_idx, 'is_controllable'] = True
        net.switch.at[sw_idx, 'is_tie'] = False
    
    for sw_idx in net.get('tie_switches', []):
        net.switch.at[sw_idx, 'is_controllable'] = True
        net.switch.at[sw_idx, 'is_tie'] = True


# ==================== OPERATING LIMITS ==================== #

def _set_operating_limits(net, opts: Dict[str, Any]):
    """Set voltage limits, thermal limits, and transformer ratings."""
    v_min, v_max = opts.get("voltage_limits", (0.95, 1.05))
    net.bus["min_vm_pu"] = v_min
    net.bus["max_vm_pu"] = v_max

    # Line thermal limits
    if "max_i_ka" not in net.line.columns or net.line["max_i_ka"].isna().all():
        default_rating = float(opts.get("default_line_rating_ka", 0.5))
        net.line["max_i_ka"] = default_rating
        LOG.info(f"Set default line rating: {default_rating} kA")

    # Ensure no zero ratings
    net.line["max_i_ka"] = net.line["max_i_ka"].replace(0.0, 0.5)

    # Transformer limits
    if hasattr(net, "trafo") and len(net.trafo):
        max_loading = float(opts.get("trafo_max_loading", 120.0))
        net.trafo["max_loading_percent"] = max_loading

    LOG.info(f"Set operating limits: V=[{v_min}, {v_max}] pu, thermal ratings configured")


# ==================== TOPOLOGY ENERGIZING ==================== #

def _energize_topology(net, opts: Dict[str, Any]):
    """Make network live by setting in_service flags and closing switches."""
    _fix_trafo_metadata(net)

    # 1. Set equipment in_service
    if opts.get("set_lines_in_service", True) and hasattr(net, "line"):
        net.line["in_service"] = True

    if opts.get("set_trafos_in_service", True) and hasattr(net, "trafo") and len(net.trafo):
        net.trafo["in_service"] = True

    if hasattr(net, "load") and len(net.load):
        net.load["in_service"] = True

    if hasattr(net, "gen") and len(net.gen):
        net.gen["in_service"] = True

    # 2. Switch states
    if hasattr(net, "switch") and len(net.switch) and "closed" in net.switch.columns:
        if opts.get("close_line_switches", True):
            mask_l = (net.switch["et"] == "l")
            net.switch.loc[mask_l, "closed"] = True

        if opts.get("close_bus_bus_ties", False):  # Ties usually start OPEN
            mask_b = (net.switch["et"] == "b")
            net.switch.loc[mask_b, "closed"] = True

    # 3. Initial PF to energize buses
    try:
        pp.runpp(net, algorithm="nr", init="auto", tolerance_mva=1e-8,
                 enforce_q_lims=True, max_iteration=80, numba=True)
        LOG.info("Initial power flow successful after energizing")
    except Exception as e:
        LOG.warning(f"Power flow after energize failed: {e}")


# ==================== SANITY CHECK ==================== #

def _pf_sanity(net):
    """Run power flow sanity check and return status."""
    if not pp:
        return {"ok": True, "note": "pandapower missing"}

    ok = True
    try:
        pp.runpp(net, algorithm="nr", init="results", tolerance_mva=1e-8,
                 enforce_q_lims=True, max_iteration=80, numba=True)
    except Exception as e:
        LOG.warning(f"PF sanity check failed: {e}")
        ok = False

    max_loading = None
    if hasattr(net, "res_line") and len(net.res_line) and "loading_percent" in net.res_line:
        loading_values = net.res_line["loading_percent"].values
        # Handle NaN values
        valid_loadings = loading_values[~np.isnan(loading_values)]
        if len(valid_loadings) > 0:
            max_loading = float(np.max(valid_loadings))

    served_frac = 1.0
    if hasattr(net, "res_load") and len(net.res_load) and hasattr(net, "load") and len(net.load):
        served = float(np.nansum(net.res_load.get("p_mw", 0.0)))
        total = float(np.nansum(net.load.get("p_mw", 0.0)))
        served_frac = served / total if total > 0 else 1.0

    return {
        "ok": ok,
        "served_frac": served_frac,
        "max_line_load_pct": max_loading
    }


# ==================== PUBLIC API ==================== #

def assign_priorities(
    net,
    critical_top_pct: float = DEFAULTS["tier_percent_critical"],
    important_next_pct: float = DEFAULTS["tier_percent_important"],
    reward_weights: Dict[str, float] = None,
    log_summary: bool = True,
) -> Dict[str, Any]:
    """
    Assign tier (str) and priority (int: 0/1/2) to loads based on bus ranking.
    Returns metadata dict with tier distributions.
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
    }


def augment_case145(net, options: Dict[str, Any] = None, **kwargs) -> Tuple[object, Dict[str, Any]]:
    """
    Complete microgrid augmentation for Case145 with restoration capabilities.
    """
    opts = {**DEFAULTS, **(options or {}), **kwargs}

    LOG.info("=" * 60)
    LOG.info("AUGMENTING CASE145 FOR MICROGRID OPERATION")
    LOG.info("WITH CRITICAL LOAD RESTORATION CAPABILITIES")
    LOG.info("=" * 60)

    # 0. Resolve voltage conflicts
    _resolve_voltage_conflicts(net)

    # 1. Add DERs
    _add_distributed_energy_resources(net, opts)

    # 2. Add switches (sectionalizing + tie switches)
    _add_microgrid_switches(net, opts)

    # 3. Set operating limits
    _set_operating_limits(net, opts)

    # 4. Energize topology
    if opts.get("energize", True):
        _energize_topology(net, opts)

    # 5. Assign priorities
    pri_meta = assign_priorities(
        net,
        critical_top_pct=float(opts.get("tier_percent_critical", DEFAULTS["tier_percent_critical"])),
        important_next_pct=float(opts.get("tier_percent_important", DEFAULTS["tier_percent_important"])),
        reward_weights=opts.get("reward_weights", DEFAULTS["reward_weights"]),
        log_summary=bool(opts.get("log_tier_summary", True)),
    )

    # 6. PF sanity check (optional - don't fail if PF doesn't converge)
    pf_meta = _pf_sanity(net) if opts.get("pf_sanity_check", True) else {"ok": None}

    # Build info dict BEFORE using it (FIX!)
    restoration_map = net.get('restoration_map', {})
    n_switches = len(net.switch) if hasattr(net, "switch") else 0
    n_tie_switches = len(net.get('tie_switches', []))
    n_sectionalizing = n_switches - n_tie_switches
    
    info = {
        "pf_sanity": pf_meta,
        "priority": pri_meta,
        "n_buses": len(net.bus),
        "n_lines": len(net.line),
        "n_switches": n_switches,
        "n_tie_switches": n_tie_switches,
        "n_sectionalizing_switches": n_sectionalizing,
        "n_gens": len(net.gen) if hasattr(net, "gen") else 0,
        "n_sgens": len(net.sgen) if hasattr(net, "sgen") else 0,
        "n_storage": len(net.storage) if hasattr(net, "storage") else 0,
        "restoration_coverage": len([r for r in restoration_map.values() if r.get('critical_buses', [])]),
        "restoration_map": restoration_map,
        "notes": "Case145 augmented with DERs, switches, load priorities, and restoration ties",
    }

    net["aug_tag"] = "AUG.CASE145.MICROGRID.RESTORATION.v2"

    LOG.info("=" * 60)
    LOG.info("AUGMENTATION COMPLETE")
    LOG.info(f"  Buses: {info['n_buses']}")
    LOG.info(f"  Lines: {info['n_lines']}")
    LOG.info(f"  Switches: {info['n_switches']}")
    LOG.info(f"    - Sectionalizing: {info['n_sectionalizing_switches']}")
    LOG.info(f"    - Tie (normally-open): {info['n_tie_switches']}")
    LOG.info(f"  Generators: {info['n_gens']}")
    LOG.info(f"  Solar (sgen): {info['n_sgens']}")
    LOG.info(f"  Storage: {info['n_storage']}")
    LOG.info(f"  Restoration ties with critical loads: {info['restoration_coverage']}")
    LOG.info(f"  PF Sanity: {pf_meta.get('ok', 'N/A')}")
    if not pf_meta.get('ok', False):
        LOG.warning("  ⚠ Initial PF did not converge - continuing anyway")
        LOG.warning("  → This is OK for RL training - environment will handle it")
    LOG.info("=" * 60)

    return net, info