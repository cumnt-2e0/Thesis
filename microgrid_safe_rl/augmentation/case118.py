#!/usr/bin/env python3
"""
Augment IEEE 118-bus network for RL microgrid control.

SCOPE: Network augmentation only
- Remove ext_grid and create internal slack generator
- Add DERs (PV + BESS)
- Add switches (sectionalizing + ties)
- Assign load priorities
- Keep network HEALTHY and BALANCED

Environment will handle:
- Stress application
- Fault injection
- Cascade triggering
"""

import logging
import pandapower as pp
import pandapower.networks as pn
import numpy as np
from typing import Dict, Any, Tuple

try:
    from pandapower.topology import create_nxgraph
    import networkx as nx
    HAS_NETWORKX = True
except:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)

DEFAULTS = {
    "add_ders": True,
    "der_penetration": 0.20,
    "pv_sizing_factor": (0.3, 0.5),
    "bess_sizing_factor": (0.2, 0.4),
    "bess_initial_soc": 50.0,
    
    "add_switches": True,
    "switch_density": 0.15,
    "n_tie_switches": 10,
    
    "tier_percent_critical": 0.10,
    "tier_percent_important": 0.20,
    
    "voltage_limits": (0.95, 1.05),
    "default_line_rating_ka": 0.5,
}


def _identify_critical_buses(net, pct_critical=0.10, pct_important=0.20):
    """Rank buses by load and assign priorities."""
    load_by_bus = net.load.groupby('bus')['p_mw'].sum()
    sorted_buses = load_by_bus.sort_values(ascending=False)
    
    n_buses_with_load = len(sorted_buses)
    n_critical = max(1, int(pct_critical * n_buses_with_load))
    n_important = int(pct_important * n_buses_with_load)
    
    critical_buses = set(sorted_buses.index[:n_critical])
    important_buses = set(sorted_buses.index[n_critical:n_critical + n_important])
    
    net.load['priority'] = 2
    net.load['tier'] = 'normal'
    
    for idx, row in net.load.iterrows():
        bus = row.bus
        if bus in critical_buses:
            net.load.at[idx, 'priority'] = 0
            net.load.at[idx, 'tier'] = 'critical'
        elif bus in important_buses:
            net.load.at[idx, 'priority'] = 1
            net.load.at[idx, 'tier'] = 'important'
    
    n_crit_loads = (net.load.priority == 0).sum()
    n_imp_loads = (net.load.priority == 1).sum()
    n_norm_loads = (net.load.priority == 2).sum()
    
    p_crit = net.load[net.load.priority == 0].p_mw.sum()
    p_imp = net.load[net.load.priority == 1].p_mw.sum()
    p_norm = net.load[net.load.priority == 2].p_mw.sum()
    
    logger.info("=== LOAD PRIORITIES ===")
    logger.info(f"  Critical:    {n_crit_loads:2d} loads, {p_crit:7.1f} MW")
    logger.info(f"  Important:   {n_imp_loads:2d} loads, {p_imp:7.1f} MW")
    logger.info(f"  Normal:      {n_norm_loads:2d} loads, {p_norm:7.1f} MW")
    
    return {
        'critical_buses': list(critical_buses),
        'important_buses': list(important_buses),
        'n_critical_loads': n_crit_loads,
        'p_critical': p_crit,
    }


def _add_distributed_ders(net, opts):
    """Add PV and BESS (NO stress applied here)."""
    if not opts.get("add_ders", True):
        return
    
    penetration = opts['der_penetration']
    
    load_by_bus = net.load.groupby('bus')['p_mw'].sum()
    n_der_buses = max(5, int(len(load_by_bus) * penetration))
    der_buses = load_by_bus.nlargest(n_der_buses).index.tolist()
    
    logger.info(f"Adding DERs to {len(der_buses)} buses ({penetration:.0%} penetration)")
    
    pv_min, pv_max = opts['pv_sizing_factor']
    bess_min, bess_max = opts['bess_sizing_factor']
    initial_soc = opts['bess_initial_soc']
    
    total_pv_capacity = 0.0
    total_bess_capacity = 0.0
    
    for bus in der_buses:
        local_load = float(load_by_bus[bus])
        
        pv_capacity = local_load * np.random.uniform(pv_min, pv_max)
        pv_output = pv_capacity * 0.7
        
        pp.create_sgen(
            net, bus=bus,
            p_mw=pv_output, q_mvar=0.0,
            type="PV", controllable=True,
            max_p_mw=pv_capacity, min_p_mw=0.0,
            max_q_mvar=pv_capacity * 0.3,
            min_q_mvar=-pv_capacity * 0.3,
            in_service=True,
            name=f"PV_{bus}"
        )
        total_pv_capacity += pv_capacity
        
        bess_capacity = local_load * np.random.uniform(bess_min, bess_max)
        bess_energy = bess_capacity * 2.0
        
        pp.create_storage(
            net, bus=bus,
            p_mw=0.0,
            max_e_mwh=bess_energy,
            soc_percent=initial_soc,
            min_p_mw=-bess_capacity,
            max_p_mw=bess_capacity,
            controllable=True,
            in_service=True,
            name=f"BESS_{bus}"
        )
        total_bess_capacity += bess_capacity
    
    logger.info(f"DERs added: {len(der_buses)} PV + {len(der_buses)} BESS")
    logger.info(f"  PV capacity: {total_pv_capacity:.1f} MW")
    logger.info(f"  BESS capacity: {total_bess_capacity:.1f} MW")
    
    return {
        'n_pv': len(der_buses),
        'n_bess': len(der_buses),
        'pv_capacity_mw': total_pv_capacity,
        'bess_capacity_mw': total_bess_capacity,
    }


def _find_tie_switch_candidates(net, n_ties=10):
    """Find strategic tie switch locations."""
    if not HAS_NETWORKX:
        logger.warning("NetworkX not available - using heuristic tie placement")
        buses = list(net.bus.index)
        step = len(buses) // (n_ties + 1)
        ties = []
        for i in range(n_ties):
            b1 = buses[i * step]
            b2 = buses[min((i + 1) * step + step//2, len(buses)-1)]
            ties.append({
                'bus1': b1, 'bus2': b2, 'path_length': 4,
                'path': [b1, b2], 'critical_count': 0, 'score': 10
            })
        return ties
    
    try:
        G = create_nxgraph(net, respect_switches=False)
    except:
        return []
    
    critical_buses = set()
    if 'priority' in net.load.columns:
        crit_loads = net.load[net.load.priority == 0]
        critical_buses = set(crit_loads.bus.values)
    
    buses = list(net.bus.index)
    candidates = []
    
    np.random.seed(42)
    sample_buses = np.random.choice(buses, size=min(100, len(buses)), replace=False) if len(buses) > 150 else buses
    
    for i, b1 in enumerate(sample_buses):
        for b2 in sample_buses[i+1:]:
            if G.has_edge(b1, b2):
                continue
            
            try:
                path = nx.shortest_path(G, b1, b2)
                path_length = len(path) - 1
                
                if 3 <= path_length <= 6:
                    critical_count = sum(1 for b in path if b in critical_buses)
                    score = critical_count * 10 + (10 - path_length)
                    
                    candidates.append({
                        'bus1': b1, 'bus2': b2,
                        'path_length': path_length,
                        'path': path,
                        'critical_count': critical_count,
                        'score': score
                    })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:n_ties]


def _add_switches(net, opts):
    """Add sectionalizing and tie switches."""
    if not opts.get("add_switches", True):
        return
    
    net.line['is_tie'] = False
    net.line['is_sectionalizing'] = False
    
    # Sectionalizing switches
    switch_density = opts['switch_density']
    n_switches = max(10, int(len(net.line) * switch_density))
    
    candidate_lines = list(net.line.index)
    np.random.shuffle(candidate_lines)
    
    sectionalizing_switches = []
    for line_idx in candidate_lines[:n_switches]:
        line = net.line.loc[line_idx]
        sw_idx = pp.create_switch(
            net, bus=line.from_bus, element=line_idx, et='l',
            closed=True, type='LS', name=f"SW_SEC_{line_idx}"
        )
        sectionalizing_switches.append(sw_idx)
        net.line.at[line_idx, 'is_sectionalizing'] = True
    
    logger.info(f"Added {len(sectionalizing_switches)} sectionalizing switches")
    
    # Tie switches
    n_ties = opts['n_tie_switches']
    tie_candidates = _find_tie_switch_candidates(net, n_ties)
    
    tie_switches = []
    tie_lines = []
    restoration_map = {}
    
    logger.info("Adding tie switches for restoration")
    
    for idx, tie_info in enumerate(tie_candidates):
        bus1, bus2 = tie_info['bus1'], tie_info['bus2']
        
        tie_line_idx = pp.create_line_from_parameters(
            net, from_bus=bus1, to_bus=bus2,
            length_km=0.5, r_ohm_per_km=0.4, x_ohm_per_km=0.4,
            c_nf_per_km=3.0, max_i_ka=opts['default_line_rating_ka'],
            name=f"TIE_{bus1}_{bus2}", in_service=False
        )
        net.line.at[tie_line_idx, 'is_tie'] = True
        tie_lines.append(tie_line_idx)
        
        tie_sw_idx = pp.create_switch(
            net, bus=bus1, element=tie_line_idx, et='l',
            closed=False, type='CB', name=f"TIE_SW_{bus1}_{bus2}"
        )
        tie_switches.append(tie_sw_idx)
        
        restorable_buses = tie_info.get('path', [])[1:-1]
        critical_buses = [b for b in restorable_buses if b in 
                         (net.load[net.load.priority == 0].bus.values if 'priority' in net.load.columns else [])]
        
        restoration_map[tie_sw_idx] = {
            'bus1': bus1, 'bus2': bus2,
            'tie_line_idx': tie_line_idx,
            'path_length': tie_info['path_length'],
            'restorable_buses': restorable_buses,
            'critical_buses': critical_buses,
            'critical_count': tie_info['critical_count'],
            'score': tie_info['score']
        }
    
    net['restoration_map'] = restoration_map
    net['tie_switches'] = tie_switches
    net['tie_lines'] = tie_lines
    
    logger.info(f"Added {len(tie_switches)} tie switches")
    
    # Mark controllability
    if 'is_controllable' not in net.switch.columns:
        net.switch['is_controllable'] = False
    if 'is_tie' not in net.switch.columns:
        net.switch['is_tie'] = False
    
    for sw_idx in sectionalizing_switches:
        net.switch.at[sw_idx, 'is_controllable'] = True
        net.switch.at[sw_idx, 'is_tie'] = False
    
    for sw_idx in tie_switches:
        net.switch.at[sw_idx, 'is_controllable'] = True
        net.switch.at[sw_idx, 'is_tie'] = True
    
    return {
        'n_sectionalizing': len(sectionalizing_switches),
        'n_ties': len(tie_switches),
        'restoration_map': restoration_map
    }


def ensure_single_internal_slack(net, slack_bus=None, p_cap_mw=1500.0, vm_set=1.0):
    """
    Remove ext_grid and create internal slack generator.
    
    This makes the network a true islanded microgrid with internal generation
    acting as the slack bus for power flow calculations.
    
    Args:
        net: Pandapower network
        slack_bus: Bus index for slack (None = auto-select)
        p_cap_mw: Slack generator capacity
        vm_set: Voltage setpoint
    
    Returns:
        int: Index of slack generator
    """
    logger.info("Configuring internal slack generator...")
    
    # 1. Remove external grid (island the system)
    if len(net.ext_grid):
        slack_bus_from_ext_grid = int(net.ext_grid.loc[0, 'bus'])
        net.ext_grid.drop(net.ext_grid.index, inplace=True)
        logger.info(f"  Removed ext_grid from bus {slack_bus_from_ext_grid}")
        
        # Use the former ext_grid bus as slack if not specified
        if slack_bus is None:
            slack_bus = slack_bus_from_ext_grid
    
    # 2. Add 'slack' column to gen if missing
    if 'slack' not in net.gen.columns:
        net.gen['slack'] = False
    
    # 3. Choose a slack bus if still None
    if slack_bus is None:
        if len(net.gen):
            # Pick the generator with largest capacity
            if 'max_p_mw' in net.gen.columns and net.gen['max_p_mw'].notna().any():
                cand = net.gen.loc[net.gen['max_p_mw'].fillna(-1e9).idxmax()]
                slack_bus = int(cand.bus)
            else:
                slack_bus = int(net.gen.iloc[0].bus)
        else:
            slack_bus = 0  # fallback to bus 0
    
    # 4. Clear any existing slack flags
    net.gen['slack'] = False
    
    # 5. Create or configure slack generator
    same_bus_gens = net.gen[net.gen.bus == slack_bus]
    
    if len(same_bus_gens) > 0:
        # Use existing generator at slack_bus
        gidx = same_bus_gens.index[0]
        net.gen.at[gidx, 'slack'] = True
        net.gen.at[gidx, 'in_service'] = True
        net.gen.at[gidx, 'vm_pu'] = vm_set
        
        # Set generous limits for slack
        net.gen.at[gidx, 'min_q_mvar'] = -1e4
        net.gen.at[gidx, 'max_q_mvar'] = 1e4
        net.gen.at[gidx, 'min_p_mw'] = -p_cap_mw
        net.gen.at[gidx, 'max_p_mw'] = p_cap_mw
        
        logger.info(f"  Using existing gen[{gidx}] at bus {slack_bus} as slack")
    else:
        # Create new internal slack generator
        gidx = pp.create_gen(
            net,
            bus=slack_bus,
            p_mw=0.0,
            vm_pu=vm_set,
            name="INTERNAL_SLACK",
            controllable=True,
            slack=True,
            min_q_mvar=-1e4,
            max_q_mvar=1e4,
            min_p_mw=-p_cap_mw,
            max_p_mw=p_cap_mw,
        )
        logger.info(f"  Created new slack gen[{gidx}] at bus {slack_bus}")
    
    # 6. Verify single slack
    n_slack = (net.gen['slack'] == True).sum()
    assert n_slack == 1, f"Expected 1 slack gen, found {n_slack}"
    
    logger.info(f"  SLACK CONFIGURATION:")
    logger.info(f"    - Slack gen index: {gidx}")
    logger.info(f"    - Slack bus: {slack_bus}")
    logger.info(f"    - Capacity: {p_cap_mw} MW")
    logger.info(f"    - Voltage setpoint: {vm_set} pu")
    logger.info(f"    - Non-slack gens: {len(net.gen) - 1}")
    
    return int(gidx)


def augment_case118(net=None, opts=None):
    """
    Augment IEEE 118-bus for RL microgrid control.
    
    Creates a HEALTHY, BALANCED islanded microgrid with:
    - Internal slack generator (no ext_grid)
    - Distributed energy resources (PV + BESS)
    - Sectionalizing and tie switches
    - Hierarchical load priorities
    
    Environment will apply stress and inject faults during episodes.
    
    Args:
        net: Existing network (None = create from case118)
        opts: Configuration dict (None = use defaults)
    
    Returns:
        tuple: (net, info_dict)
    """
    if net is None:
        net = pn.case118()
    
    opts = {**DEFAULTS, **(opts or {})}
    
    logger.info("=" * 60)
    logger.info("AUGMENTING CASE118 FOR RL MICROGRID CONTROL")
    logger.info("=" * 60)
    
    # 1. Configure internal slack (CRITICAL: Do this first)
    slack_gen_idx = ensure_single_internal_slack(net, slack_bus=None)
    slack_bus = int(net.gen.loc[slack_gen_idx, 'bus'])
    
    # 2. Assign load priorities
    priority_info = _identify_critical_buses(
        net, opts['tier_percent_critical'], opts['tier_percent_important']
    )
    
    # 3. Add DERs
    der_info = _add_distributed_ders(net, opts)
    
    # 4. Add switches
    switch_info = _add_switches(net, opts)
    
    # 5. Set limits
    net.bus['min_vm_pu'] = opts['voltage_limits'][0]
    net.bus['max_vm_pu'] = opts['voltage_limits'][1]
    
    if 'max_i_ka' not in net.line.columns or net.line.max_i_ka.isna().all():
        net.line['max_i_ka'] = opts['default_line_rating_ka']
    
    # 6. Validate network health
    logger.info("Validating network configuration...")
    try:
        pp.runpp(net, algorithm='nr', max_iteration=30)
        pf_ok = net.converged
        v_min = net.res_bus.vm_pu.min()
        v_max = net.res_bus.vm_pu.max()
        max_loading = net.res_line.loading_percent.max()
        
        logger.info(f"  Power flow: CONVERGED")
        logger.info(f"  Voltage range: [{v_min:.3f}, {v_max:.3f}] pu")
        logger.info(f"  Max line loading: {max_loading:.1f}%")
    except Exception as e:
        pf_ok = False
        v_min = v_max = None
        max_loading = None
        logger.warning(f"  Power flow: FAILED - {e}")
    
    # 7. Build metadata
    total_load = net.load.p_mw.sum()
    total_gen_capacity = net.gen.max_p_mw.sum() if 'max_p_mw' in net.gen.columns else net.gen.p_mw.sum() * 1.5
    
    info = {
        'name': 'case118_microgrid',
        'n_buses': len(net.bus),
        'n_lines': len(net.line),
        'n_loads': len(net.load),
        'n_generators': len(net.gen),
        'n_sgen': len(net.sgen),
        'n_storage': len(net.storage),
        'n_switches': len(net.switch) if hasattr(net, 'switch') else 0,
        'n_sectionalizing_switches': switch_info['n_sectionalizing'] if switch_info else 0,
        'n_tie_switches': switch_info['n_ties'] if switch_info else 0,
        'n_pv': der_info['n_pv'] if der_info else 0,
        'n_bess': der_info['n_bess'] if der_info else 0,
        'pv_capacity_mw': der_info['pv_capacity_mw'] if der_info else 0,
        'bess_capacity_mw': der_info['bess_capacity_mw'] if der_info else 0,
        'total_load_mw': total_load,
        'total_gen_capacity_mw': total_gen_capacity,
        'n_critical_loads': priority_info['n_critical_loads'],
        'n_important_loads': len(net.load[net.load.priority == 1]),
        'n_normal_loads': len(net.load[net.load.priority == 2]),
        'p_critical_mw': priority_info['p_critical'],
        'p_important_mw': net.load[net.load.priority == 1].p_mw.sum(),
        'p_normal_mw': net.load[net.load.priority == 2].p_mw.sum(),
        'slack_bus': slack_bus,
        'slack_gen_idx': slack_gen_idx,
        'restoration_map': switch_info['restoration_map'] if switch_info else {},
        'pf_converged': pf_ok,
        'voltage_range': (v_min, v_max),
        'max_line_loading_pct': max_loading,
    }
    
    net['aug_tag'] = 'AUG.CASE118.v1'
    
    logger.info("=" * 60)
    logger.info("AUGMENTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Network: {info['n_buses']} buses, {info['n_lines']} lines")
    logger.info(f"Slack: gen[{info['slack_gen_idx']}] at bus {info['slack_bus']}")
    logger.info(f"Generators: {info['n_generators']} total (1 slack + {info['n_generators']-1} non-slack)")
    logger.info(f"DERs: {info['n_pv']} PV ({info['pv_capacity_mw']:.1f} MW) + {info['n_bess']} BESS ({info['bess_capacity_mw']:.1f} MW)")
    logger.info(f"Switches: {info['n_sectionalizing_switches']} sect + {info['n_tie_switches']} ties")
    logger.info(f"Loads: {info['n_critical_loads']} critical, {info['n_important_loads']} important, {info['n_normal_loads']} normal")
    logger.info(f"Power Flow: {'CONVERGED' if pf_ok else 'FAILED'}")
    logger.info("=" * 60)
    
    return net, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "=" * 60)
    print("TESTING CASE118 AUGMENTATION")
    print("=" * 60 + "\n")
    
    net, info = augment_case118()
    
    print("\n" + "=" * 60)
    print("VERIFICATION CHECKS")
    print("=" * 60)
    
    # Check 1: No ext_grid
    print(f"\n1. External Grid Removed:")
    print(f"   ext_grid count: {len(net.ext_grid)}")
    print(f"   Status: {'PASS' if len(net.ext_grid) == 0 else 'FAIL'}")
    
    # Check 2: Single slack generator
    print(f"\n2. Slack Generator:")
    slack_gens = net.gen[net.gen['slack'] == True]
    print(f"   Slack generators: {len(slack_gens)}")
    print(f"   Status: {'PASS' if len(slack_gens) == 1 else 'FAIL'}")
    
    if len(slack_gens) == 1:
        slack_idx = slack_gens.index[0]
        print(f"   Slack gen index: {slack_idx}")
        print(f"   Slack bus: {info['slack_bus']}")
        print(f"   Capacity: {net.gen.loc[slack_idx, 'max_p_mw']:.1f} MW")
    
    # Check 3: Non-slack generators
    print(f"\n3. Non-Slack Generators:")
    non_slack = net.gen[net.gen['slack'] == False]
    print(f"   Non-slack generators: {len(non_slack)}")
    print(f"   Total generators: {len(net.gen)}")
    print(f"   Status: {'PASS' if len(non_slack) == len(net.gen) - 1 else 'FAIL'}")
    
    # Check 4: Network summary
    print(f"\n4. Network Summary:")
    print(f"   Buses: {info['n_buses']}")
    print(f"   Lines: {info['n_lines']}")
    print(f"   Loads: {info['n_loads']} ({info['n_critical_loads']} critical)")
    print(f"   PV: {info['n_pv']} ({info['pv_capacity_mw']:.1f} MW)")
    print(f"   BESS: {info['n_bess']} ({info['bess_capacity_mw']:.1f} MW)")
    print(f"   Switches: {info['n_switches']} ({info['n_tie_switches']} tie)")
    
    # Check 5: Power flow
    print(f"\n5. Power Flow:")
    print(f"   Converged: {info['pf_converged']}")
    if info['pf_converged']:
        print(f"   Voltage: [{info['voltage_range'][0]:.3f}, {info['voltage_range'][1]:.3f}] pu")
        print(f"   Max loading: {info['max_line_loading_pct']:.1f}%")
    print(f"   Status: {'PASS' if info['pf_converged'] else 'FAIL'}")
    
    # Overall
    all_passed = (
        len(net.ext_grid) == 0 and
        len(slack_gens) == 1 and
        len(non_slack) == len(net.gen) - 1
    )
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED - Ready for training!")
    else:
        print("SOME CHECKS FAILED - Review errors above")
    print("=" * 60 + "\n")