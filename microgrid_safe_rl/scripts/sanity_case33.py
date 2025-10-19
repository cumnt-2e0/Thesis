#!/usr/bin/env python3
# Sanity probe for case33bw PF + utilisation + existing cascade output.
# IMPORTANT: We do NOT add a new cascade engine here. We just read whatever
# your environment already emits (seed_fault, trip_this_step, etc.).

from __future__ import annotations
import argparse
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

import pandapower as pp
import pandapower.networks as pn

from microgrid_safe_rl.envs.microgrid_control_env import MicrogridControlEnv
from microgrid_safe_rl.augmentation.case33 import augment_case33

LOG = logging.getLogger("sanity.case33")
LOG.setLevel(logging.INFO)
LOG.addHandler(logging.StreamHandler())


def load_cfg(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_banner():
    print("=" * 80)
    print("=== CASE33BW ISLANDED MICROGRID SANITY CHECK ===")
    print("=" * 80)
    print("PF + derived line utilisation + disturbance events\n")


def derived_line_loading_percent(net) -> np.ndarray:
    if not hasattr(net, "res_line") or len(net.res_line) == 0:
        return np.array([])
    # S = sqrt(P^2 + Q^2) with P in MW, Q in MVAr -> MVA
    p = net.res_line.p_from_mw.to_numpy(dtype=float)
    q = net.res_line.q_from_mvar.to_numpy(dtype=float)
    s_mva = np.sqrt(p**2 + q**2)
    rating = net.line.get("rating_mva", pd.Series(np.full(len(net.line), 5.0))).to_numpy(dtype=float)
    rating = np.where(rating > 1e-6, rating, 5.0)
    return (s_mva / rating) * 100.0


def describe_network(env: MicrogridControlEnv):
    net = env.net

    # Ensure PF results present (explicit options to get branch results)
    pp.set_user_pf_options(net, init="flat", enforce_q_lims=True,
                           calculate_voltage_angles=True,
                           only_v_results=False, check_connectivity=True,
                           trafo_model="t", recycle=None)
    try:
        pp.runpp(net)
    except Exception as e:
        print(f"PF failed during describe_network(): {e}")

    print("NETWORK TOPOLOGY:")
    buses = len(net.bus) if hasattr(net, "bus") else 0
    lines = len(net.line) if hasattr(net, "line") else 0
    switches = len(net.switch) if hasattr(net, "switch") else 0
    closed = int(net.switch["closed"].sum()) if switches else 0
    open_ = switches - closed
    print(f"  Buses: {buses}")
    print(f"  Lines: {lines}")
    print(f"  Switches: {switches} ({closed} closed, {open_} open)\n")

    print("POWER SOURCES (results):")
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        slack_bus = int(net.ext_grid.bus.iloc[0])
        p_slack = float(net.res_ext_grid.p_mw.sum()) if hasattr(net, "res_ext_grid") else 0.0
        print(f"  Ext_grid (slack): bus {slack_bus}, supplying {p_slack:.2f} MW")
    if hasattr(net, "res_gen") and len(net.res_gen):
        print(f"  Genset: {float(net.res_gen.p_mw.sum()):.2f} MW")
    if hasattr(net, "res_sgen") and len(net.res_sgen):
        p_pv = float(net.res_sgen.p_mw.sum())
        n_pv = len(net.sgen) if hasattr(net, "sgen") else 0
        print(f"  PV: {n_pv} units, {p_pv:.2f} MW")
    if hasattr(net, "storage") and len(net.storage):
        e_total = float(net.storage.max_e_mwh.sum())
        print(f"  BESS: {len(net.storage)} units, {e_total:.2f} MWh")

    # DER unavailability info
    if hasattr(env, '_der_bound_frac_ep') and env.der_cfg.get('enabled', False):
        print(f"\nDER UNAVAILABILITY:")
        print(f"  Episode generation bound: {env._der_bound_frac_ep:.1%} of load")
        print(f"  Implied deficit: {(1 - env._der_bound_frac_ep):.1%}")

    total_nameplate = float(net.load.p_mw.sum()) if hasattr(net, "load") and len(net.load) else 0.0
    served = float(net.res_load.p_mw.sum()) if hasattr(net, "res_load") and len(net.res_load) else 0.0
    print("\nLOADS:")
    print(f"  Total (nameplate): {total_nameplate:.2f} MW (points: {len(net.load) if hasattr(net,'load') else 0})")
    if hasattr(net.load, 'priority'):
        crit = float(net.load[net.load.priority == 0].p_mw.sum())
        imp = float(net.load[net.load.priority == 1].p_mw.sum())
        norm = float(net.load[net.load.priority == 2].p_mw.sum())
        print(f"    - Critical: {crit:.2f} MW")
        print(f"    - Important: {imp:.2f} MW")
        print(f"    - Normal: {norm:.2f} MW")
    
    print("\nPOWER BALANCE (results):")
    p_slack = float(net.res_ext_grid.p_mw.sum()) if hasattr(net, "res_ext_grid") else 0.0
    p_gen   = float(net.res_gen.p_mw.sum()) if hasattr(net, "res_gen") else 0.0
    p_pv    = float(net.res_sgen.p_mw.sum()) if hasattr(net, "res_sgen") else 0.0
    supply  = p_slack + p_gen + p_pv
    print(f"  Supply: {supply:.2f} MW (Slack: {p_slack:.2f}, Genset: {p_gen:.2f}, PV: {p_pv:.2f})")
    print(f"  Demand (served): {served:.2f} MW")
    print(f"  Balance: {supply - served:+.2f} MW")
    
    if served < total_nameplate * 0.95:
        print(f"  ⚠️ LOAD CURTAILED: {total_nameplate - served:.2f} MW not served")

    # --- Line loading (derived) ---
    print("\nBASELINE LINE LOADING:")
    dlp = derived_line_loading_percent(net)
    if dlp.size:
        dmax = float(np.nanmax(dlp))
        dmean = float(np.nanmean(dlp))
        active = int((dlp > 0.05).sum())
        n_overload = int((dlp > 100).sum())
        print(f"  Derived (from P/Q & ratings): Max {dmax:.1f}%, Mean {dmean:.1f}%, Active lines {active}/{lines}")
        if n_overload > 0:
            print(f"  ⚠️ {n_overload} lines overloaded (>100%)")
        print("\n  Top 10 loaded lines (derived):")
        order = np.argsort(dlp)[::-1][:10]
        for i in order:
            print(f"    Line {int(i)}: {dlp[i]:.1f}%")
    else:
        print("  ⚠ No derived line loading available (no res_line?).")


def run_episode(env: MicrogridControlEnv, ep_idx: int, steps: int = 50) -> Dict[str, Any]:
    print("-" * 80)
    print(f"EPISODE {ep_idx+1}")
    print("-" * 80)

    obs, info = env.reset()

    # PF-derived served fraction
    net = env.net
    served_mw = float(net.res_load.p_mw.sum()) if hasattr(net, "res_load") and len(net.res_load) else 0.0
    nameplate = float(net.load.p_mw.sum()) if hasattr(net, "load") and len(net.load) else 1.0
    served_frac = served_mw / max(nameplate, 1e-9)
    max_derived = float(np.nanmax(derived_line_loading_percent(net))) if hasattr(net, "res_line") else 0.0

    print(f"RESET: PF={info.get('powerflow_success', True)}, served={served_frac:.3f}, "
          f"max_load(derived)≈{max_derived:.1f}%")

    # Show DER deficit if enabled
    if hasattr(env, '_der_bound_frac_ep') and env.der_cfg.get('enabled', False):
        print(f"DER deficit: {(1 - env._der_bound_frac_ep):.1%} "
              f"(bound={env._der_bound_frac_ep:.3f})")

    # Check for scheduled seed events
    if hasattr(env, '_events_by_step') and env._events_by_step:
        print("\nScheduled events:")
        for step in sorted(env._events_by_step.keys())[:5]:
            events = env._events_by_step.get(step, [])
            for ev in events:
                lines = ev.get('target', '?').replace('line:', '')
                print(f"  Step {step}: line {lines} {ev.get('op', 'trip')}")

    cascade_trips = 0
    total_trips = 0
    min_served = served_frac
    max_load = max_derived

    for t in range(steps):
        # No-op action (middle of action space)
        a = env.action_space.n // 2
        obs, r, done, trunc, info = env.step(a)

        # Check for events applied (your env's debug output)
        events_applied = info.get("events_applied", [])
        if events_applied:
            print(f"\nStep {t:02d}: Events: {', '.join(events_applied)}")
            
            # Count cascades vs seed faults
            for event in events_applied:
                if "cascade" in event.lower():
                    cascade_trips += 1
                    total_trips += 1
                elif "trip" in event:
                    total_trips += 1
            
            # Show updated state
            cur_max = float(np.nanmax(derived_line_loading_percent(env.net))) if hasattr(env.net, "res_line") else 0.0
            cur_served = info.get('served_total_frac', 0)
            print(f"  POST: PF={info.get('powerflow_success', True)}, "
                  f"served={cur_served:.3f}, "
                  f"max_load(derived)≈{cur_max:.1f}%")
            
            max_load = max(max_load, cur_max)
            min_served = min(min_served, cur_served)

        if done or trunc:
            break

    print(f"\nEpisode complete:")
    print(f"  Steps: {t+1}/{steps}")
    print(f"  Total trips: {total_trips} (cascade: {cascade_trips})")
    print(f"  Final served: {info.get('served_total_frac', 0):.3f}")
    print(f"  Min served: {min_served:.3f}")
    print(f"  Max line loading: {max_load:.1f}%")
    
    # Check for bound violations
    if hasattr(env, '_der_bound_frac_ep') and env.der_cfg.get('enabled', False):
        if min_served > env._der_bound_frac_ep + 0.01:
            print(f"  ⚠️ BOUND VIOLATION: served {min_served:.3f} > bound {env._der_bound_frac_ep:.3f}")
    
    return {
        "cascade_trips": cascade_trips,
        "total_trips": total_trips,
        "served": info.get("served_total_frac", 0),
        "min_served": min_served,
        "max_load": max_load,
    }


def main():
    parser = argparse.ArgumentParser(description="Sanity check for case33bw PF/utilisation and existing cascade")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--env_cfg", type=str, default="microgrid_safe_rl/configs/env_case33.yaml")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    print_banner()
    print("Creating environment...")

    # Load config
    cfg = load_cfg(args.env_cfg)
    cfg.setdefault("env_id", "case33")
    
    # Create base network
    base = pn.case33bw()
    net, _ = augment_case33(
        base,
        keep_slack=False,
        force_radial=True,
        target_pv_frac=0.22,
        target_bess_p_frac=0.10,
        target_bess_e_hours=1.0,
        vm_pu_slack=1.00,
        line_rating_mva=5.0,
        run_pf_after=True,
    )

    # Create environment
    env = MicrogridControlEnv(net, cfg)
    print("✓ Environment created\n")

    # Populate res_* then describe PF reality
    env.reset()
    describe_network(env)

    # Run episodes
    results = []
    for i in range(args.episodes):
        result = run_episode(env, i, steps=args.steps)
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Episodes run: {args.episodes}")
    
    if results:
        avg_trips = np.mean([r['total_trips'] for r in results])
        avg_cascades = np.mean([r['cascade_trips'] for r in results])
        avg_served = np.mean([r['min_served'] for r in results])
        max_loading = max(r['max_load'] for r in results)
        
        print(f"\nAverages:")
        print(f"  Line trips: {avg_trips:.1f} (cascades: {avg_cascades:.1f})")
        print(f"  Min served: {avg_served:.3f}")
        print(f"  Peak line loading: {max_loading:.1f}%")
        
        # Check if DER unavailability is working
        if any(hasattr(env, 'der_cfg') and env.der_cfg.get('enabled', False) for _ in range(1)):
            print(f"\nDER unavailability: {'ENABLED' if env.der_cfg.get('enabled', False) else 'DISABLED'}")
            if env.der_cfg.get('enabled', False):
                deficit_range = env.der_cfg.get('p_deficit_frac', [0, 0])
                print(f"  Deficit range: {deficit_range[0]:.0%} - {deficit_range[1]:.0%}")

    print("\nDone.")


if __name__ == "__main__":
    import sys
    sys.exit(main())