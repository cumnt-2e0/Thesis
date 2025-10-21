#!/usr/bin/env python3
# path: microgrid_safe_rl/scripts/sanity.py
import argparse
import numpy as np

from microgrid_safe_rl.utils.config import load_yaml
from microgrid_safe_rl.envs.factory import make_env

TIER_NAMES = {0: "critical", 1: "important", 2: "normal"}

def _fmt_or(env, attr, default):
    return getattr(env, attr, default)

def _print_hdr(env):
    print("=" * 80)
    print("=== CASCADE SANITY CHECK (Enhanced Diagnostics) ===")
    print("=" * 80)
    print("Seeds are random; subfaults rely on overload holds + proximity.\n")

    hop_limit        = _fmt_or(env, "cascade_hop_limit", 2)
    overload_min     = _fmt_or(env, "cascade_overload_min", 118.0)
    min_hold_steps   = _fmt_or(env, "cascade_min_hold", 2)
    per_wave_max     = _fmt_or(env, "per_wave_max", 1)
    max_additional   = _fmt_or(env, "max_additional", 4)
    n_seeds          = _fmt_or(env, "seed_n_seeds", 1)
    win_frac         = list(_fmt_or(env, "seed_step_window_frac", (0.05, 0.35)))
    excl_bridges     = _fmt_or(env, "seed_exclude_bridges", True)
    enabled          = _fmt_or(env, "cascade_enabled", True)
    lambda0          = _fmt_or(env, "lambda0", 0.10)
    rho0             = _fmt_or(env, "rho0", 0.90)

    print(f"CASCADE CONFIG:")
    print(f"  enabled={enabled}, hop_limit={hop_limit}, overload_min={overload_min:.1f}%")
    print(f"  min_hold={min_hold_steps}, per_wave_max={per_wave_max}, max_additional={max_additional}")
    print(f"  lambda0={lambda0:.3f}, rho0={rho0:.2f}")
    print(f"  seed: n={n_seeds}, window={win_frac}, exclude_bridges={excl_bridges}")

    # Local stress config
    cfg = getattr(env, "stress_local_cfg", None)
    if cfg:
        print(f"\nLOCAL STRESS CONFIG:")
        print(f"  enabled={cfg.get('enabled')}, hops={cfg.get('hops')}")
        print(f"  p_scale={cfg.get('p_scale'):.2f}x, q_scale={cfg.get('q_scale'):.2f}x")
        print(f"  line_rating_scale={cfg.get('line_rating_scale'):.2f}x")
        print(f"  ensure_gate={cfg.get('ensure_gate')}, gate_margin={cfg.get('gate_margin_pct')}%")
        print(f"  max_p_scale={cfg.get('max_p_scale')}x, min_line_scale={cfg.get('min_line_scale')}x")
    
    # DER config
    der_cfg = getattr(env, "der_cfg", None)
    if der_cfg and der_cfg.get("enabled"):
        print(f"\nDER UNAVAILABILITY CONFIG:")
        print(f"  deficit_frac={der_cfg.get('p_deficit_frac')}")
        print(f"  outage_prob={der_cfg.get('random_outage_prob', 0.0):.1%}")
        print(f"  max_disabled={der_cfg.get('max_disabled', 0)}")
    
    print("\n" + "=" * 80 + "\n")

def _pf_snapshot(env, label="SNAP", info=None):
    if info is None:
        info = {}

    pf_ok = bool(info.get("powerflow_success", False))
    served = info.get("served_total_frac", None)
    try:
        served = float(served) if served is not None else float("nan")
    except Exception:
        served = float("nan")

    n_bus_en = int(info.get("energized_buses", -1))

    # Line loading stats
    mx = float("nan")
    loaded_lines = int(info.get("loaded_lines", -1))
    above_100 = above_110 = above_120 = 0
    top_lines = []
    
    try:
        if hasattr(env.net, "res_line") and len(getattr(env.net, "res_line", [])):
            lp = np.asarray(env.net.res_line.loading_percent.values, dtype=float)
            lp = lp[np.isfinite(lp)]
            if lp.size:
                mx = float(np.max(lp))
                above_100 = int(np.sum(lp > 100))
                above_110 = int(np.sum(lp > 110))
                above_120 = int(np.sum(lp > 120))
                if loaded_lines < 0:
                    loaded_lines = int(np.sum(lp > 0.1))
                s = env.net.res_line.loading_percent.sort_values(ascending=False).head(15)
                top_lines = [(int(i), float(v)) for i, v in s.items()]
    except Exception:
        pass

    # Load stats
    try:
        P_now = float(env.net.load["p_mw"].sum()) if hasattr(env.net, "load") and "p_mw" in env.net.load else float("nan")
        if hasattr(env.net, "load") and "p_base_mw" in env.net.load:
            P_base = float(env.net.load["p_base_mw"].sum())
            d_pct = (100.0 * (P_now - P_base) / P_base) if P_base > 1e-9 else float("nan")
            p_str = f"{P_now:.1f}MW (base {P_base:.1f}MW, Δ={d_pct:+.1f}%)"
        else:
            p_str = f"{P_now:.1f}MW"
    except Exception:
        p_str = "nan"

    # DER bound
    der_bound = float(info.get("der_bound_frac", 1.0))

    # Stress state
    stress_active = bool(info.get("stress_active", getattr(env, "_local_stress_active", False)))
    if "stress_ttl" in info:
        stress_ttl = int(info["stress_ttl"])
    else:
        expiry = getattr(env, "_local_stress_expiry", None)
        if expiry is None:
            stress_ttl = 0
        else:
            stress_ttl = max(0, int(expiry - getattr(env, "current_step", 0)))

    print(f"{label}:")
    print(f"  PF={pf_ok}, served={served:.3f}, max_load={mx:.1f}%")
    print(f"  Lines: >100%={above_100}, >110%={above_110}, >120%={above_120}")
    print(f"  energized_buses={n_bus_en}, loaded_lines={loaded_lines}")
    print(f"  Load: {p_str}")
    print(f"  DER_bound={der_bound:.3f}, stress_active={stress_active}, ttl={stress_ttl}")
    
    if top_lines:
        top_str = "; ".join([f"L{i}:{v:.1f}%" for i, v in top_lines])
        print(f"  Top 15 lines: {top_str}")
    print()

def _eligible_neighbors(env):
    fn = getattr(env, "_eligible_neighbors", None)
    if fn is None:
        return []
    try:
        return fn()
    except Exception:
        return []

def _quick_prob_from_map(prob_map, loading_pct):
    p = 0.0
    for th, val in prob_map:
        if loading_pct >= th:
            p = float(val)
        else:
            break
    return float(max(0.0, min(1.0, p)))

def _step_debug(env, step_idx, candidates, applied, verbose=False):
    print(f"Step {step_idx:02d} • candidates: {len(candidates)}")
    if applied:
        print(f"         • applied: {applied}")

    if not verbose or not candidates:
        return

    prob_map = _fmt_or(env, "prob_loading_map", [(115.0, 1.0)])
    holds = _fmt_or(env, "_hold", {})
    min_hold = _fmt_or(env, "cascade_min_hold", 1)
    overload_min = _fmt_or(env, "cascade_overload_min", 110.0)
    
    if hasattr(env.net, "res_line") and len(getattr(env.net, "res_line", [])):
        # Show top candidates by loading
        stats = []
        for li in candidates[:30]:  # Top 30
            if li not in env.net.res_line.index:
                continue
            try:
                L = float(env.net.res_line.at[li, "loading_percent"])
            except Exception:
                L = float("nan")
            hold = int(holds.get(int(li), 0))
            scale = _quick_prob_from_map(prob_map, L if np.isfinite(L) else 0.0)
            
            # Flag if eligible for trip
            eligible = (hold >= min_hold) and (L >= overload_min)
            marker = " ✓" if eligible else ""
            
            stats.append((li, L, hold, scale, eligible))
        
        # Sort by loading
        stats.sort(key=lambda t: t[1], reverse=True)
        
        if stats:
            print(f"         • Top candidates by loading (✓ = eligible for trip):")
            for li, L, hold, scale, eligible in stats[:15]:
                marker = " ✓" if eligible else ""
                print(f"           L{li:>4}: {L:6.1f}% hold={hold:2d}/{min_hold} scale={scale:.3f}{marker}")

    # Full debug dump if very verbose
    if hasattr(env, "_debug_dump_cascade_state"):
        last_trip = _fmt_or(env, "_last_trip_step", -10**9)
        if applied or (env.current_step - last_trip) <= 5:
            print(f"         • Full cascade state dump:")
            env._debug_dump_cascade_state(prefix="           ")

def run_episode(env, steps, verbose, diagnose=False):
    # --- RESET
    obs, info = env.reset()
    _pf_snapshot(env, "RESET", info=info)

    # Priority summary
    try:
        if hasattr(env, "net"):
            print_priority_summary(env.net)
    except Exception:
        pass

    # Seed schedule
    seed_sched = sorted(getattr(env, "_events_by_step", {}).items())
    compact = [(stp, [ev.get("target") for ev in evs if ev.get("op") == "trip"]) 
               for stp, evs in seed_sched]
    print(f"EPISODE • seed schedule: {compact}\n")

    first_seed_step = compact[0][0] if compact else None

    casc_trips = 0
    last_info = info
    
    for _ in range(steps):
        # Pre-trip snapshot
        if first_seed_step is not None and env.current_step == (first_seed_step - 1):
            _pf_snapshot(env, f"PRE-TRIP@t={env.current_step}", info=last_info)
            if verbose and hasattr(env, "_debug_dump_cascade_state"):
                print("         • Pre-trip cascade state:")
                env._debug_dump_cascade_state(prefix="           ")

        # No-op action (middle)
        action = env.action_space.n // 2
        obs, rew, done, trunc, info = env.step(action)
        last_info = info

        # Post-trip snapshot
        applied = info.get("events_applied", [])
        if any(("line:" in s and "trip" in s) for s in applied):
            _pf_snapshot(env, f"POST-TRIP@t={env.current_step}", info=info)
            
            # Diagnose stress effectiveness
            if diagnose and hasattr(env, "_diagnose_stress_effectiveness"):
                env._diagnose_stress_effectiveness()
            
            if verbose and hasattr(env, "_debug_dump_cascade_state"):
                print("         • Post-trip cascade state:")
                env._debug_dump_cascade_state(prefix="           ")

        # Count cascade trips
        casc_trips += sum(1 for s in applied if "trip(cascade)" in s)

        # Show candidates
        cands = _eligible_neighbors(env)
        _step_debug(env, env.current_step, cands, applied, verbose=verbose)

        if done:
            print(f"\nEpisode terminated at step {env.current_step}")
            break

    # Final summary
    print("\n" + "=" * 80)
    print("EPISODE SUMMARY:")
    print(f"  Steps completed: {env.current_step}/{steps}")
    print(f"  Cascade trips: {casc_trips}")
    print(f"  Final served_total: {last_info.get('served_total_frac', 0.0):.3f}")
    print(f"  Final served_crit:  {last_info.get('served_crit_frac', 0.0):.3f}")
    print("=" * 80 + "\n")

    return dict(
        seed_schedule=compact,
        cascade_trips=casc_trips,
        served_total_frac=float(last_info.get("served_total_frac", 0.0)),
        steps_completed=env.current_step,
    )

def print_priority_summary(net):
    if not (hasattr(net, "load") and len(net.load)):
        print("\nPRIORITY • no loads present.\n")
        return

    col_tier = None
    if "tier" in net.load.columns:
        col_tier = "tier"
    elif "priority_tier" in net.load.columns:
        col_tier = "priority_tier"

    print("\n" + "=" * 80)
    print("PRIORITY SUMMARY:")
    print("=" * 80)
    
    if col_tier:
        tier_counts = net.load[col_tier].value_counts().reindex(
            ["critical", "important", "normal"], fill_value=0
        )
        print(f"Tier counts: {dict(tier_counts)}")
    elif "priority" in net.load.columns:
        pr = net.load["priority"].astype(int)
        counts = {TIER_NAMES.get(k, str(k)): int((pr == k).sum()) for k in (0, 1, 2)}
        print(f"Tier counts: {counts}")

    if "p_mw" in net.load.columns:
        if col_tier:
            mw_by_tier = net.load.groupby(col_tier)["p_mw"].agg(["count", "sum"]).reindex(
                ["critical", "important", "normal"], fill_value=0.0
            )
            print("\nLoads by tier (count, total MW):")
            for tier, row in mw_by_tier.iterrows():
                print(f"  {tier:<12} count={int(row['count']):4d}  p_total={row['sum']:8.1f} MW")
        elif "priority" in net.load.columns:
            pr = net.load["priority"].astype(int)
            print("\nLoads by tier (count, total MW):")
            for k in (0, 1, 2):
                mask = (pr == k)
                cnt = int(mask.sum())
                total = float(net.load.loc[mask, "p_mw"].sum())
                print(f"  {TIER_NAMES[k]:<12} count={cnt:4d}  p_total={total:8.1f} MW")

    if "p_mw" in net.load.columns and "bus" in net.load.columns:
        bus_p = net.load.groupby("bus")["p_mw"].sum().sort_values(ascending=False).head(10)
        print("\nTop 10 buses by total load:")
        for b, p in bus_p.items():
            sub = net.load[net.load["bus"] == b]
            if col_tier:
                mix = sub[col_tier].value_counts().to_dict()
            elif "priority" in sub.columns:
                pr_vals = sub["priority"].astype(int)
                mix = {TIER_NAMES.get(int(k), str(k)): int((pr_vals == k).sum()) 
                       for k in (0, 1, 2) if (pr_vals == k).any()}
            else:
                mix = {}
            name = net.bus.at[b, "name"] if "name" in net.bus.columns else f"{b}"
            print(f"  Bus {b:>4} ({name}): {p:8.1f} MW  mix={mix}")
    
    print("=" * 80 + "\n")

def _step_debug(env, step_idx, candidates, applied, verbose=False):
    print(f"Step {step_idx:02d} • candidates: {len(candidates)}")
    if applied:
        print(f"         • applied: {applied}")
    
    # NEW: Check if stress is active
    stress_active = getattr(env, "_local_stress_active", False)
    stress_lines = getattr(env, "_local_stress_lines", set())
    if stress_active:
        print(f"         • STRESS ACTIVE: {len(stress_lines)} lines affected")
    
    # Show loading in stressed region
    if stress_active and hasattr(env.net, "res_line"):
        stressed_loading = env.net.res_line.loading_percent.loc[list(stress_lines)]
        print(f"         • Stressed region: max={stressed_loading.max():.1f}%, "
              f"mean={stressed_loading.mean():.1f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True, help="Path to env.yaml")
    ap.add_argument("--env_id", type=str, default="case145", help="Feeder env_id")
    ap.add_argument("--scenario", type=str, default="", help="Scenario JSON/YAML")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true", help="Show detailed cascade state")
    ap.add_argument("--diagnose", action="store_true", help="Run stress diagnostics")
    args = ap.parse_args()

    np.random.seed(args.seed)
    cfg = load_yaml(args.env)
    cfg["max_steps"] = int(args.steps)

    if args.scenario:
        try:
            scenario = load_yaml(args.scenario)
        except Exception:
            scenario = {"name": f"scenario:{args.scenario}", "events": [], "repairs": []}
    else:
        scenario = {"name": "baseline", "events": [], "repairs": []}

    env_obj = make_env(args.env_id, cfg, scenario)
    env = env_obj[0] if isinstance(env_obj, (list, tuple)) else env_obj

    _print_hdr(env)

    casc_ep = 0
    examples = []
    
    for ep in range(args.episodes):
        print(f"\n{'#' * 80}")
        print(f"### EPISODE {ep + 1}/{args.episodes}")
        print(f"{'#' * 80}\n")
        
        res = run_episode(env, args.steps, args.verbose, diagnose=args.diagnose)
        casc_ep += int(res["cascade_trips"] > 0)
        if len(examples) < 3:
            examples.append((res["seed_schedule"], res["cascade_trips"]))

    print("\n" + "=" * 80)
    print("=== FINAL SUMMARY ===")
    print("=" * 80)
    print(f"Episodes run: {args.episodes}")
    print(f"Episodes with ≥1 cascade trip: {casc_ep}/{args.episodes}")
    print("\nSample results:")
    for i, (sched, n) in enumerate(examples, 1):
        print(f"  Episode {i}: cascade_trips={n:2d}, schedule={sched}")
    print("=" * 80)

if __name__ == "__main__":
    main()