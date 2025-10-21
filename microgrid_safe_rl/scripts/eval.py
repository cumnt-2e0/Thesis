#!/usr/bin/env python3
"""
Enhanced evaluation script with detailed trajectory logging.
Answers: What switches does the agent operate? When? Why?
"""
import argparse
import os
import csv
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from microgrid_safe_rl.utils.config import load_yaml
from microgrid_safe_rl.envs.factory import make_env

ALGOS = {"ppo": PPO, "sac": SAC}

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate trained RL agent with detailed trajectory logging"
    )
    p.add_argument("--env_id", required=True, help="feeder id (e.g., case33, case145)")
    p.add_argument("--model_path", required=True, help="path to SB3 model (.zip or directory)")
    p.add_argument("--episodes", type=int, default=5, help="number of episodes to run")
    p.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    p.add_argument("--env_cfg", default="microgrid_safe_rl/configs/env_case33.yaml")
    p.add_argument("--scenario_cfg", default="microgrid_safe_rl/configs/scenario/eval_case33.yaml")
    p.add_argument("--save_csv", default=None, help="write per-episode summary metrics CSV")
    p.add_argument("--save_per_step", default=None, help="write per-step basic CSV")
    p.add_argument("--save_trajectory", default=None, help="write detailed per-step trajectory CSV")
    p.add_argument("--nodes", type=int, default=None, help="Total buses (for shed calc)")
    p.add_argument("--seed", type=int, default=0, help="base seed (incremented per-episode)")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    return p.parse_args()

def load_model(model_path, algo, env):
    """Load trained model from .zip file or directory."""
    if os.path.isdir(model_path):
        zips = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".zip")]
        assert len(zips) > 0, f"No .zip found in {model_path}"
        zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        model_path = zips[0]
        print(f"Loading most recent model: {model_path}")
    return ALGOS[algo].load(model_path, env=env, print_system_info=False)

def first_isolation_step(infos):
    """Find first step where isolation occurred."""
    for t, inf in enumerate(infos, start=1):
        if inf.get("isolation_happened", False):
            return t
    return None

def parse_action_detail(env, action, info):
    """Decode what action was taken."""
    action = int(action)
    
    # No-op
    if action == env._switch_count:
        return "no_op", ""
    
    # Switch open
    if action < env._switch_count:
        sw_id = env.all_switch_ids[action]
        return "switch_open", f"line_{sw_id}"
    
    # Switch close
    if action < env._switch_count * 2:
        sw_id = env.all_switch_ids[action - env._switch_count]
        return "switch_close", f"line_{sw_id}"
    
    # Load shedding
    if action > env._switch_count * 2:
        offset = action - (env._switch_count * 2 + 1)
        load_idx = offset // 2
        direction = "increase" if (offset % 2) else "decrease"
        return f"load_{direction}", f"load_{load_idx}"
    
    return "unknown", ""

def create_trajectory_writer(filepath):
    """Create detailed trajectory CSV writer."""
    if filepath is None:
        return None, None
    
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    f = open(filepath, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "episode", "step", "time_s", "reward",
        # Topology state
        "live_lines", "energized_buses", "isolated_buses",
        # Loads
        "served_critical_frac", "served_important_frac", "served_total_frac",
        "served_mw", "total_load_mw", "shed_mw",
        # DER state
        "der_available_mw", "der_online_mw", "der_bound_mw", "der_deficit_frac",
        # Electrical
        "v_min_pu", "v_max_pu", "v_violations", "line_loading_max_pct",
        # Actions taken
        "action_type", "action_detail",
        "switch_toggled", "load_shed_id", "load_shed_amount_mw",
        # Cascade/fault events
        "cascade_tripped_lines", "relay_trips",
        # Safety
        "action_masked", "mask_reason",
        "powerflow_success", "powerflow_iterations",
    ])
    return f, writer

def log_detailed_step(writer, episode, step, env, action, reward, info):
    """Log comprehensive step information to trajectory CSV."""
    if writer is None:
        return
    
    action_type, action_detail = parse_action_detail(env, action, info)
    
    # Topology - count active lines and energized buses
    live_lines = len([l for l in env.net.line.itertuples() if l.in_service])
    
    # Count energized buses (those with valid voltage results)
    energized_buses = 0
    if hasattr(env.net, 'res_bus') and len(env.net.res_bus) > 0:
        for idx in range(len(env.net.bus)):
            if idx < len(env.net.res_bus):
                vm = env.net.res_bus.vm_pu.iloc[idx]
                if not np.isnan(vm) and vm > 0:
                    energized_buses += 1
    
    isolated_buses = len(env.net.bus) - energized_buses
    
    # Loads
    served_crit = info.get("served_crit_frac", np.nan)
    served_imp = info.get("served_imp_frac", np.nan)
    served_tot = info.get("served_total_frac", np.nan)
    served_mw = info.get("served_mw", np.nan)
    total_load = env.net.load.p_mw.sum()
    shed_mw = max(0, total_load - served_mw) if not np.isnan(served_mw) else np.nan
    
    # DER state
    der_available = 0.0
    der_online = 0.0
    if hasattr(env.net, 'sgen') and len(env.net.sgen) > 0:
        for sg in env.net.sgen.itertuples():
            if not getattr(sg, 'out_of_service', False):
                der_available += sg.p_mw
            if getattr(sg, 'in_service', True) and not getattr(sg, 'out_of_service', False):
                der_online += sg.p_mw
    
    der_bound = info.get("der_bound_mw", np.nan)
    der_deficit = info.get("der_bound_frac", np.nan)
    
    # Voltage
    v_min = v_max = np.nan
    v_violations = 0
    if hasattr(env.net, "res_bus") and len(env.net.res_bus) > 0:
        v_vals = env.net.res_bus.vm_pu.values
        valid_v = v_vals[~np.isnan(v_vals)]
        if len(valid_v) > 0:
            v_min = float(np.min(valid_v))
            v_max = float(np.max(valid_v))
            v_violations = int(np.sum((valid_v < 0.95) | (valid_v > 1.05)))
    
    # Line loading
    loading_max = np.nan
    if hasattr(env.net, "res_line") and len(env.net.res_line) > 0:
        load_vals = env.net.res_line.loading_percent.values
        valid_loads = load_vals[~np.isnan(load_vals)]
        if len(valid_loads) > 0:
            loading_max = float(np.max(valid_loads))
    
    # Cascade events
    cascade_trips = info.get("cascade_tripped", [])
    if isinstance(cascade_trips, list):
        cascade_str = ",".join(map(str, cascade_trips))
    else:
        cascade_str = str(cascade_trips) if cascade_trips else ""
    
    # Safety mask info
    action_masked = info.get("action_was_masked", False)
    mask_reason = info.get("mask_reason", "")
    
    # Action details from info
    switch_toggled = info.get("toggled_switch", "")
    load_shed_id = info.get("load_shed_id", "")
    load_shed_amt = info.get("load_shed_amount_mw", np.nan)
    
    # Powerflow convergence
    pf_success = info.get("powerflow_success", True)
    pf_iters = info.get("powerflow_iterations", 0)
    
    # Get step seconds from env
    step_seconds = getattr(env, 'step_seconds', 5.0)
    
    writer.writerow([
        episode, step, step * step_seconds, reward,
        live_lines, energized_buses, isolated_buses,
        served_crit, served_imp, served_tot,
        served_mw, total_load, shed_mw,
        der_available, der_online, der_bound, der_deficit,
        v_min, v_max, v_violations, loading_max,
        action_type, action_detail,
        switch_toggled, load_shed_id, load_shed_amt,
        cascade_str, "",  # relay trips placeholder
        int(action_masked), mask_reason,
        int(pf_success), pf_iters,
    ])

def main():
    args = parse_args()
    
    print("="*80)
    print("ENHANCED EVALUATION SCRIPT")
    print("="*80)
    print(f"Environment:  {args.env_id}")
    print(f"Model:        {args.model_path}")
    print(f"Episodes:     {args.episodes}")
    print(f"Scenario:     {args.scenario_cfg}")
    print(f"Deterministic: {args.deterministic}")
    print("="*80)
    
    # Load configs
    env_cfg = load_yaml(args.env_cfg)
    scenario = load_yaml(args.scenario_cfg) if os.path.exists(args.scenario_cfg) else {}
    
    # Create environment
    env = make_env(args.env_id, env_cfg, scenario)
    
    # Load trained model
    model = load_model(args.model_path, args.algo, env)
    
    # Create CSV writers
    per_ep_file = None
    per_ep_writer = None
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        per_ep_file = open(args.save_csv, "w", newline="")
        per_ep_writer = csv.writer(per_ep_file)
        per_ep_writer.writerow([
            "episode", "reward", "steps",
            "isolated", "iso_step",
            "served_total", "served_crit", "served_imp",
            "pf_success_any", "pf_fail_steps",
            "switch_toggles", "cascade_trips_total",
            "shed_frac", "shed_nodes",
            "der_bound_frac", "der_bound_mw_last", "served_mw_last"
        ])
    
    per_step_file = None
    per_step_writer = None
    if args.save_per_step:
        os.makedirs(os.path.dirname(args.save_per_step) or ".", exist_ok=True)
        per_step_file = open(args.save_per_step, "w", newline="")
        per_step_writer = csv.writer(per_step_file)
        per_step_writer.writerow([
            "episode", "step", "reward",
            "served_total", "served_crit", "served_imp",
            "fault_live", "powerflow_success",
            "toggled_switch", "cascade_tripped",
            "der_bound_frac", "served_mw", "der_bound_mw",
            "isolation_happened"
        ])
    
    trajectory_file, trajectory_writer = create_trajectory_writer(args.save_trajectory)
    
    # Run episodes
    ep_rewards = []
    ep_steps = []
    ep_infos = []
    
    for ep in range(args.episodes):
        print(f"\n--- Episode {ep+1}/{args.episodes} ---")
        
        # Reset with seed
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_r = 0.0
        steps = 0
        infos = []
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=args.deterministic)
            
            # Step environment
            obs, r, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            
            total_r += float(r)
            steps += 1
            infos.append(info)
            
            # Log detailed trajectory
            if trajectory_writer:
                log_detailed_step(trajectory_writer, ep+1, steps, env, action, r, info)
            
            # Log basic per-step
            if per_step_writer:
                cascade_trips = info.get("cascade_tripped", [])
                if isinstance(cascade_trips, list):
                    cascade_str = ",".join(map(str, cascade_trips))
                else:
                    cascade_str = str(cascade_trips) if cascade_trips else ""
                
                per_step_writer.writerow([
                    ep+1, steps, float(r),
                    info.get("served_total_frac", np.nan),
                    info.get("served_crit_frac", np.nan),
                    info.get("served_imp_frac", np.nan),
                    int(info.get("fault_live", False)),
                    int(info.get("powerflow_success", True)),
                    info.get("toggled_switch", ""),
                    cascade_str,
                    info.get("der_bound_frac", np.nan),
                    info.get("served_mw", np.nan),
                    info.get("der_bound_mw", np.nan),
                    int(info.get("isolation_happened", False)),
                ])
        
        ep_rewards.append(total_r)
        ep_steps.append(steps)
        ep_infos.append(infos)
        
        # Print episode summary
        comps = infos[-1].get("reward_components", {})
        print(f"  Reward: {total_r:.2f}")
        print(f"  Steps:  {steps}")
        print(f"  Final served - Critical: {infos[-1].get('served_crit_frac', 0)*100:.1f}%, "
              f"Important: {infos[-1].get('served_imp_frac', 0)*100:.1f}%, "
              f"Total: {infos[-1].get('served_total_frac', 0)*100:.1f}%")
        if comps:
            print(f"  Reward components: {comps}")
        
        # ---- Per-episode metrics ----
        if per_ep_writer:
            iso_step = first_isolation_step(infos)
            isolated = iso_step is not None
            
            # PF success
            pf_any = any(inf.get("powerflow_success", True) for inf in infos)
            pf_fail_steps = sum(1 for inf in infos if not inf.get("powerflow_success", True))
            
            # Switch toggles
            switch_toggles = sum(1 for inf in infos if inf.get("toggled_switch", ""))
            
            # Cascade trips total
            cascade_trips_total = 0
            for inf in infos:
                c = inf.get("cascade_tripped", None)
                if isinstance(c, list):
                    cascade_trips_total += len(c)
                elif c not in (None, "", "[]"):
                    try:
                        cascade_trips_total += int(c)
                    except Exception:
                        cascade_trips_total += 1
            
            # Final served fractions
            crit_frac = float(infos[-1].get("served_crit_frac", 0.0))
            imp_frac = float(infos[-1].get("served_imp_frac", 0.0))
            tot_frac = float(infos[-1].get("served_total_frac", 0.0))
            
            # DER context
            der_bound_frac = float(infos[-1].get("der_bound_frac", np.nan))
            der_bound_mw = float(infos[-1].get("der_bound_mw", np.nan))
            served_mw_last = float(infos[-1].get("served_mw", np.nan))
            
            # Shed metrics
            n_nodes = args.nodes or len(env.net.bus)
            shed_frac = max(0.0, 1.0 - tot_frac)
            shed_nodes = int(round(shed_frac * n_nodes))
            
            per_ep_writer.writerow([
                ep+1, total_r, steps,
                int(isolated), (iso_step if iso_step is not None else -1),
                tot_frac, crit_frac, imp_frac,
                int(pf_any), pf_fail_steps,
                switch_toggles, cascade_trips_total,
                shed_frac, shed_nodes,
                der_bound_frac, der_bound_mw, served_mw_last
            ])
    
    # ---- Final summary ----
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    iso_steps = [first_isolation_step(infs) for infs in ep_infos]
    success = [s is not None for s in iso_steps]
    succ_rate = float(np.mean(success)) if len(success) else 0.0
    tti = np.mean([s for s in iso_steps if s is not None]) if any(success) else float("nan")
    
    crit_mean = np.mean([infs[-1].get("served_crit_frac", 0.0) for infs in ep_infos])
    imp_mean = np.mean([infs[-1].get("served_imp_frac", 0.0) for infs in ep_infos])
    tot_mean = np.mean([infs[-1].get("served_total_frac", 0.0) for infs in ep_infos])
    
    print(f"\nIsolation success rate: {succ_rate*100:.1f}%")
    if any(success):
        print(f"Time-to-isolate (successful): {tti:.2f} steps")
    
    print(f"\nFinal Load Served (average):")
    print(f"  Critical:  {crit_mean*100:.1f}%")
    print(f"  Important: {imp_mean*100:.1f}%")
    print(f"  Total:     {tot_mean*100:.1f}%")
    
    print(f"\nReward: {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"Steps:  {np.mean(ep_steps):.1f} ± {np.std(ep_steps):.1f}")
    
    # Close files
    if per_ep_file:
        per_ep_file.close()
        print(f"\n✓ Saved episode summary: {args.save_csv}")
    
    if per_step_file:
        per_step_file.close()
        print(f"✓ Saved per-step log: {args.save_per_step}")
    
    if trajectory_file:
        trajectory_file.close()
        print(f"✓ Saved detailed trajectory: {args.save_trajectory}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()