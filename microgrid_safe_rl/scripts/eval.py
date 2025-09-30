#!/usr/bin/env python3
import argparse, os, csv, numpy as np
from stable_baselines3 import PPO, SAC
from microgrid_safe_rl.utils.config import load_yaml
from microgrid_safe_rl.envs.factory import make_env

ALGOS = {"ppo": PPO, "sac": SAC}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True, help="feeder id (e.g., case145)")
    p.add_argument("--model_path", required=True, help="path to SB3 model (.zip or directory)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    p.add_argument("--env_cfg", default="env.yaml")
    p.add_argument("--scenario_cfg", default="scenario/eval_line_only.yaml")
    p.add_argument("--save_csv", default=None, help="write per-episode metrics CSV")
    p.add_argument("--save_per_step", default=None, help="write per-step CSV (for trajectory plots)")
    p.add_argument("--nodes", type=int, default=None, help="Total buses (for shed calc)")
    p.add_argument("--seed", type=int, default=0, help="base seed (incremented per-episode)")
    return p.parse_args()

def load_model(model_path, algo, env):
    if os.path.isdir(model_path):
        zips = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".zip")]
        assert len(zips) > 0, f"No .zip found in {model_path}"
        zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        model_path = zips[0]
    return ALGOS[algo].load(model_path, env=env, print_system_info=False)

def first_isolation_step(infos):
    for t, inf in enumerate(infos, start=1):
        if inf.get("isolation_happened", False):
            return t
    return None

def main():
    args = parse_args()
    env_cfg = load_yaml(args.env_cfg)
    scenario = load_yaml(args.scenario_cfg)

    # Deterministic-ish eval per episode if your env uses np RNG
    env = make_env(args.env_id, env_cfg, scenario)

    model = load_model(args.model_path, args.algo, env)

    # CSV writers (lazy-open)
    per_ep_writer = None
    per_step_writer = None
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        f = open(args.save_csv, "w", newline="")
        per_ep_writer = csv.writer(f)
        per_ep_writer.writerow([
            "episode","reward","steps",
            "isolated","iso_step",
            "served_total","served_crit","served_imp",
            "pf_success_any","pf_fail_steps",
            "switch_toggles","cascade_trips_total",
            "shed_frac","shed_nodes",
            "der_bound_frac","der_bound_mw_last","served_mw_last"
        ])

    if args.save_per_step:
        os.makedirs(os.path.dirname(args.save_per_step), exist_ok=True)
        f2 = open(args.save_per_step, "w", newline="")
        per_step_writer = csv.writer(f2)
        per_step_writer.writerow([
            "episode","step","reward",
            "served_total","served_crit","served_imp",
            "fault_live","powerflow_success",
            "toggled_switch","cascade_tripped",
            "der_bound_frac","served_mw","der_bound_mw",
            "isolation_happened"
        ])

    ep_rewards, ep_steps, ep_infos = [], [], []

    for ep in range(args.episodes):
        # optional per-episode seed bump (if your env respects np RNG from reset)
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_r = 0.0
        steps = 0
        infos = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_r += float(r)
            steps += 1
            infos.append(info)

            # per-step logging (optional)
            if per_step_writer:
                per_step_writer.writerow([
                    ep+1, steps, float(r),
                    info.get("served_total_frac", np.nan),
                    info.get("served_crit_frac",  np.nan),
                    info.get("served_imp_frac",   np.nan),
                    int(info.get("fault_live", False)),
                    int(info.get("powerflow_success", False)),
                    info.get("toggled_switch", ""),
                    # cascade_tripped may be list or missing
                    (",".join(map(str, info.get("cascade_tripped", [])))
                     if isinstance(info.get("cascade_tripped", None), list)
                     else (info.get("cascade_tripped") if info.get("cascade_tripped") is not None else "")),
                    info.get("der_bound_frac", np.nan),
                    info.get("served_mw", np.nan),
                    info.get("der_bound_mw", np.nan),
                    int(info.get("isolation_happened", False)),
                ])

        ep_rewards.append(total_r)
        ep_steps.append(steps)
        ep_infos.append(infos)

        comps = infos[-1].get("reward_components", {})
        print(f"Episode {ep+1}: reward={total_r:.2f}  steps={steps}  comps={comps}")

        # ---- Per-episode metrics ----
        iso_step = first_isolation_step(infos)
        isolated = iso_step is not None

        # PF success if at least one step succeeded
        pf_any = any(inf.get("powerflow_success", False) for inf in infos)
        pf_fail_steps = sum(1 for inf in infos if not inf.get("powerflow_success", False))

        # Switch toggles count
        switch_toggles = sum(1 for inf in infos if "toggled_switch" in inf)

        # Cascade trips total (count of line ids)
        cascade_trips_total = 0
        for inf in infos:
            c = inf.get("cascade_tripped", None)
            if isinstance(c, list):
                cascade_trips_total += len(c)
            elif c not in (None, "", "[]"):
                # if your logger ever stores a scalar count/string
                try:
                    cascade_trips_total += int(c)
                except Exception:
                    cascade_trips_total += 1

        # Served fractions @ end
        crit_frac = float(infos[-1].get("served_crit_frac", 0.0))
        imp_frac  = float(infos[-1].get("served_imp_frac", 0.0))
        tot_frac  = float(infos[-1].get("served_total_frac", 0.0))

        # DER bound context
        der_bound_frac = float(infos[-1].get("der_bound_frac", np.nan))
        der_bound_mw   = float(infos[-1].get("der_bound_mw", np.nan))
        served_mw_last = float(infos[-1].get("served_mw", np.nan))

        # Shed metrics
        n_nodes = args.nodes or len(env.net.bus)
        shed_frac = max(0.0, 1.0 - tot_frac)
        shed_nodes = int(round(shed_frac * n_nodes))

        if per_ep_writer:
            per_ep_writer.writerow([
                ep+1, total_r, steps,
                int(isolated), (iso_step if iso_step is not None else -1),
                tot_frac, crit_frac, imp_frac,
                int(pf_any), pf_fail_steps,
                switch_toggles, cascade_trips_total,
                shed_frac, shed_nodes,
                der_bound_frac, der_bound_mw, served_mw_last
            ])

    # ---- Console summary ----
    iso_steps = [first_isolation_step(infs) for infs in ep_infos]
    success = [s is not None for s in iso_steps]
    succ_rate = float(np.mean(success)) if len(success) else 0.0
    tti = np.mean([s for s in iso_steps if s is not None]) if any(success) else float("nan")

    crit_mean = np.mean([infs[-1].get("served_crit_frac", 0.0) for infs in ep_infos]) if ep_infos else 0.0
    imp_mean  = np.mean([infs[-1].get("served_imp_frac", 0.0) for infs in ep_infos]) if ep_infos else 0.0
    tot_mean  = np.mean([infs[-1].get("served_total_frac", 0.0) for infs in ep_infos]) if ep_infos else 0.0

    print(f"\nIsolation success: {succ_rate*100:.1f}%")
    if any(success):
        print(f"Time-to-isolate (successful only): {tti:.2f} steps")
    print(f"Served @end: crit={crit_mean:.3f}  imp={imp_mean:.3f}  total={tot_mean:.3f}")
    print(f"Avg reward over {args.episodes} eps: {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")

    # Close CSVs
    if args.save_csv:
        f.close()
        print(f"Saved metrics: {args.save_csv}")
    if args.save_per_step:
        f2.close()
        print(f"Saved per-step log: {args.save_per_step}")

if __name__ == "__main__":
    main()
