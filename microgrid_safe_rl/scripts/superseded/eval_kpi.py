import argparse, csv, os, numpy as np
from stable_baselines3 import PPO
from microgrid_safe_rl.envs.factory import make_env
from microgrid_safe_rl.utils.config import load_yaml

def run_episode(model, env):
    obs, info = env.reset()
    done = False
    total = 0.0
    comps_sum = {"crit":0.0,"imp":0.0,"volt":0.0,"shed":0.0,"switch":0.0,"fault":0.0}
    toggles = 0
    steps = 0
    last_info = info
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total += float(r); steps += 1
        done = term or trunc
        rc = info.get("reward_components", {})
        for k in comps_sum: comps_sum[k] += float(rc.get(k, 0.0))
        if "toggled_switch" in info: toggles += 1
        last_info = info

    # KPIs at end of episode
    v = env.net.res_bus.vm_pu.values if hasattr(env.net, "res_bus") and len(env.net.res_bus) else np.array([])
    vmin, vmax = env.v_limits
    volt_viol = int(np.sum((v < vmin) | (v > vmax))) if v.size else 0

    base = env.net.load.get("p_base_mw", env.net.load.p_mw) if len(env.net.load) else []
    cur  = env.net.load.p_mw if len(env.net.load) else []
    prio = env.net.load.get("priority", 0) if len(env.net.load) else []
    crit_served = float(cur[prio==2].sum())/float(base[prio==2].sum()) if len(env.net.load) and float(base[prio==2].sum())>1e-9 else 1.0
    imp_served  = float(cur[prio==1].sum())/float(base[prio==1].sum()) if len(env.net.load) and float(base[prio==1].sum())>1e-9 else 1.0
    shed_total  = float((base-cur).clip(lower=0).sum()) if len(env.net.load) else 0.0

    fault_success = comps_sum["fault"] > 0.0  # one-off bonus means we isolated at least once
    return {
        "reward": total, "steps": steps,
        "crit_served": crit_served, "imp_served": imp_served,
        "shed_total_mw": shed_total, "volt_viol_buses": volt_viol,
        "switch_toggles": toggles, "fault_isolated": int(fault_success),
        **{f"R_{k}": comps_sum[k] for k in comps_sum}
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--env_cfg", default="env.yaml")
    p.add_argument("--scenario_cfg", default="scenario/baseline.yaml")
    p.add_argument("--out", default="artifacts/runs/eval_kpi.csv")
    args = p.parse_args()

    env = make_env(args.env_id, load_yaml(args.env_cfg), load_yaml(args.scenario_cfg))
    model = PPO.load(args.model_path, env=None)

    rows = [run_episode(model, env) for _ in range(args.episodes)]
    keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    # print summary
    avg = {k: np.mean([r[k] for r in rows]) for k in rows[0].keys() if isinstance(rows[0][k], (int,float))}
    print("Averages over", args.episodes, "episodes:")
    for k,v in avg.items():
        if k.startswith("R_"): continue
        print(f"  {k}: {v:.3f}")
    print("Avg reward components:", {k: round(np.mean([r[k] for r in rows]),2) for k in [f'R_{c}' for c in ['crit','imp','volt','shed','switch','fault']]})
    print("Saved CSV ->", args.out)

if __name__ == "__main__":
    main()
