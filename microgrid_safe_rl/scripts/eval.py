# microgrid_safe_rl/scripts/eval.py
import argparse, os, json, numpy as np
from stable_baselines3 import PPO, SAC
from microgrid_safe_rl.utils.config import load_yaml
from microgrid_safe_rl.envs.factory import make_env

ALGOS = {"ppo": PPO, "sac": SAC}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True, help="feeder id (e.g., case145)")
    p.add_argument("--model_path", required=True, help="path prefix to SB3 model (directory or zip)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    p.add_argument("--env_cfg", default="env.yaml")
    p.add_argument("--scenario_cfg", default="scenario/eval_line_only.yaml")
    p.add_argument("--save_csv", default=None)
    return p.parse_args()

def load_model(model_path, algo, env):
    # allow either model.zip or directory with latest.zip-like
    if os.path.isdir(model_path):
        # pick most recent .zip inside
        zips = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".zip")]
        assert len(zips) > 0, f"No .zip found in {model_path}"
        zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        model_path = zips[0]
    return ALGOS[algo].load(model_path, env=env, print_system_info=False)

def main():
    args = parse_args()
    env_cfg = load_yaml(args.env_cfg)
    scenario = load_yaml(args.scenario_cfg)
    env = make_env(args.env_id, env_cfg, scenario)
    model = load_model(args.model_path, args.algo, env)

    ep_rewards, ep_steps, ep_infos = [], [], []

    for ep in range(args.episodes):
        obs, info = env.reset()
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
        ep_rewards.append(total_r)
        ep_steps.append(steps)
        ep_infos.append(infos)
        comps = infos[-1].get("reward_components", {})
        print(f"Episode {ep+1}: reward={total_r:.2f} steps={steps}  comps={comps}")

    # ---- Metrics ----
    def first_isolation_step(infos):
        for t, inf in enumerate(infos, start=1):
            if inf.get("isolation_happened", False):
                return t
        return None

    iso_steps = [first_isolation_step(infs) for infs in ep_infos]
    success = [s is not None for s in iso_steps]
    succ_rate = float(np.mean(success)) if len(success) else 0.0
    tti = np.mean([s for s in iso_steps if s is not None]) if any(success) else float("nan")

    crit_frac = [infs[-1].get("served_crit_frac", 0.0) for infs in ep_infos]
    imp_frac  = [infs[-1].get("served_imp_frac", 0.0) for infs in ep_infos]
    tot_frac  = [infs[-1].get("served_total_frac", 0.0) for infs in ep_infos]

    print(f"\nIsolation success: {succ_rate*100:.1f}%")
    if any(success):
        print(f"Time-to-isolate (successful eps only): {tti:.2f} steps")
    print(f"Served fractions @end: crit={np.mean(crit_frac):.3f}, imp={np.mean(imp_frac):.3f}, total={np.mean(tot_frac):.3f}")
    print(f"Avg reward over {args.episodes} eps: {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")

    # Optional CSV
    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode","reward","steps","isolated","iso_step","served_total","served_crit","served_imp"])
            for i,(R,S,ok,st,tf,cf,if_) in enumerate(zip(ep_rewards, ep_steps, success, iso_steps, tot_frac, crit_frac, imp_frac), start=1):
                w.writerow([i, R, S, int(ok), (st if st is not None else -1), tf, cf, if_])
        print(f"Saved metrics: {args.save_csv}")

if __name__ == "__main__":
    main()
