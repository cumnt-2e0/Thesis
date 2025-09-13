import argparse, csv, os, numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from microgrid_safe_rl.utils.config import load_yaml
from microgrid_safe_rl.utils.seed import set_global_seed
from microgrid_safe_rl.envs.factory import make_env

FEEDERS = ["case33", "case145", "case300", "case1888"]

def eval_model(model, env, episodes=3):
    metrics = []
    for _ in range(episodes):
        obs, info = env.reset()
        done=False; total=0.0; steps=0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total += float(r); steps += 1; done = term or trunc
        v = env.net.res_bus.vm_pu.values if hasattr(env.net,'res_bus') and len(env.net.res_bus) else []
        volt_viol = float(np.sum((v<env.v_limits[0]) | (v>env.v_limits[1]))) if len(v) else 0.0
        base = env.net.load.get('p_base_mw', env.net.load.p_mw)
        cur  = env.net.load.p_mw
        prio = env.net.load.get('priority', 0)
        crit_served = float(cur[prio==2].sum())/float(base[prio==2].sum()) if float(base[prio==2].sum())>1e-9 else 1.0
        imp_served  = float(cur[prio==1].sum())/float(base[prio==1].sum()) if float(base[prio==1].sum())>1e-9 else 1.0
        shed_total  = float((base-cur).clip(lower=0).sum())
        metrics.append({'reward': total, 'steps': steps, 'volt_viol': volt_viol, 'crit_served': crit_served, 'imp_served': imp_served, 'shed_total': shed_total})
    return {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0].keys()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=100000)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--out', type=str, default='artifacts/runs/summary.csv')
    p.add_argument('--agent_cfg', type=str, default='agent/ppo.yaml')
    p.add_argument('--env_cfg', type=str, default='env.yaml')
    p.add_argument('--scenario_cfg', type=str, default='scenario/baseline.yaml')
    args = p.parse_args()

    agent_cfg = load_yaml(args.agent_cfg); env_cfg = load_yaml(args.env_cfg); scenario = load_yaml(args.scenario_cfg)

    rows = []
    for feeder in FEEDERS:
        set_global_seed(0)
        env = make_env(feeder, env_cfg, scenario)
        venv = DummyVecEnv([lambda: Monitor(env)])
        model = PPO(agent_cfg.get('policy','MlpPolicy'), venv, **(agent_cfg.get('params',{})))
        model.learn(total_timesteps=args.timesteps)
        os.makedirs('artifacts/models', exist_ok=True)
        model.save(f'artifacts/models/ppo_{feeder}')
        kpis = eval_model(model, env, episodes=args.episodes)
        rows.append({'feeder': feeder, **kpis})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote summary to {args.out}")

if __name__ == '__main__':
    main()
