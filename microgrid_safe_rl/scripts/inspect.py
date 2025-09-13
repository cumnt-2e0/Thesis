import argparse
from microgrid_safe_rl.utils.config import load_yaml
from microgrid_safe_rl.envs.factory import make_env

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True)
    p.add_argument("--env_cfg", default="env.yaml")
    p.add_argument("--scenario_cfg", default="scenario/line_only.yaml")
    args = p.parse_args()

    env = make_env(args.env_id, load_yaml(args.env_cfg), load_yaml(args.scenario_cfg))
    buses = len(env.net.bus)
    K = len(env.switch_ids)
    L = len(env.load_ids)
    print(f"env={args.env_id}")
    print(f"  buses={buses}, switches(K)={K}, loads(L)={L}")
    print(f"  obs_dim={env.observation_space.shape[0]}, actions={env.action_space.n}")

if __name__ == "__main__":
    main()
