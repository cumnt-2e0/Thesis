import argparse
from microgrid_safe_rl.utils.config import load_yaml
from microgrid_safe_rl.envs.factory import make_env

FEEDERS = ["ieee33","ieee123","cigre_mv","rte1888"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="*", default=FEEDERS)
    p.add_argument('--env_cfg', type=str, default='env.yaml')
    p.add_argument('--scenario_cfg', type=str, default='scenario/baseline.yaml')
    args = p.parse_args()

    env_cfg = load_yaml(args.env_cfg)
    scenario = load_yaml(args.scenario_cfg)

    for name in args.envs:
        print(f"== {name} ==")
        env = make_env(name, env_cfg, scenario)
        obs, info = env.reset()
        print("obs shape:", obs.shape, "faulted_line:", info.get("faulted_line"))
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print("step -> r:", r, "term:", term, "voltages in res_bus:", len(getattr(env.net, "res_bus", [])))

if __name__ == "__main__":
    main()
