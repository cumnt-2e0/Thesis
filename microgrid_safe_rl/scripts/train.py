import argparse, os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO

from microgrid_safe_rl.utils.logging import setup_logging
from microgrid_safe_rl.utils.seed import set_global_seed
from microgrid_safe_rl.utils.config import load_yaml
from microgrid_safe_rl.envs.factory import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="case33",choices=["case33", "case145", "case300", "case1888"])
    p.add_argument("--total_timesteps", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="artifacts/models")

    # progress / logging
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bar during training")
    p.add_argument("--log_interval", type=int, default=10, help="SB3 log interval (in rollouts/episodes)")

    # tensorboard
    p.add_argument("--tb_logdir", type=str, default="artifacts/runs", help="TensorBoard log dir (set empty to disable)")

    # periodic evaluation
    p.add_argument("--eval_every", type=int, default=0, help="Evaluate every N steps (0 to disable)")
    p.add_argument("--eval_episodes", type=int, default=3, help="Episodes per evaluation")

    # configs
    p.add_argument("--agent_cfg", type=str, default="agent/ppo.yaml", help="Path or package-relative")
    p.add_argument("--env_cfg", type=str, default="env.yaml", help="Path or package-relative")
    p.add_argument("--scenario_cfg", type=str, default="scenario/baseline.yaml", help="Path or package-relative")
    args = p.parse_args()

    setup_logging("INFO")
    set_global_seed(args.seed)

    agent_cfg = load_yaml(args.agent_cfg)
    env_cfg = load_yaml(args.env_cfg)
    scenario = load_yaml(args.scenario_cfg)

    # --- training env ---
    train_env = make_env(args.env_id, env_cfg, scenario)
    venv = DummyVecEnv([lambda: Monitor(train_env)])

    # --- model w/ tensorboard ---
    tb_dir = args.tb_logdir or None
    model = PPO(
        agent_cfg.get("policy", "MlpPolicy"),
        venv,
        verbose=1,
        tensorboard_log=tb_dir,
        **(agent_cfg.get("params", {})),
    )
    if tb_dir:
        os.makedirs(tb_dir, exist_ok=True)
        # also log to stdout for instant updates
        model.set_logger(configure(tb_dir, ["stdout", "tensorboard"]))

    # --- optional periodic evaluation ---
    callback = None
    if args.eval_every and args.eval_every > 0:
        eval_env = DummyVecEnv([lambda: Monitor(make_env(args.env_id, env_cfg, scenario))])
        callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.save_dir, f"best_{args.env_id}"),
            log_path=os.path.join(args.save_dir, f"eval_{args.env_id}"),
            eval_freq=args.eval_every,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
            warn=True,
        )

    # --- train ---
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        progress_bar=bool(args.progress),  # requires SB3 >= 2.0 and tqdm
        callback=callback,
    )

    # --- save ---
    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, f"ppo_{args.env_id}")
    model.save(out)
    print(f"Saved model: {out}")


if __name__ == "__main__":
    main()
