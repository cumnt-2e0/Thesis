#!/usr/bin/env python3
"""
Dedicated training script for case118 microgrid environment.

Focus: Line faults + DER degradation + cascading behavior.
Agent learns: (1) Setpoint-based load shedding under generation deficit,
              (2) Fault isolation switching,
              (3) Load restoration after cascading trips resolve.
"""

import os
import logging
import argparse
from pathlib import Path

# Suppress numba & pandapower noise
os.environ["NUMBA_DISABLE_JIT"] = "0"
os.environ["NUMBA_WARNINGS"] = "0"
os.environ["NUMBA_DEBUG"] = "0"
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("numba.core").setLevel(logging.ERROR)
logging.getLogger("pandapower").setLevel(logging.WARNING)

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback, EveryNTimesteps
)
from stable_baselines3.common.logger import configure

from microgrid_safe_rl.utils.logging import setup_logging
from microgrid_safe_rl.utils.seed import set_global_seed
from microgrid_safe_rl.utils.config import load_yaml
from microgrid_safe_rl.envs.factory import make_env


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train RL agent on case118 microgrid")

    # Training parameters
    p.add_argument("--total_timesteps", type=int, default=1_000_000,
                   help="Total training timesteps (default: 1M)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--run_name", type=str, default=None,
                   help="Run name for artifacts (default: auto-generated)")

    # Parallelization
    p.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    p.add_argument("--vec_backend", choices=["dummy", "subproc"], default="subproc",
                   help="Vectorization backend")

    # Features
    p.add_argument("--normalize", action="store_true",
                   help="Enable VecNormalize (observation + reward normalization)")
    p.add_argument("--progress", action="store_true",
                   help="Show training progress bar")

    # Checkpointing / Eval
    p.add_argument("--checkpoint_freq", type=int, default=50_000,
                   help="Save checkpoint every N timesteps")
    p.add_argument("--eval_freq", type=int, default=25_000,
                   help="Evaluate every N timesteps")
    p.add_argument("--eval_episodes", type=int, default=5,
                   help="Number of evaluation episodes")

    # Config (case118 defaults)
    p.add_argument("--agent_cfg", type=str,
                   default="microgrid_safe_rl/configs/agent/ppo.yaml")
    p.add_argument("--env_cfg", type=str,
                   default="microgrid_safe_rl/configs/env_case118.yaml")
    p.add_argument("--scenario_cfg", type=str,
                   default="microgrid_safe_rl/configs/scenario/train/baseline.yaml")

    return p.parse_args()


# =============================================================================
# Callbacks
# =============================================================================

class Case118MetricsCallback(BaseCallback):
    """Log case118-specific metrics to TensorBoard.

    Safe logging: only records keys if present in info.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        info = infos[0] or {}

        # Service metrics (core)
        self.logger.record("metrics/served_total", float(info.get("served_total_frac", 0.0)))
        self.logger.record("metrics/served_critical", float(info.get("served_crit_frac", 0.0)))
        self.logger.record("metrics/served_important", float(info.get("served_imp_frac", 0.0)))

        # System health
        self.logger.record("metrics/pf_success", int(bool(info.get("powerflow_success", False))))
        if "live_lines" in info:
            self.logger.record("metrics/live_lines", float(info["live_lines"]))
        if "energized_buses" in info:
            self.logger.record("metrics/energized_buses", float(info["energized_buses"]))

        # Cascades / restoration (if env exposes them)
        if "cascade_trips" in info:
            self.logger.record("metrics/cascade_trips", float(info["cascade_trips"]))
        if "restored_t1" in info:
            self.logger.record("metrics/restored_t1", float(info["restored_t1"]))
        if "restored_t2" in info:
            self.logger.record("metrics/restored_t2", float(info["restored_t2"]))
        if "restored_t3" in info:
            self.logger.record("metrics/restored_t3", float(info["restored_t3"]))

        # Reward components
        rc = info.get("reward_components", {})
        for key in (
            "service_t1", "service_t2", "service_t3",
            "volt_violation", "thermal_violation", "switch_ops",
            "restore_tier1_delta", "bound_violation"
        ):
            if key in rc:
                self.logger.record(f"reward/{key}", float(rc[key]))

        return True


# =============================================================================
# Environment factory
# =============================================================================

def make_case118_env(env_cfg: dict, scenario: dict):
    """Create case118 environment (called in subprocesses)."""
    import logging as _logging
    _logging.getLogger("numba").setLevel(logging.ERROR)
    _logging.getLogger("pandapower").setLevel(logging.WARNING)
    return make_env("case118", env_cfg, scenario)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Setup
    setup_logging("INFO")
    set_global_seed(args.seed)

    # Run name
    if args.run_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"case118_{timestamp}"

    # Paths
    save_dir = Path("artifacts/models") / args.run_name
    tb_dir = Path("artifacts/runs") / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Training Case118 Microgrid RL Agent")
    print("=" * 80)
    print(f"Run name:        {args.run_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs:   {args.n_envs}")
    print(f"Seed:            {args.seed}")
    print(f"Normalize:       {args.normalize}")
    print(f"Models:          {save_dir}")
    print(f"TensorBoard:     {tb_dir}")
    print("=" * 80, "\n")

    # Load configs
    agent_cfg = load_yaml(args.agent_cfg)
    env_cfg = load_yaml(args.env_cfg)
    scenario = load_yaml(args.scenario_cfg)

    # --- Ensure disturbances apply every episode (if your env honors these keys) ---
    # These setdefault calls are no-ops if the keys don't exist in your scenario schema.
    scenario.setdefault("apply_disturbances_at_reset", True)
    # More explicit knobs if your env supports them:
    scenario.setdefault("der_unavailability", {}).setdefault("enabled", True)
    scenario["der_unavailability"].setdefault("deficit_range", [0.30, 0.40])

    scenario.setdefault("line_faults", {}).setdefault("enabled", True)
    # e.g., 1–3 random line faults per episode; tweak to your liking
    scenario["line_faults"].setdefault("count_range", [1, 3])
    scenario["line_faults"].setdefault("randomize_each_episode", True)

    scenario.setdefault("cascades", {}).setdefault("enabled", True)
    # -------------------------------------------------------------------------------

    # PPO parameters
    ppo_params = agent_cfg.get("ppo", {})
    policy = ppo_params.get("policy", "MlpPolicy")

    # Policy kwargs with activation function mapping
    policy_kwargs = dict(ppo_params.get("policy_kwargs", {}))
    if "activation_fn" in policy_kwargs:
        act_str = str(policy_kwargs["activation_fn"]).lower()
        activation_map = {
            "tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU
        }
        policy_kwargs["activation_fn"] = activation_map.get(act_str, nn.Tanh)

    # Info keys for Monitor (only those you know exist)
    info_keys = (
        "served_total_frac",
        "served_crit_frac",
        "served_imp_frac",
        "powerflow_success",
        "live_lines",
        "energized_buses",
        # Avoid cascade_trips unless your env always sets it; callback logs it safely anyway.
    )

    # Vectorized training env (NO warm-up stepping)
    print("Creating training environments...")
    vec_cls = SubprocVecEnv if (args.vec_backend == "subproc" and args.n_envs > 1) else DummyVecEnv
    train_env_raw = make_vec_env(
        make_case118_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
        env_kwargs={"env_cfg": env_cfg, "scenario": scenario},
        monitor_kwargs={"info_keywords": info_keys},
    )

    if args.normalize:
        train_env = VecNormalize(
            train_env_raw,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
    else:
        train_env = train_env_raw

    print(f"✓ Created {args.n_envs} training environments")

    # Evaluation env (single, deterministic)
    print("Creating evaluation environment...")
    eval_env_raw = make_vec_env(
        make_case118_env,
        n_envs=1,
        seed=args.seed + 999,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"env_cfg": env_cfg, "scenario": scenario},
        monitor_kwargs={"info_keywords": info_keys},
    )
    if args.normalize:
        eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=True, training=False)
        # Share stats (no stepping!)
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
    else:
        eval_env = eval_env_raw

    print("✓ Created evaluation environment\n")

    # PPO model
    print("Creating PPO model...")
    model = PPO(
        policy=policy,
        env=train_env,
        learning_rate=ppo_params.get("learning_rate", 3e-4),
        n_steps=ppo_params.get("n_steps", 2048),
        batch_size=ppo_params.get("batch_size", 64),
        n_epochs=ppo_params.get("n_epochs", 10),
        gamma=ppo_params.get("gamma", 0.99),
        gae_lambda=ppo_params.get("gae_lambda", 0.95),
        clip_range=ppo_params.get("clip_range", 0.2),
        ent_coef=ppo_params.get("ent_coef", 0.01),
        vf_coef=ppo_params.get("vf_coef", 0.5),
        max_grad_norm=ppo_params.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(tb_dir),
        device="auto",
    )
    model.set_logger(configure(str(tb_dir), ["stdout", "tensorboard"]))
    print("✓ PPO model created\n")

    # Callbacks
    callbacks = []

    # Checkpoints
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    if args.checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(save_freq=1, save_path=str(ckpt_dir), name_prefix="ckpt")
        callbacks.append(EveryNTimesteps(n_steps=args.checkpoint_freq, callback=ckpt_cb))

    # Evaluation
    if args.eval_freq > 0:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best"),
            log_path=str(save_dir / "eval_logs"),
            eval_freq=1,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(EveryNTimesteps(n_steps=args.eval_freq, callback=eval_cb))

    # Metrics (includes restoration/cascade if present)
    callbacks.append(Case118MetricsCallback(log_freq=1000))
    callback = CallbackList(callbacks)

    # Train
    print("=" * 80)
    print("Starting training...")
    print("=" * 80, "\n")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=args.progress,
        )
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
        print("=" * 80)

    # Save final artifacts
    final_path = save_dir / "final_model"
    model.save(str(final_path))
    print(f"✓ Saved final model: {final_path}.zip")

    if args.normalize:
        vecnorm_path = save_dir / "final_model_vecnorm.pkl"
        train_env.save(str(vecnorm_path))
        print(f"✓ Saved VecNormalize stats: {vecnorm_path}")

    print("\n" + "=" * 80)
    print("View training progress:")
    print(f"  tensorboard --logdir={tb_dir}\n")
    print("Test the trained agent:")
    print(f"  python microgrid_safe_rl/scripts/evaluate.py \\")
    print(f"    --model {final_path}.zip \\")
    print(f"    --env_cfg {args.env_cfg}")
    print("=" * 80)


if __name__ == "__main__":
    main()
