#!/usr/bin/env python3
"""
Dedicated HOT-START training script for case33 microgrid environment.
Focus: Fine-tune a pre-trained model to handle:
    1. Line faults (new task)
    2. DER degradation (existing task)
"""

import os
import logging
import argparse
from pathlib import Path

# Suppress numba warnings
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['NUMBA_DEBUG'] = '0'
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('numba.core').setLevel(logging.ERROR)
logging.getLogger('pandapower').setLevel(logging.WARNING)

import numpy as np
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


# ============================================================================
# Configuration
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train RL agent on case33 microgrid")
    
    # --- HOT-START: These arguments are CRITICAL ---
    p.add_argument("--init_model", type=str, required=True,
                   help="Path to a .zip model to continue training from")
    p.add_argument("--init_vecnorm", type=str, required=True,
                   help="Path to a .pkl VecNormalize stat file to load")
    
    # --- NEW DEFAULTS FOR FINE-TUNING ---
    p.add_argument("--total_timesteps", type=int, default= 500_000,
                   help="Total training timesteps (default: 500k for fine-tuning)")
    p.add_argument("--run_name", type=str, default="case33_hotstart_faults_3",
                   help="Run name for artifacts (default: case33_hotstart_faults_3)")
    p.add_argument("--env_cfg", type=str,
                   default="microgrid_safe_rl/configs/env_case33.yaml",
                   help="Environment config (default: env_case33.yaml)")
    
    # --- Standard args from original script ---
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--n_envs", type=int, default=8,
                   help="Number of parallel environments")
    p.add_argument("--vec_backend", choices=["dummy", "subproc"], default="subproc",
                   help="Vectorization backend")
    p.add_argument("--normalize", action="store_true", default=True,
                   help="Enable VecNormalize (DEFAULT: True, required for hot-start)")
    p.add_argument("--progress", action="store_true",
                   help="Show training progress bar")
    
    # Checkpointing
    p.add_argument("--checkpoint_freq", type=int, default=50_000,
                   help="Save checkpoint every N timesteps")
    p.add_argument("--eval_freq", type=int, default=25_000,
                   help="Evaluate every N timesteps")
    p.add_argument("--eval_episodes", type=int, default=5,
                   help="Number of evaluation episodes")
    
    # Config overrides
    p.add_argument("--agent_cfg", type=str, 
                   default="microgrid_safe_rl/configs/agent/ppo.yaml")
    p.add_argument("--scenario_cfg", type=str,
                   default="microgrid_safe_rl/configs/scenario/train/baseline.yaml")
    
    return p.parse_args()


# ============================================================================
# Callbacks (Identical to train_case33.py)
# ============================================================================

class Case33MetricsCallback(BaseCallback):
    """Log case33-specific metrics to TensorBoard."""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
        
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        
        info = infos[0]
        
        self.logger.record("metrics/served_total", info.get("served_total_frac", 0.0))
        self.logger.record("metrics/served_critical", info.get("served_crit_frac", 0.0))
        self.logger.record("metrics/served_important", info.get("served_imp_frac", 0.0))
        self.logger.record("metrics/pf_success", int(info.get("powerflow_success", False)))
        self.logger.record("metrics/live_lines", info.get("live_lines", 0))
        self.logger.record("metrics/energized_buses", info.get("energized_buses", 0))
        
        rc = info.get("reward_components", {})
        for key in ("service_t1", "service_t2", "service_t3", 
                   "volt_violation", "thermal_violation", "switch_ops",
                   "restore_tier1_delta", "bound_violation"):
            if key in rc:
                self.logger.record(f"reward/{key}", float(rc[key]))
        
        return True


# ============================================================================
# Environment factory (Identical to train_case33.py)
# ============================================================================

def make_case33_env(env_cfg: dict, scenario: dict):
    """Create case33 environment (called in subprocesses)."""
    import logging
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.getLogger('pandapower').setLevel(logging.WARNING)
    
    return make_env("case33", env_cfg, scenario)


# ============================================================================
# Main training loop (Modified for Hot-Start)
# ============================================================================

def main():
    args = parse_args()
    
    setup_logging("INFO")
    set_global_seed(args.seed)
    
    save_dir = Path("artifacts/models") / args.run_name
    tb_dir = Path("artifacts/runs") / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("HOT-START Training Case33 Microgrid RL Agent (with Faults)")
    print("=" * 80)
    print(f"Run name:        {args.run_name}")
    print(f"Loading model:   {args.init_model}")
    print(f"Loading vecnorm: {args.init_vecnorm}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Env Config:      {args.env_cfg}")
    print(f"Models:          {save_dir}")
    print(f"TensorBoard:     {tb_dir}")
    print("=" * 80)
    print()
    
    # Load configs
    agent_cfg = load_yaml(args.agent_cfg)
    env_cfg = load_yaml(args.env_cfg)
    scenario = load_yaml(args.scenario_cfg)
    
    ppo_params = agent_cfg.get("ppo", {})
    policy = ppo_params.get("policy", "MlpPolicy")
    
    policy_kwargs = ppo_params.get("policy_kwargs", {})
    if "activation_fn" in policy_kwargs:
        act_str = policy_kwargs["activation_fn"].lower()
        activation_map = {
            "tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU,
        }
        policy_kwargs["activation_fn"] = activation_map.get(act_str, nn.Tanh)
    
    info_keys = ("served_total_frac", "served_crit_frac", "served_imp_frac", "powerflow_success")
    
    # Create vectorized training environment
    print("Creating training environments...")
    vec_cls = SubprocVecEnv if (args.vec_backend == "subproc" and args.n_envs > 1) else DummyVecEnv
    
    train_env = make_vec_env(
        make_case33_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
        env_kwargs={"env_cfg": env_cfg, "scenario": scenario},
        monitor_kwargs={"info_keywords": info_keys},
    )
    
    # --- VECNORMALIZE LOADING ---
    # We assume normalization was used in the first run, as it's critical
    if not args.normalize:
        print("WARNING: --normalize flag is OFF, but hot-starting typically")
        print("         requires loading old VecNormalize stats.")
        print("         If the original model used VecNormalize, this WILL FAIL.")
    
    if args.init_vecnorm:
        print(f"Loading VecNormalize stats from: {args.init_vecnorm}")
        train_env = VecNormalize.load(args.init_vecnorm, train_env)
        train_env.training = True  # IMPORTANT: Set to training mode
        print("✓ VecNormalize stats loaded.")
    else:
        print("ERROR: --init_vecnorm path is required for hot-starting.")
        print("       Please provide the .pkl file from the previous run.")
        return

    print(f"✓ Created {args.n_envs} training environments")
    
    # Create evaluation environment (single env, deterministic)
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        make_case33_env,
        n_envs=1,
        seed=args.seed + 999,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"env_cfg": env_cfg, "scenario": scenario},
        monitor_kwargs={"info_keywords": info_keys},
    )
    
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        training=False,  # Don't update stats during eval
    )
    # Copy normalization stats from training env
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    print("✓ Created evaluation environment")
    print()
    
    # --- MODEL LOADING ---
    if not args.init_model:
        print("ERROR: --init_model path is required for hot-starting.")
        return

    print(f"Loading and continuing from: {args.init_model}")
    model = PPO.load(
        args.init_model,
        env=train_env,  # Pass the (wrapped) env
        tensorboard_log=str(tb_dir),
        policy_kwargs=policy_kwargs, # Re-apply policy kwargs
        # You can optionally reduce the learning rate for fine-tuning
        # learning_rate=1e-5, 
    )
    model.set_logger(configure(str(tb_dir), ["stdout", "tensorboard"]))
    print("✓ Model loaded.")
    
    
    # Setup callbacks
    callbacks = []
    
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    if args.checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(
            save_freq=1,
            save_path=str(ckpt_dir),
            name_prefix="ckpt",
        )
        callbacks.append(EveryNTimesteps(n_steps=args.checkpoint_freq, callback=ckpt_cb))
    
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
    
    callbacks.append(Case33MetricsCallback(log_freq=1000))
    
    callback = CallbackList(callbacks)
    
    # Train!
    print("=" * 80)
    print("Starting fine-tuning...")
    print("=" * 80)
    print()
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=args.progress,
            reset_num_timesteps=False # IMPORTANT: Continue step count from loaded model
        )
        
        print()
        print("=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("Training interrupted by user")
        print("=" * 80)
    
    # Save final model
    final_path = save_dir / "final_model"
    model.save(str(final_path))
    print(f"✓ Saved final model: {final_path}.zip")
    
    vecnorm_path = save_dir / "final_model_vecnorm.pkl"
    train_env.save(str(vecnorm_path))
    print(f"✓ Saved VecNormalize stats: {vecnorm_path}")
    
    print()
    print("=" * 80)
    print("View training progress:")
    print(f"  tensorboard --logdir={tb_dir.parent}")
    print("=" * 80)


if __name__ == "__main__":
    main()