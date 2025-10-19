#!/usr/bin/env python3
import os
import logging

# CRITICAL: Disable Numba logging BEFORE any imports that might trigger it
os.environ['NUMBA_DISABLE_JIT'] = '0'  # Keep JIT enabled
os.environ['NUMBA_WARNINGS'] = '0'     # Disable warnings
os.environ['NUMBA_DEBUG'] = '0'        # Disable debug
os.environ['NUMBA_DEVELOPER_MODE'] = '0'

# Set up logging before any imports
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('numba.core').setLevel(logging.ERROR)
logging.getLogger('numba.core.ssa').setLevel(logging.ERROR)
logging.getLogger('numba.core.byteflow').setLevel(logging.ERROR)
logging.getLogger('numba.core.interpreter').setLevel(logging.ERROR)
logging.getLogger('numba.core.ssa').setLevel(logging.ERROR)
logging.getLogger('pandapower').setLevel(logging.WARNING)

import argparse
from typing import Tuple, Dict, Any, Optional

import torch.nn as nn
from stable_baselines3 import PPO, SAC
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

ALGOS = {"ppo": PPO, "sac": SAC}

# Whitelist of SB3 kwarg names we allow to reach the model __init__
_ALLOWED_KEYS = {
    # Common
    "policy", "learning_rate", "gamma", "device", "policy_kwargs",
    # PPO
    "n_steps", "batch_size", "n_epochs", "gae_lambda",
    "clip_range", "clip_range_vf", "ent_coef", "vf_coef",
    "max_grad_norm", "use_sde", "sde_sample_freq", "normalize_advantage",
    # SAC
    "buffer_size", "tau", "train_freq", "gradient_steps", "learning_starts",
    "target_update_interval", "target_entropy",
}

# Map friendly strings to torch activations
_ACTIVATION_MAP = {
    "tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU, "selu": nn.SELU, "sigmoid": nn.Sigmoid,
}

# ------------------------- CLI ------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    # Env / training
    p.add_argument("--env_id", type=str, default="case33",
                   choices=["case33", "case145", "case300", "case1888"])
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="artifacts/models/s4",
                   help="All model files (checkpoints/best/final) land under here")

    # Algo + warm start
    p.add_argument("--algo", choices=list(ALGOS.keys()), default=None,
                   help="SB3 algorithm (overrides agent_cfg.algo if set)")
    p.add_argument("--init_model", type=str, default=None,
                   help="Path to SB3 .zip to continue training from")

    # Vectorization (multi-core)
    p.add_argument("--n_envs", type=int, default=1,
                   help="Number of parallel envs (use CPU cores here)")
    p.add_argument("--vec_backend", choices=["dummy", "subproc"], default="subproc",
                   help="Vectorized env backend")
    p.add_argument("--normalize", action="store_true",
                   help="Wrap with VecNormalize (obs & reward)")

    # Logging / TB
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bar")
    p.add_argument("--log_interval", type=int, default=10, help="SB3 log interval (episodes)")
    p.add_argument("--tb_logdir", type=str, default="artifacts/runs/s4",
                   help="TensorBoard dir; TB logs go exactly here")

    # Config files
    p.add_argument("--agent_cfg", type=str, default="agent/ppo.yaml", help="Agent YAML")
    p.add_argument("--env_cfg", type=str, default="env.yaml", help="Environment YAML")
    p.add_argument("--scenario_cfg", type=str, default="scenario/baseline.yaml", help="Scenario YAML")

    # Checkpointing / eval (values are in TRUE TIMESTEPS)
    p.add_argument("--checkpoint_freq", type=int, default=100_000,
                   help="Save a checkpoint every N timesteps")
    p.add_argument("--eval_freq", type=int, default=50_000,
                   help="Run evaluation every N timesteps")
    p.add_argument("--eval_episodes", type=int, default=5)
    return p.parse_args()

# ----------------------- Helpers ----------------------- #
def _coerce_activation(policy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(policy_kwargs, dict):
        return policy_kwargs
    act = policy_kwargs.get("activation_fn", None)
    if isinstance(act, str):
        act_l = act.strip().lower()
        if act_l in _ACTIVATION_MAP:
            policy_kwargs["activation_fn"] = _ACTIVATION_MAP[act_l]
    return policy_kwargs

def _extract_algo_and_params(agent_cfg: dict, cli_algo: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Accepts:
      1) {'algo': 'ppo', 'ppo': {...}}
      2) {'policy': 'MlpPolicy', 'params': {...}}
      3) flat dict with SB3 kwargs (legacy)
    Returns (algo_name, params_dict).
    """
    algo_cfg = (cli_algo or agent_cfg.get("algo") or "ppo").lower()

    # 1) Preferred nested block
    if algo_cfg in agent_cfg and isinstance(agent_cfg[algo_cfg], dict):
        block = dict(agent_cfg[algo_cfg])
        block["policy"] = block.get("policy", "MlpPolicy")
        if "policy_kwargs" in block:
            block["policy_kwargs"] = _coerce_activation(block["policy_kwargs"])
        params = {k: v for k, v in block.items() if k in _ALLOWED_KEYS}
        return algo_cfg, params

    # 2) {policy, params}
    if "policy" in agent_cfg or "params" in agent_cfg:
        policy = agent_cfg.get("policy", "MlpPolicy")
        params = dict(agent_cfg.get("params", {}))
        params["policy"] = policy
        if "policy_kwargs" in params:
            params["policy_kwargs"] = _coerce_activation(params["policy_kwargs"])
        params = {k: v for k, v in params.items() if k in _ALLOWED_KEYS}
        return (cli_algo or agent_cfg.get("algo", "ppo")).lower(), params

    # 3) Flat legacy
    flat = {k: v for k, v in agent_cfg.items() if k != "algo"}
    if "policy_kwargs" in flat:
        flat["policy_kwargs"] = _coerce_activation(flat["policy_kwargs"])
    flat = {k: v for k, v in flat.items() if k in _ALLOWED_KEYS}
    flat.setdefault("policy", "MlpPolicy")
    return (cli_algo or agent_cfg.get("algo", "ppo")).lower(), flat

# Compact cascade debug → TB
class CascadeDebugCallback(BaseCallback):
    def __init__(self, every_steps: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.n = every_steps

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        info = infos[0]
        if self.num_timesteps % self.n == 0:
            self.logger.record("debug/served_total_frac", info.get("served_total_frac", float("nan")))
            self.logger.record("debug/cascade_remaining", info.get("cascade_remaining", -1))
            self.logger.record("debug/fault_live", int(info.get("fault_live", False)))
            if "cascade_tripped" in info:
                self.logger.record("debug/cascade_tripped_last", len(info["cascade_tripped"]))
            rc = info.get("reward_components", {})
            for k in ("fault", "fault_live", "restore_delta", "volt", "shed"):
                if k in rc:
                    self.logger.record(f"rew/{k}", float(rc[k]))
        return True

# Top-level env factory for subproc safety
def _env_ctor(env_id: str, env_cfg: dict, scenario: dict):
    # Suppress numba in child processes too
    import logging
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.getLogger('numba.core').setLevel(logging.ERROR)
    return make_env(env_id, env_cfg, scenario)

# ------------------------- Main ------------------------ #
def main():
    args = parse_args()
    
    setup_logging("INFO")
    
    if args.env_id in ["case33"]:
        logging.getLogger("microgrid_safe_rl.envs").setLevel(logging.DEBUG)
    
    set_global_seed(args.seed)

    agent_cfg = load_yaml(args.agent_cfg)
    env_cfg = load_yaml(args.env_cfg)
    scenario = load_yaml(args.scenario_cfg)

    algo_name, model_kwargs = _extract_algo_and_params(agent_cfg, args.algo)
    if algo_name not in ALGOS:
        raise SystemExit(f"Unsupported algo '{algo_name}'. Supported: {list(ALGOS)}")
    Model = ALGOS[algo_name]

    info_keys = (
        "served_total_frac", "served_crit_frac", "served_imp_frac",
        "fault_live", "isolation_happened", "powerflow_success",
        "cascade_remaining", "der_bound_frac", "served_mw", "der_bound_mw",
    )

    # Vectorized env
    vec_cls = SubprocVecEnv if (args.vec_backend == "subproc" and args.n_envs > 1) else DummyVecEnv
    venv = make_vec_env(
        _env_ctor,
        n_envs=max(1, args.n_envs),
        seed=args.seed,
        vec_env_cls=vec_cls,
        env_kwargs={"env_id": args.env_id, "env_cfg": env_cfg, "scenario": scenario},
        monitor_kwargs={"info_keywords": info_keys},
    )

    # CRITICAL FIX: VecNormalize warmup
    vecnorm: Optional[VecNormalize] = None
    if args.normalize:
        print("[train] Initializing VecNormalize with warmup...")
        vecnorm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        
        # Warmup: collect some samples to initialize running statistics
        vecnorm.reset()
        warmup_steps = 100
        print(f"[train] Running {warmup_steps} warmup steps to initialize normalization stats...")
        
        for i in range(warmup_steps):
            # CRITICAL: Must be numpy array, not list
            actions = np.array([venv.action_space.sample() for _ in range(args.n_envs)])
            obs, rewards, dones, infos = venv.step(actions)
            
            # Check for any issues during warmup
            if i % 20 == 0:
                if not np.all(np.isfinite(obs)):
                    print(f"WARNING: Non-finite observations at warmup step {i}")
                if not np.all(np.isfinite(rewards)):
                    print(f"WARNING: Non-finite rewards at warmup step {i}: {rewards}")
        
        print(f"[train] Warmup complete. Obs mean: {vecnorm.obs_rms.mean[:5]}, std: {np.sqrt(vecnorm.obs_rms.var[:5])}")
        vecnorm.reset()
        venv = vecnorm

    # Model creation
    tb_dir = args.tb_logdir or None
    if args.init_model:
        print(f"[train] Loading init model from: {args.init_model}")
        model = Model.load(args.init_model, env=venv, print_system_info=False)
        model.verbose = 1
    else:
        policy = model_kwargs.pop("policy", "MlpPolicy")
        model = Model(policy, venv, verbose=1, tensorboard_log=tb_dir, **model_kwargs)


    # Logger
    if tb_dir:
        os.makedirs(tb_dir, exist_ok=True)
        model.set_logger(configure(tb_dir, ["stdout", "tensorboard"]))

    # ----- Callbacks (by TRUE TIMESTEPS) -----
    callbacks = []

    # Checkpoints
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(save_freq=1, save_path=ckpt_dir, name_prefix="ckpt")
        callbacks.append(EveryNTimesteps(n_steps=args.checkpoint_freq, callback=ckpt_cb))

    # Eval on separate single env
    eval_env = make_vec_env(
        _env_ctor, n_envs=1, seed=args.seed + 42, vec_env_cls=DummyVecEnv,
        env_kwargs={"env_id": args.env_id, "env_cfg": env_cfg, "scenario": scenario},
        monitor_kwargs={"info_keywords": info_keys},
    )
    if args.normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
        if isinstance(venv, VecNormalize):
            eval_env.obs_rms = venv.obs_rms
            eval_env.ret_rms = venv.ret_rms

    best_dir = os.path.join(args.save_dir, "best")
    eval_log_dir = os.path.join(args.save_dir, "eval_logs")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    if args.eval_freq > 0:
        eval_cb = EvalCallback(
            eval_env, best_model_save_path=best_dir, log_path=eval_log_dir,
            eval_freq=1, n_eval_episodes=args.eval_episodes,
            deterministic=True, render=False
        )
        callbacks.append(EveryNTimesteps(n_steps=args.eval_freq, callback=eval_cb))

    # Compact cascade debug → TB
    callbacks.append(CascadeDebugCallback(every_steps=1_000))

    callback = CallbackList(callbacks)

    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        progress_bar=bool(args.progress),
        callback=callback,
    )

    # Save final model (+ VecNormalize stats if used)
    os.makedirs(args.save_dir, exist_ok=True)
    final_path = os.path.join(args.save_dir, f"{algo_name}_{args.env_id}_final")
    model.save(final_path)
    if isinstance(venv, VecNormalize):
        venv.save(final_path + "_vecnorm.pkl")
    print(f"Saved model: {final_path}")

if __name__ == "__main__":
    main()