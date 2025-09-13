#!/usr/bin/env python3
"""
Per-step evaluation and logging.

Usage:
  python -m microgrid_safe_rl.scripts.eval_per_step \
    --env_id case145 \
    --model_path artifacts/models/ppo_case145 \
    --episodes 30 --env_cfg env.yaml \
    --scenario_cfg scenario/eval_very_hard.yaml \
    --out artifacts/models/case145_eval_steps.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from microgrid_safe_rl.envs.factory import make_env
from microgrid_safe_rl.utils.config import load_yaml
# Reuse the model loader from the regular eval script
from microgrid_safe_rl.scripts.eval import load_model


def served_fractions(net) -> tuple[float, float, float]:
    """(crit_served, imp_served, total_served) ∈ [0,1]."""
    if not hasattr(net, "load") or len(net.load) == 0:
        return 1.0, 1.0, 1.0

    df = net.load
    base = df["p_base_mw"].astype(float) if "p_base_mw" in df.columns else df["p_mw"].astype(float)
    cur  = df["p_mw"].astype(float)
    prio = df["priority"] if "priority" in df.columns else np.zeros(len(df), dtype=int)

    def frac(mask):
        denom = float(base[mask].sum())
        return float(cur[mask].sum()) / denom if denom > 1e-9 else 1.0

    f_crit = frac(prio == 2)
    f_imp  = frac(prio == 1)
    denomT = float(base.sum())
    f_tot  = float(cur.sum()) / denomT if denomT > 1e-9 else 1.0
    return float(np.clip(f_crit, 0, 1)), float(np.clip(f_imp, 0, 1)), float(np.clip(f_tot, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--env_cfg", default="env.yaml")
    ap.add_argument("--scenario_cfg", default=None)
    ap.add_argument("--algo", default="ppo", help="Used only if the loader needs it; PPO by default.")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    # Build env and load model
    env_cfg = load_yaml(args.env_cfg) if args.env_cfg else {}
    scen_cfg = load_yaml(args.scenario_cfg) if args.scenario_cfg else {}
    env = make_env(args.env_id, env_cfg, scen_cfg)
    model = load_model(args.model_path, args.algo, env)

    rows = []
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        cum_r = 0.0
        done = False
        truncated = False
        step_idx = 0

        # Log step-by-step until termination
        while not (done or truncated):
            step_idx += 1
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, info = env.step(action)
                cum_r += float(r)
            except Exception as e:
                # If something goes wrong, mark PF failure and continue to exit loop
                info = (info or {})
                info["powerflow_success"] = False
                r = 0.0
                done = True

            f2, f1, fT = served_fractions(env.net)

            rows.append(dict(
                episode=ep,
                step=step_idx,
                reward_step=float(r),
                reward_cum=float(cum_r),
                served_crit=f2,
                served_imp=f1,
                served_total=fT,
                fault_live=bool(getattr(env, "_fault_live", False)),
                pf_success=bool(info.get("powerflow_success", True)),
                toggled_switch=int(info.get("toggled_switch", -1)) if "toggled_switch" in info else -1,
                shedding_load=int(info.get("shedding_load", -1)) if "shedding_load" in info else -1,
            ))

            # safety guard if env doesn't expose max_steps
            if step_idx >= int(getattr(env.unwrapped, "max_steps", 10_000)):
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved per-step CSV: {out}")


if __name__ == "__main__":
    main()
