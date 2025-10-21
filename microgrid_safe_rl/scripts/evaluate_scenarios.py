#!/usr/bin/env python3
"""
Evaluate a trained RL agent against a baseline for specific fault scenarios.

Compares the agent's ability to restore load using tie switches and shedding
against a "do-nothing" baseline under DER deficiency and a line fault.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.patches as mpatches # Added for legend


# Ensure project root is discoverable or adjust imports as needed
from microgrid_safe_rl.envs.factory import make_env
from microgrid_safe_rl.utils.logging import setup_logging
from microgrid_safe_rl.utils.seed import set_global_seed

# Configure logging
log = logging.getLogger("evaluate_scenarios")
setup_logging("INFO") # Set desired logging level

# --- Define Scenarios ---
# Each dict defines: line_idx (faulted line), fault_step
SCENARIOS = {
    "Scenario 1 (Fault L2)": {"line_idx": 2, "fault_step": 10},
    "Scenario 2 (Fault L18)": {"line_idx": 18, "fault_step": 10},
    "Scenario 3 (Fault L6)": {"line_idx": 6, "fault_step": 10},
    "Scenario 4 (Fault L25)": {"line_idx": 25, "fault_step": 10},
}

# --- Helper Function to Run One Episode ---

def run_episode(env, agent, fault_info: dict, seed: int):
    """Runs a single episode, injects fault, returns metrics."""
    metrics = {
        "steps": [],
        "reward": [],
        "served_total_frac": [],
        "served_crit_frac": [],
        "served_imp_frac": [],
        "pf_success": [],
        "live_lines": [],
        "energized_buses": [],
        "volt_violation": [],
        "thermal_violation": [],
        "bound_violation": [],
        "switches_closed": [],
        "fault_occurred": False,
    }
    start_time = time.time()
    
    # Initialize step counter here to solve UnboundLocalError in case of early exception
    step = 0 

    try:
        # --- FIX 1: Use env.seed() then env.reset() ---
        env.seed(seed)
        obs = env.reset()
        log.debug(f"Reset complete for seed {seed}.")
        # --- End FIX 1 ---

        # --- FIX 2: Use env.get_attr() ---
        # This safely gets the attribute from the wrapped env
        # It returns a list, so we take the first (and only) element.
        no_op_action = env.get_attr('_switch_count')[0]
        max_steps = env.get_attr('max_steps')[0]
        # --- End FIX 2 ---

        terminated = truncated = False

        while not (terminated or truncated):
            current_step_for_logic = step + 1
            metrics["steps"].append(current_step_for_logic)

            # --- Fault Injection ---
            if current_step_for_logic == fault_info["fault_step"] and not metrics["fault_occurred"]:
                fault_line_idx = fault_info["line_idx"]
                try:
                    # --- FIX 3: Use env.get_attr() to get env instance ---
                    actual_env = env.get_attr('unwrapped')[0]
                    
                    if fault_line_idx in actual_env.net.line.index:
                        actual_env.net.line.loc[fault_line_idx, "in_service"] = False
                        if hasattr(actual_env, '_tripped'):
                             actual_env._tripped.add(int(fault_line_idx))
                        metrics["fault_occurred"] = True
                        log.info(f"Seed {seed}, Step {current_step_for_logic}: Manually injected fault on line {fault_line_idx}")
                        
                        # --- FIX 4: Remove private access to _runpf() ---
                        # The env.step() call below will handle running the PF
                        # _ = actual_env._runpf() 
                        # --- End FIX 4 ---
                    else:
                        log.warning(f"Seed {seed}, Step {current_step_for_logic}: Fault line {fault_line_idx} not found in net.line.index!")
                except Exception as e_fault:
                    log.error(f"Seed {seed}, Step {current_step_for_logic}: Error injecting fault on line {fault_line_idx}: {e_fault}")
            # --- End Fault Injection ---

            # --- Action Selection ---
            if agent:
                action, _ = agent.predict(obs, deterministic=True)
            else: # Baseline
                action = np.array([no_op_action])
            # --- End Action Selection ---

            # --- FIX 5: Correct env.step() unpacking (4 values) ---
            obs, reward, done, info = env.step(action)
            # --- End FIX 5 ---

            step += 1 # Increment loop counter

            # Extract metrics from info (list of dicts)
            info_dict = info[0]
            reward_val = reward[0]

            log_step = info_dict.get("episode_step", current_step_for_logic)
            log.debug(f"  Step {log_step}: Action={action[0]}, Reward={reward_val:.2f}")

            metrics["reward"].append(reward_val)
            metrics["served_total_frac"].append(info_dict.get("served_total_frac", 0.0))
            metrics["served_crit_frac"].append(info_dict.get("served_crit_frac", 0.0))
            metrics["served_imp_frac"].append(info_dict.get("served_imp_frac", 0.0))
            metrics["pf_success"].append(info_dict.get("powerflow_success", False))
            metrics["live_lines"].append(info_dict.get("live_lines", -1))
            metrics["energized_buses"].append(info_dict.get("energized_buses", -1))

            rc = info_dict.get("reward_components", {})
            metrics["volt_violation"].append(rc.get("volt_violation", 0.0))
            metrics["thermal_violation"].append(rc.get("thermal_violation", 0.0))
            metrics["bound_violation"].append(rc.get("bound_violation", 0.0))

            try:
                 # --- FIX 6: Use env.get_attr() ---
                 sw_closed = env.get_attr('net')[0].switch['closed'].sum()
                 metrics["switches_closed"].append(sw_closed)
            except Exception:
                 metrics["switches_closed"].append(np.nan)

            # --- FIX 7: Corrected Termination Handling ---
            terminated = done[0] # done is an array for VecEnv
            truncated = info_dict.get("TimeLimit.truncated", False) 

            if step >= max_steps:
                 truncated = True
                 terminated = True # Force loop to end
            # --- End FIX 7 ---

    except Exception as e:
        log.error(f"Error during episode run (seed {seed}): {e}", exc_info=True)
        # Add NaNs to metrics lists to maintain structure
        # 'step' is now defined, so this part is safer
        max_len = max(step + 1, max(len(v) for v in metrics.values() if isinstance(v, list)))
        for k, v in metrics.items():
            if isinstance(v, list):
                 pad_len = max_len - len(v)
                 if pad_len > 0:
                      v.extend([np.nan] * pad_len)

    # Calculate summary stats
    total_reward = sum(m for m in metrics["reward"] if np.isfinite(m))
    final_served = metrics["served_total_frac"][-1] if metrics["served_total_frac"] and pd.notna(metrics["served_total_frac"][-1]) else 0.0
    num_steps = len(metrics["steps"])
    run_time = time.time() - start_time

    volt_penalties = [v for v in metrics["volt_violation"] if v < -1e-6]
    therm_penalties = [v for v in metrics["thermal_violation"] if v < -1e-6]
    bound_penalties = [v for v in metrics["bound_violation"] if v < -1e-6]

    summary = {
        "total_reward": total_reward,
        "final_served_total_frac": final_served,
        "steps_completed": num_steps,
        "fault_occurred": metrics["fault_occurred"],
        "run_time_s": run_time,
        "pf_failures": sum(1 for pf in metrics["pf_success"] if not pf),
        "avg_volt_viol": np.mean(volt_penalties) if volt_penalties else 0.0,
        "avg_therm_viol": np.mean(therm_penalties) if therm_penalties else 0.0,
        "avg_bound_viol": np.mean(bound_penalties) if bound_penalties else 0.0,
    }

    log.debug(f"Episode finished (seed {seed}). Steps: {num_steps}, Reward: {total_reward:.2f}, Time: {run_time:.1f}s")
    return metrics, summary


# --- Main Evaluation Logic ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate RL agent vs Baseline on fault scenarios")
    parser.add_argument("--model", required=True, type=str, help="Path to the trained agent .zip file")
    parser.add_argument("--vecnorm", required=True, type=str, help="Path to the VecNormalize .pkl file")
    parser.add_argument("--env_cfg", required=True, type=str, help="Path to the environment config YAML file used for training/evaluation")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Number of episodes to run per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for evaluations")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save plots and results")
    args = parser.parse_args()

    # --- Setup ---
    set_global_seed(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Starting evaluation...")
    log.info(f"Model: {args.model}")
    log.info(f"VecNorm: {args.vecnorm}")
    log.info(f"Env Config: {args.env_cfg}")
    log.info(f"Episodes/Scenario: {args.n_eval_episodes}")
    log.info(f"Output Dir: {output_path.resolve()}")

    # --- Load Agent ---
    try:
        agent = PPO.load(args.model, device='cpu')
        log.info("Agent loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load agent: {e}", exc_info=True)
        return

    # --- Load Env Config ---
    try:
        with open(args.env_cfg, 'r') as f:
            env_config = yaml.safe_load(f)
        log.info("Environment config loaded.")
        env_config.setdefault('cascade', {})['enabled'] = True
        env_config.setdefault('cascade', {}).setdefault('seed', {})['n_seeds'] = 0
        log.info("Modified env config: Disabled random fault seeds for controlled evaluation.")
    except Exception as e:
        log.error(f"Failed to load environment config: {e}", exc_info=True)
        return

    # --- Run Evaluations ---
    all_results = {} 

    for scenario_name, fault_info in SCENARIOS.items():
        log.info(f"\n--- Running Scenario: {scenario_name} (Fault Line {fault_info['line_idx']} @ Step {fault_info['fault_step']}) ---")
        all_results[scenario_name] = {'agent': [], 'baseline': []}

        for i in range(args.n_eval_episodes):
            episode_seed = args.seed + i
            log.info(f"  Episode {i+1}/{args.n_eval_episodes} (Seed: {episode_seed})...")

            def create_eval_env():
                 return make_env("case33", env_config, scenario={})

            # Agent Run
            eval_env_agent = None 
            try:
                log.debug("  Setting up environment for AGENT run...")
                eval_env_agent = make_vec_env(create_eval_env, n_envs=1, vec_env_cls=DummyVecEnv)
                eval_env_agent = VecNormalize.load(args.vecnorm, eval_env_agent)
                eval_env_agent.training = False 
                eval_env_agent.norm_reward = False 
                log.debug("  Running AGENT episode...")
                metrics_agent, summary_agent = run_episode(eval_env_agent, agent, fault_info, episode_seed)
                all_results[scenario_name]['agent'].append({"metrics": metrics_agent, "summary": summary_agent})
            except Exception as e:
                 log.error(f"  AGENT run failed for seed {episode_seed}: {e}", exc_info=True)
                 all_results[scenario_name]['agent'].append({"metrics": {}, "summary": {}})
            finally:
                if eval_env_agent: eval_env_agent.close() 


            # Baseline Run
            eval_env_baseline = None
            try:
                log.debug("  Setting up environment for BASELINE run...")
                eval_env_baseline = make_vec_env(create_eval_env, n_envs=1, vec_env_cls=DummyVecEnv)
                eval_env_baseline = VecNormalize.load(args.vecnorm, eval_env_baseline)
                eval_env_baseline.training = False
                eval_env_baseline.norm_reward = False
                log.debug("  Running BASELINE episode...")
                metrics_base, summary_base = run_episode(eval_env_baseline, None, fault_info, episode_seed) # Agent=None
                all_results[scenario_name]['baseline'].append({"metrics": metrics_base, "summary": summary_base})
            except Exception as e:
                 log.error(f"  BASELINE run failed for seed {episode_seed}: {e}", exc_info=True)
                 all_results[scenario_name]['baseline'].append({"metrics": {}, "summary": {}})
            finally:
                if eval_env_baseline: eval_env_baseline.close()


    # --- Process and Plot Results ---
    log.info("\n--- Processing and Plotting Results ---")
    num_scenarios = len(SCENARIOS)
    fig, axes = plt.subplots(num_scenarios, 1, figsize=(10, 5 * num_scenarios), sharex=True, squeeze=False)
    axes = axes.flatten()

    summary_stats = [] 
    
    # Get max_steps from config one last time for plotting range
    max_steps_plot = env_config.get('max_steps', 120)
    steps_index = np.arange(1, max_steps_plot + 1)

    for i, (scenario_name, results) in enumerate(all_results.items()):
        ax = axes[i]
        fault_step = SCENARIOS[scenario_name]['fault_step']

        agent_metrics_list = [res["metrics"] for res in results["agent"] if res["metrics"]]
        baseline_metrics_list = [res["metrics"] for res in results["baseline"] if res["metrics"]]

        def get_mean_std_over_time(metrics_list, metric_key):
             all_series = []
             for m in metrics_list:
                  # Check if metrics dict is not empty and keys exist
                  if m and metric_key in m and 'steps' in m and m['steps']:
                       # Create Series, ensuring index aligns with steps recorded
                       s = pd.Series(m[metric_key], index=m['steps'])
                       # Reindex to full step range, forward fill missing values
                       s = s.reindex(steps_index).ffill()
                       all_series.append(s)

             if not all_series:
                 log.warning(f"No valid data found for metric '{metric_key}' in scenario '{scenario_name}'")
                 return pd.Series(np.nan, index=steps_index), pd.Series(np.nan, index=steps_index)

             combined_df = pd.concat(all_series, axis=1)
             mean_ts = combined_df.mean(axis=1)
             std_ts = combined_df.std(axis=1).fillna(0)
             return mean_ts, std_ts


        agent_mean, agent_std = get_mean_std_over_time(agent_metrics_list, 'served_total_frac')
        base_mean, base_std = get_mean_std_over_time(baseline_metrics_list, 'served_total_frac')

        # Plotting
        ax.plot(steps_index, agent_mean, label="Agent", color="blue", linewidth=2)
        ax.fill_between(steps_index,
                        (agent_mean - agent_std).clip(lower=0), 
                        (agent_mean + agent_std).clip(upper=1.05),
                        color="blue", alpha=0.2)

        ax.plot(steps_index, base_mean, label="Baseline (No Action)", color="red", linestyle="--", linewidth=2)
        ax.fill_between(steps_index,
                        (base_mean - base_std).clip(lower=0),
                        (base_mean + base_std).clip(upper=1.05),
                        color="red", alpha=0.2)

        ax.axvline(fault_step, color='black', linestyle=':', linewidth=1.5, label=f'Fault @ Step {fault_step}')

        ax.set_title(scenario_name)
        ax.set_ylabel("Fraction of Total Load Served")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, max_steps_plot + 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # --- Calculate and store summary stats ---
        agent_summaries = pd.DataFrame([res["summary"] for res in results["agent"] if res["summary"]])
        base_summaries = pd.DataFrame([res["summary"] for res in results["baseline"] if res["summary"]])

        summary_stats.append({
            "Scenario": scenario_name,
            "Agent_Mean_Reward": agent_summaries["total_reward"].mean() if not agent_summaries.empty else np.nan,
            "Agent_Std_Reward": agent_summaries["total_reward"].std() if not agent_summaries.empty else np.nan,
            "Baseline_Mean_Reward": base_summaries["total_reward"].mean() if not base_summaries.empty else np.nan,
            "Baseline_Std_Reward": base_summaries["total_reward"].std() if not base_summaries.empty else np.nan,
            "Agent_Mean_Final_Served": agent_summaries["final_served_total_frac"].mean() if not agent_summaries.empty else np.nan,
            "Baseline_Mean_Final_Served": base_summaries["final_served_total_frac"].mean() if not base_summaries.empty else np.nan,
            "Agent_Mean_PF_Failures": agent_summaries["pf_failures"].mean() if not agent_summaries.empty else np.nan,
            "Baseline_Mean_PF_Failures": base_summaries["pf_failures"].mean() if not base_summaries.empty else np.nan, # Corrected typo
            "Agent_Avg_Volt_Penalty": agent_summaries["avg_volt_viol"].mean() if not agent_summaries.empty else np.nan,
            "Baseline_Avg_Volt_Penalty": base_summaries["avg_volt_viol"].mean() if not base_summaries.empty else np.nan,
        })


    axes[-1].set_xlabel("Episode Step")
    fig.suptitle(f"Agent vs Baseline Performance ({args.n_eval_episodes} runs/scenario)", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 

    # Save plot
    plot_filename = output_path / f"scenario_comparison_plot_N{args.n_eval_episodes}.png"
    try:
        plt.savefig(plot_filename, dpi=300)
        log.info(f"Comparison plot saved to: {plot_filename}")
    except Exception as e:
        log.error(f"Failed to save plot: {e}")
    # plt.show()

    # --- Print Summary ---
    summary_df = pd.DataFrame(summary_stats)
    log.info("\n--- Summary Statistics ---")
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(summary_df.to_string(index=False, float_format="%.3f"))

    # Save summary to CSV
    summary_filename = output_path / f"scenario_summary_stats_N{args.n_eval_episodes}.csv"
    try:
        summary_df.to_csv(summary_filename, index=False, float_format="%.4f")
        log.info(f"Summary statistics saved to: {summary_filename}")
    except Exception as e:
        log.error(f"Failed to save summary CSV: {e}")

    log.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()