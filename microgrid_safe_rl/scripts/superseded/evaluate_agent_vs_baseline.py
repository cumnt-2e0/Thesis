#!/usr/bin/env python3
"""
Comparative evaluation: RL Agent vs Rule-Based Baseline

Tests both controllers on identical fault scenarios and plots:
1. Critical load preservation over time
2. Total load served comparison
3. Voltage profiles
4. Switching actions timeline
5. Statistical summary across multiple scenarios

Output: Publication-quality figures + CSV results for thesis
"""

import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
import pandapower.networks as pn

sys.path.insert(0, str(Path(__file__).parent.parent))

from microgrid_safe_rl.augmentation.case33 import augment_case33
from microgrid_safe_rl.envs.microgrid_control_env import MicrogridControlEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
LOG = logging.getLogger("EVALUATE")

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# ============================================================================
# TEST SCENARIOS
# ============================================================================

TEST_SCENARIOS = [
    {
        "name": "scenario1_line10_deficit25",
        "description": "Line 10 fault, 25% DER deficit",
        "fault_line": 10,
        "fault_step": 25,
        "der_deficit": 0.25,
    },
    {
        "name": "scenario2_line18_deficit30",
        "description": "Line 18 fault, 30% DER deficit",
        "fault_line": 18,
        "fault_step": 25,
        "der_deficit": 0.30,
    },
    {
        "name": "scenario3_line25_deficit20",
        "description": "Line 25 fault, 20% DER deficit",
        "fault_line": 25,
        "fault_step": 30,
        "der_deficit": 0.20,
    },
    {
        "name": "scenario4_line5_deficit35",
        "description": "Line 5 fault, 35% DER deficit (hard)",
        "fault_line": 5,
        "fault_step": 20,
        "der_deficit": 0.35,
    },
]


# ============================================================================
# BASELINE CONTROLLER
# ============================================================================

class RuleBasedController:
    """
    Simple rule-based baseline for comparison.
    
    Strategy:
    1. When fault detected: open nearest switch to isolate
    2. When DER deficit detected: shed tier 3, then tier 2, preserve tier 1
    3. No sophisticated optimization
    """
    
    def __init__(self, env):
        self.env = env
        self.fault_detected = False
        self.shed_tier3_done = False
        self.shed_tier2_done = False
    
    def reset(self):
        self.fault_detected = False
        self.shed_tier3_done = False
        self.shed_tier2_done = False
    
    def select_action(self, obs, info):
        """Select action based on simple rules."""
        # Check if fault detected (relay trip)
        n_bus = len(self.env.net.bus)
        n_line = len(self.env.net.line)
        
        # Parse observation (simplified - assumes structure)
        relay_offset = n_bus + n_line + len(self.env.all_switch_ids) + n_line
        relay_flags = obs[relay_offset:relay_offset + n_line]
        
        # If new fault detected
        if np.any(relay_flags > 0.5) and not self.fault_detected:
            self.fault_detected = True
            # Find first tripped line
            tripped_lines = np.where(relay_flags > 0.5)[0]
            if len(tripped_lines) > 0:
                # Open nearest switch (simplified: try first available switch)
                return 0  # Open first switch
        
        # Check if we need to shed load (DER bound violation)
        # Simplified: if served_total < der_bound, shed loads
        served_total = info.get("served_total_frac", 1.0)
        der_bound = getattr(self.env, "_der_bound_frac_ep", 1.0)
        
        if served_total > (der_bound + 0.05):  # 5% margin
            # Need to shed
            if not self.shed_tier3_done:
                # Shed tier 3 loads
                self.shed_tier3_done = True
                # Find first tier 3 load and shed it
                tier3_loads = self.env.net.load[self.env.net.load["priority"] == 2]
                if len(tier3_loads) > 0:
                    load_idx = tier3_loads.index[0]
                    # Action: shed load (increase action)
                    action_idx = self.env._switch_count * 2 + 1 + load_idx * 2 + 1
                    return action_idx
            
            elif not self.shed_tier2_done:
                # Shed tier 2 loads
                self.shed_tier2_done = True
                tier2_loads = self.env.net.load[self.env.net.load["priority"] == 1]
                if len(tier2_loads) > 0:
                    load_idx = tier2_loads.index[0]
                    action_idx = self.env._switch_count * 2 + 1 + load_idx * 2 + 1
                    return action_idx
        
        # Default: no-op
        return self.env._switch_count


# ============================================================================
# DATA COLLECTION
# ============================================================================

@dataclass
class EpisodeResult:
    """Store results from one episode run."""
    scenario_name: str
    controller: str  # "agent" or "baseline"
    
    # Time series
    steps: List[int]
    served_critical: List[float]
    served_important: List[float]
    served_total: List[float]
    voltages_min: List[float]
    voltages_max: List[float]
    line_loading_max: List[float]
    switches_opened: List[Tuple[int, int]]  # (step, switch_id)
    loads_shed: List[Tuple[int, int, float]]  # (step, load_id, amount)
    
    # Summary statistics
    final_critical: float
    final_important: float
    final_total: float
    total_switches: int
    total_load_shed_mw: float
    pf_failures: int
    reward_total: float


def run_episode(env, controller, scenario: dict, controller_type: str) -> EpisodeResult:
    """Run one episode with given controller and scenario."""
    
    LOG.info(f"Running {controller_type} on {scenario['name']}")
    
    # Inject scenario into environment
    env.scenario = {
        "name": scenario["name"],
        "events": [
            {
                "t": scenario["fault_step"] * env.step_seconds,
                "target": f"line:{scenario['fault_line']}",
                "op": "trip",
            }
        ],
    }
    
    # Override DER deficit
    env.der_cfg["p_deficit_frac"] = [scenario["der_deficit"], scenario["der_deficit"]]
    
    # Reset
    obs, info = env.reset(seed=42)
    
    if hasattr(controller, 'reset'):
        controller.reset()
    
    # Data collection
    steps = []
    served_crit = []
    served_imp = []
    served_tot = []
    v_min = []
    v_max = []
    line_max = []
    switches = []
    sheds = []
    
    total_reward = 0.0
    pf_fails = 0
    done = False
    step = 0
    
    while not done and step < env.max_steps:
        # Select action
        if controller_type == "agent":
            action, _ = controller.predict(obs, deterministic=True)
        else:
            action = controller.select_action(obs, info)
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        
        # Record
        steps.append(step)
        served_crit.append(info.get("served_crit_frac", 0.0))
        served_imp.append(info.get("served_imp_frac", 0.0))
        served_tot.append(info.get("served_total_frac", 0.0))
        
        if hasattr(env.net, "res_bus") and len(env.net.res_bus):
            v = env.net.res_bus.vm_pu.values
            v_min.append(float(np.nanmin(v)))
            v_max.append(float(np.nanmax(v)))
        else:
            v_min.append(1.0)
            v_max.append(1.0)
        
        if hasattr(env.net, "res_line") and len(env.net.res_line):
            lp = env.net.res_line.loading_percent.values
            line_max.append(float(np.nanmax(lp)))
        else:
            line_max.append(0.0)
        
        # Track actions
        if action < env._switch_count:
            switches.append((step, int(action)))
        elif action > env._switch_count * 2:
            # Load shed action
            sheds.append((step, 0, 0.1))  # Simplified
        
        total_reward += reward
        if not info.get("powerflow_success", True):
            pf_fails += 1
        
        step += 1
    
    # Calculate total shed
    total_shed = sum(sh[2] for sh in sheds) if sheds else 0.0
    
    return EpisodeResult(
        scenario_name=scenario["name"],
        controller=controller_type,
        steps=steps,
        served_critical=served_crit,
        served_important=served_imp,
        served_total=served_tot,
        voltages_min=v_min,
        voltages_max=v_max,
        line_loading_max=line_max,
        switches_opened=switches,
        loads_shed=sheds,
        final_critical=served_crit[-1] if served_crit else 0.0,
        final_important=served_imp[-1] if served_imp else 0.0,
        final_total=served_tot[-1] if served_tot else 0.0,
        total_switches=len(switches),
        total_load_shed_mw=total_shed,
        pf_failures=pf_fails,
        reward_total=total_reward,
    )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_comparison(agent_result: EpisodeResult, baseline_result: EpisodeResult, output_dir: Path):
    """Create comparison plots for one scenario."""
    
    scenario_name = agent_result.scenario_name
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Agent vs Baseline: {scenario_name}", fontsize=16, fontweight='bold')
    
    # Plot 1: Critical load served
    ax = axes[0, 0]
    ax.plot(agent_result.steps, np.array(agent_result.served_critical) * 100, 
            label="RL Agent", linewidth=2, color='#2ecc71')
    ax.plot(baseline_result.steps, np.array(baseline_result.served_critical) * 100,
            label="Rule-Based", linewidth=2, linestyle='--', color='#e74c3c')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Critical Load Served (%)")
    ax.set_title("Critical Load Preservation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 2: Total load served
    ax = axes[0, 1]
    ax.plot(agent_result.steps, np.array(agent_result.served_total) * 100,
            label="RL Agent", linewidth=2, color='#3498db')
    ax.plot(baseline_result.steps, np.array(baseline_result.served_total) * 100,
            label="Rule-Based", linewidth=2, linestyle='--', color='#e67e22')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Total Load Served (%)")
    ax.set_title("Total Load Served")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Voltage profile
    ax = axes[1, 0]
    ax.fill_between(agent_result.steps, agent_result.voltages_min, agent_result.voltages_max,
                     alpha=0.3, label="RL Agent", color='#9b59b6')
    ax.fill_between(baseline_result.steps, baseline_result.voltages_min, baseline_result.voltages_max,
                     alpha=0.3, label="Rule-Based", color='#95a5a6')
    ax.axhline(y=0.95, color='r', linestyle=':', label="V limits")
    ax.axhline(y=1.05, color='r', linestyle=':')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title("Voltage Profile (Min-Max Range)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Switching actions
    ax = axes[1, 1]
    agent_sw = [s[0] for s in agent_result.switches_opened]
    baseline_sw = [s[0] for s in baseline_result.switches_opened]
    ax.eventplot([agent_sw], colors=['#2ecc71'], lineoffsets=1, linelengths=0.4, label="RL Agent")
    ax.eventplot([baseline_sw], colors=['#e74c3c'], lineoffsets=0.5, linelengths=0.4, label="Rule-Based")
    ax.set_xlabel("Time Step")
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["Rule-Based", "RL Agent"])
    ax.set_title("Switching Actions Timeline")
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, max(agent_result.steps)])
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{scenario_name}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    LOG.info(f"Saved plot: {scenario_name}_comparison.png")


def plot_aggregate_results(all_results: List[EpisodeResult], output_dir: Path):
    """Plot aggregate statistics across all scenarios."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Aggregate Performance: RL Agent vs Rule-Based Baseline", 
                 fontsize=16, fontweight='bold')
    
    # Separate agent and baseline results
    agent_results = [r for r in all_results if r.controller == "agent"]
    baseline_results = [r for r in all_results if r.controller == "baseline"]
    
    scenarios = [r.scenario_name for r in agent_results]
    
    # Plot 1: Critical load preservation
    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.35
    agent_crit = [r.final_critical * 100 for r in agent_results]
    baseline_crit = [r.final_critical * 100 for r in baseline_results]
    
    ax.bar(x - width/2, agent_crit, width, label='RL Agent', color='#2ecc71')
    ax.bar(x + width/2, baseline_crit, width, label='Rule-Based', color='#e74c3c')
    ax.set_ylabel('Critical Load Served (%)')
    ax.set_title('Critical Load Preservation')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("scenario", "S").split("_")[0] for s in scenarios], rotation=45)
    ax.legend()
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='k', linestyle=':', alpha=0.5)
    
    # Plot 2: Total load served
    ax = axes[1]
    agent_total = [r.final_total * 100 for r in agent_results]
    baseline_total = [r.final_total * 100 for r in baseline_results]
    
    ax.bar(x - width/2, agent_total, width, label='RL Agent', color='#3498db')
    ax.bar(x + width/2, baseline_total, width, label='Rule-Based', color='#e67e22')
    ax.set_ylabel('Total Load Served (%)')
    ax.set_title('Total Load Served')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("scenario", "S").split("_")[0] for s in scenarios], rotation=45)
    ax.legend()
    
    # Plot 3: Switching operations
    ax = axes[2]
    agent_switches = [r.total_switches for r in agent_results]
    baseline_switches = [r.total_switches for r in baseline_results]
    
    ax.bar(x - width/2, agent_switches, width, label='RL Agent', color='#9b59b6')
    ax.bar(x + width/2, baseline_switches, width, label='Rule-Based', color='#95a5a6')
    ax.set_ylabel('Number of Switch Operations')
    ax.set_title('Switching Operations')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("scenario", "S").split("_")[0] for s in scenarios], rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    LOG.info("Saved aggregate comparison plot")


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def save_results_csv(all_results: List[EpisodeResult], output_dir: Path):
    """Save results to CSV for analysis."""
    
    # Summary table
    summary_data = []
    for result in all_results:
        summary_data.append({
            "Scenario": result.scenario_name,
            "Controller": result.controller,
            "Critical_Served_%": result.final_critical * 100,
            "Important_Served_%": result.final_important * 100,
            "Total_Served_%": result.final_total * 100,
            "Switches": result.total_switches,
            "Load_Shed_MW": result.total_load_shed_mw,
            "PF_Failures": result.pf_failures,
            "Total_Reward": result.reward_total,
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = output_dir / "evaluation_summary.csv"
    df.to_csv(csv_path, index=False)
    LOG.info(f"Saved summary CSV: {csv_path}")
    
    # Calculate improvements
    agent_results = df[df["Controller"] == "agent"]
    baseline_results = df[df["Controller"] == "baseline"]
    
    improvements = []
    for scenario in agent_results["Scenario"].unique():
        agent_row = agent_results[agent_results["Scenario"] == scenario].iloc[0]
        baseline_row = baseline_results[baseline_results["Scenario"] == scenario].iloc[0]
        
        improvements.append({
            "Scenario": scenario,
            "Critical_Improvement_%": agent_row["Critical_Served_%"] - baseline_row["Critical_Served_%"],
            "Total_Improvement_%": agent_row["Total_Served_%"] - baseline_row["Total_Served_%"],
            "Switch_Reduction": baseline_row["Switches"] - agent_row["Switches"],
            "Reward_Improvement": agent_row["Total_Reward"] - baseline_row["Total_Reward"],
        })
    
    imp_df = pd.DataFrame(improvements)
    imp_csv_path = output_dir / "improvements.csv"
    imp_df.to_csv(imp_csv_path, index=False)
    LOG.info(f"Saved improvements CSV: {imp_csv_path}")
    
    # Print summary
    LOG.info("\n" + "=" * 80)
    LOG.info("EVALUATION SUMMARY")
    LOG.info("=" * 80)
    LOG.info(f"\nAverage Critical Load Preservation:")
    LOG.info(f"  RL Agent:     {agent_results['Critical_Served_%'].mean():.2f}%")
    LOG.info(f"  Rule-Based:   {baseline_results['Critical_Served_%'].mean():.2f}%")
    LOG.info(f"  Improvement:  {imp_df['Critical_Improvement_%'].mean():.2f}%")
    
    LOG.info(f"\nAverage Total Load Served:")
    LOG.info(f"  RL Agent:     {agent_results['Total_Served_%'].mean():.2f}%")
    LOG.info(f"  Rule-Based:   {baseline_results['Total_Served_%'].mean():.2f}%")
    LOG.info(f"  Improvement:  {imp_df['Total_Improvement_%'].mean():.2f}%")
    
    LOG.info(f"\nAverage Switch Operations:")
    LOG.info(f"  RL Agent:     {agent_results['Switches'].mean():.1f}")
    LOG.info(f"  Rule-Based:   {baseline_results['Switches'].mean():.1f}")
    LOG.info(f"  Reduction:    {imp_df['Switch_Reduction'].mean():.1f}")
    
    LOG.info("\n" + "=" * 80)


def main():
    """Run full evaluation pipeline."""
    
    # Setup
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"evaluation_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOG.info("=" * 80)
    LOG.info("STARTING COMPARATIVE EVALUATION")
    LOG.info(f"Output directory: {output_dir}")
    LOG.info("=" * 80)
    
    # Load network
    LOG.info("\nLoading and augmenting Case33...")
    net = pn.case33bw()
    net_aug, info = augment_case33(net)
    LOG.info(f"Network ready: {info['n_buses']} buses, {info['n_lines']} lines")
    
    # Load config
    import yaml
    config_path = "configs/env_case33_final.yaml"
    if not Path(config_path).exists():
        LOG.error(f"Config not found: {config_path}")
        LOG.info("Using default config...")
        config = {
            "step_seconds": 5.0,
            "max_steps": 120,
            "voltage_limits": [0.95, 1.05],
            "reward_weights": {
                "tier1": 25.0, "tier2": 6.0, "tier3": 1.8,
                "volt_violation": -3.0, "thermal_violation": -2.0,
                "switch_cost": -0.02, "pf_failure": -100.0,
            },
            "safety": {
                "max_closed_ties": 1, "voltage_margin_pu": 0.01,
                "loading_margin_pct": 5.0, "per_step_switch_budget": 1,
            },
            "controls": {
                "allow_generation_control": False,
                "allow_storage_control": False,
            },
            "enable_setpoint_shedding": True,
            "shed_step": 0.10,
            "cascade": {"enabled": False},
            "stress": {"local_on_fault": {"enabled": False}},
            "der_unavailability": {
                "enabled": True,
                "p_deficit_frac": [0.18, 0.30],
                "scaling_range": [0.55, 0.88],
                "random_outage_prob": 0.12,
            },
            "observation": {
                "telemetry_latency_steps": 0,
                "add_measurement_noise": False,
            },
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create environment
    env = MicrogridControlEnv(net_aug, config)
    LOG.info("Environment created")
    
    # Load trained agent
    model_path = "logs/case33_curriculum_*/case33_curriculum_complete.zip"
    import glob
    model_files = glob.glob(model_path)
    
    if not model_files:
        LOG.error(f"No trained model found at: {model_path}")
        LOG.info("Looking for any .zip model in logs/...")
        model_files = glob.glob("logs/**/*.zip", recursive=True)
        if model_files:
            model_path = model_files[0]
            LOG.info(f"Using: {model_path}")
        else:
            LOG.error("No trained model found! Please train first.")
            return
    else:
        model_path = model_files[0]
    
    LOG.info(f"Loading trained agent: {model_path}")
    agent = PPO.load(model_path)
    
    # Create baseline controller
    baseline = RuleBasedController(env)
    
    # Run evaluations
    all_results = []
    
    for scenario in TEST_SCENARIOS:
        LOG.info("\n" + "-" * 80)
        LOG.info(f"Testing: {scenario['description']}")
        LOG.info("-" * 80)
        
        # Run with agent
        agent_result = run_episode(env, agent, scenario, "agent")
        all_results.append(agent_result)
        
        # Run with baseline
        baseline_result = run_episode(env, baseline, scenario, "baseline")
        all_results.append(baseline_result)
        
        # Plot comparison
        plot_comparison(agent_result, baseline_result, output_dir)
        
        # Quick summary
        LOG.info(f"\nResults for {scenario['name']}:")
        LOG.info(f"  Agent    - Critical: {agent_result.final_critical*100:.1f}%, "
                f"Total: {agent_result.final_total*100:.1f}%, "
                f"Switches: {agent_result.total_switches}")
        LOG.info(f"  Baseline - Critical: {baseline_result.final_critical*100:.1f}%, "
                f"Total: {baseline_result.final_total*100:.1f}%, "
                f"Switches: {baseline_result.total_switches}")
    
    # Aggregate plots
    plot_aggregate_results(all_results, output_dir)
    
    # Save CSV
    save_results_csv(all_results, output_dir)
    
    # Save raw results as JSON
    json_path = output_dir / "raw_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    LOG.info(f"Saved raw results: {json_path}")
    
    LOG.info("\n" + "=" * 80)
    LOG.info("EVALUATION COMPLETE")
    LOG.info(f"Results saved to: {output_dir}")
    LOG.info("=" * 80)


if __name__ == "__main__":
    main()