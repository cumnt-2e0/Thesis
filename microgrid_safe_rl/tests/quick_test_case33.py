#!/usr/bin/env python3
"""Quick test to verify case33 environment is working for training."""

import numpy as np
import yaml
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from microgrid_safe_rl.augmentation.case33 import augment_case33
from microgrid_safe_rl.envs.microgrid_control_env import MicrogridControlEnv
import logging # Added for logging control

# --- Helper Functions for Detailed Logging ---

def log_gen_status(env: MicrogridControlEnv, prefix: str = "  "):
    """Logs the status and power output of generators."""
    print(f"{prefix}Generator Status:")
    if not hasattr(env.net, "gen") or len(env.net.gen) == 0:
        print(f"{prefix}  - No 'gen' table found.")
        return

    # Include sgen if present
    gen_df = env.net.gen.copy()
    if hasattr(env.net, "sgen") and len(env.net.sgen) > 0:
        sgen_df = env.net.sgen.copy()
        # Make columns consistent for concatenation
        if 'slack' not in sgen_df.columns: sgen_df['slack'] = False
        if 'in_service' not in sgen_df.columns: sgen_df['in_service'] = True
        # Use negative index for sgen to avoid collision if needed, or adjust based on actual structure
        sgen_df.index = [f"s_{i}" for i in sgen_df.index]
        gen_df = pd.concat([gen_df, sgen_df[['bus', 'p_mw', 'in_service', 'slack']]], ignore_index=False)


    if len(gen_df) == 0:
        print(f"{prefix}  - No generators or sgens found.")
        return

    # Check if results are available
    has_res = hasattr(env.net, "res_gen") and len(env.net.res_gen) > 0
    if has_res:
         # Merge results cautiously, handling potential index mismatches
        res_gen_df = env.net.res_gen.copy()
        if hasattr(env.net, "res_sgen") and len(env.net.res_sgen) > 0:
            res_sgen_df = env.net.res_sgen.copy()
            res_sgen_df.index = [f"s_{i}" for i in res_sgen_df.index]
            res_gen_df = pd.concat([res_gen_df, res_sgen_df[['p_mw']]], ignore_index=False)

        # Ensure index alignment before accessing .loc
        common_idx = gen_df.index.intersection(res_gen_df.index)
        gen_df['p_mw_actual'] = res_gen_df.loc[common_idx, 'p_mw']


    for idx, row in gen_df.iterrows():
        status = "ON" if row.get('in_service', True) else "OFF"
        is_slack = " (SLACK)" if row.get('slack', False) else ""
        p_set = row.get('p_mw', float('nan'))
        p_actual_str = f", Actual: {row.get('p_mw_actual', 'N/A'):.3f} MW" if has_res and idx in common_idx else ""
        print(f"{prefix}  - Gen {idx:<5}: Bus {int(row.bus):<3} Status: {status:<3}{is_slack:<8} | Setpoint: {p_set:.3f} MW{p_actual_str}")

    p_total_set = gen_df.loc[gen_df['in_service'].fillna(True), 'p_mw'].sum()
    print(f"{prefix}  - Total Online Setpoint: {p_total_set:.3f} MW")
    if has_res:
         p_total_actual = res_gen_df.loc[gen_df.loc[common_idx, 'in_service'].fillna(True), 'p_mw'].sum()
         print(f"{prefix}  - Total Actual Output:   {p_total_actual:.3f} MW")


def log_line_status(env: MicrogridControlEnv, top_k: int = 15, prefix: str = "  "):
    """Logs line status and loading percentages."""
    print(f"{prefix}Line Status & Loading (Top {top_k}):")
    if not hasattr(env.net, "line") or len(env.net.line) == 0:
        print(f"{prefix}  - No 'line' table found.")
        return

    line_df = env.net.line.copy()
    num_lines = len(line_df)
    in_service_count = int(line_df['in_service'].sum())

    print(f"{prefix}  - Lines In Service: {in_service_count} / {num_lines}")

    if not hasattr(env.net, "res_line") or len(env.net.res_line) == 0:
        print(f"{prefix}  - Power flow results (res_line) not available.")
        # Still show which lines are meant to be off
        off_lines = line_df.index[~line_df['in_service']].tolist()
        if off_lines:
             print(f"{prefix}  - Lines OUT of Service: {off_lines}")
        return

    # Use max_i_ka for rating calculation if rating_mva isn't present
    if 'rating_mva' not in line_df.columns and 'max_i_ka' in line_df.columns and 'vn_kv' in env.net.bus.columns:
         vn_kv = env.net.bus.loc[line_df['from_bus'], 'vn_kv'].values
         line_df['rating_mva'] = np.sqrt(3) * vn_kv * line_df['max_i_ka']
         line_df['rating_mva'] = line_df['rating_mva'].fillna(1.0).replace(0, 1.0) # Add fallback
         print(f"{prefix}  - NOTE: Calculated 'rating_mva' from max_i_ka for loading %")


    # Calculate loading percentage robustly
    res_line_df = env.net.res_line.copy()
    # Align indices
    res_line_df = res_line_df.reindex(line_df.index)

    p_mw = res_line_df['p_from_mw'].fillna(0.0)
    q_mvar = res_line_df['q_from_mvar'].fillna(0.0)
    s_mva = np.sqrt(p_mw**2 + q_mvar**2)

    # Use calculated or existing rating_mva, ensuring it's non-zero
    rating_mva = line_df.get('rating_mva', pd.Series(np.ones(num_lines), index=line_df.index)).fillna(1.0).replace(0, 1.0) # Default to 1 MVA if missing/zero

    loading_pct = (s_mva / rating_mva) * 100.0
    loading_pct = loading_pct.fillna(0.0) # Handle any remaining NaNs


    line_df['loading_pct'] = loading_pct

    # Sort by loading and select top K
    top_lines = line_df.sort_values(by='loading_pct', ascending=False).head(top_k)

    for idx, row in top_lines.iterrows():
        status = " ON" if row['in_service'] else "OFF"
        loading = row['loading_pct']
        print(f"{prefix}  - Line {idx:<3}: Bus {int(row.from_bus):<3} -> {int(row.to_bus):<3} | Status: {status} | Loading: {loading:6.2f}%")

    # Separately list lines that are OFF
    off_lines = line_df.index[~line_df['in_service']].tolist()
    if off_lines:
         print(f"{prefix}  - Lines OUT of Service: {off_lines}")

# --- End Helper Functions ---


def test_basic_functionality():
    """Test basic environment creation and stepping."""
    # This test remains largely the same, using the simpler test config
    print("=" * 80)
    print("QUICK TEST: case33 Environment (Basic)")
    print("=" * 80)

    # Load config
    config_path = "microgrid_safe_rl/configs/env_case33.yaml" # Use the simple test config here
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Create network with switches
    print("\n1. Creating network...")
    base = pn.case33bw()
    net, meta = augment_case33(
        base,
        keep_slack=False, # Test islanded mode
        add_switches=True,
        add_tie_switches=True, # Ensure ties are added for consistency
        run_pf_after=False, # PF check done in env reset
    )

    print(f"   ✓ Network created")
    print(f"     - Buses: {meta['buses']}")
    print(f"     - Lines: {meta['num_lines']} (incl. {meta.get('num_ties', 0)} ties)")
    print(f"     - Switches: {meta['num_switches']} (incl. {meta.get('num_ties', 0)} tie switches)")
    print(f"     - Loads: {meta['loads']}")
    print(f"     - Total load: {meta['total_load_mw']:.2f} MW")
    print(f"     - Has slack: {meta['has_slack']}")

    if meta['num_switches'] == 0:
        print("   ✗ ERROR: No switches added! Check augment_case33()")
        return False

    # Create environment
    print("\n2. Creating environment...")
    env = MicrogridControlEnv(net, cfg, scenario={}) # Empty scenario for basic test
    print(f"   ✓ Environment created")
    print(f"     - Action space: {env.action_space}")
    print(f"     - Observation space: {env.observation_space.shape}")
    print(f"     - Switch count: {env._switch_count}")

    if env._switch_count == 0:
        print("   ✗ ERROR: Environment has 0 switches!")
        return False

    # Test reset
    print("\n3. Testing reset...")
    try:
        obs, info = env.reset(seed=42)
        print(f"   ✓ Reset successful")
        print(f"     - Observation shape: {obs.shape}")
        print(f"     - Observation valid: {np.all(np.isfinite(obs))}")
        print(f"     - PF success: {info.get('powerflow_success', False)}")
        print(f"     - Served total: {info.get('served_total_frac', 0):.3f}")

        if not np.all(np.isfinite(obs)):
            print(f"   ✗ ERROR: Observation contains NaN/Inf!")
            return False

    except Exception as e:
        print(f"   ✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test stepping with different actions
    print("\n4. Testing actions...")
    # Map action indices to switch IDs for clarity
    switch_ids = env.all_switch_ids
    no_op_action = env._switch_count # Action S is no-op
    actions_to_test = [
        (no_op_action, "no-op"),
    ]
    # Add open/close for the first *controllable* switch if available
    if env._switch_count > 0:
        first_sw_idx = 0
        actions_to_test.append( (first_sw_idx, f"open sw {switch_ids[first_sw_idx]}") )
         # Action S+1+idx corresponds to closing switch idx
        actions_to_test.append( (no_op_action + 1 + first_sw_idx, f"close sw {switch_ids[first_sw_idx]}") )


    for action, desc in actions_to_test:
        try:
            obs, reward, done, trunc, info = env.step(action)
            print(f"   ✓ Action {action} ({desc})")
            print(f"     - Reward: {reward:.2f}")
            print(f"     - Observation valid: {np.all(np.isfinite(obs))}")
            print(f"     - PF success: {info.get('powerflow_success', False)}")

            if not np.all(np.isfinite(obs)):
                print(f"   ✗ ERROR: Observation invalid after action!")
                return False

        except Exception as e:
            print(f"   ✗ Action {action} failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ Basic functionality tests passed.")
    return True


# --- NEW TEST FUNCTION ---
def test_fault_and_der_scenario():
    """Test environment with line faults and DER unavailability enabled."""
    print("\n" + "=" * 80)
    print("DETAILED TEST: Faults & DER Unavailability")
    print("=" * 80)

    # Load the specific config for this test
    config_path = "microgrid_safe_rl/configs/env_case33.yaml"
    print(f"\n1. Loading Fault/DER Config: {config_path}")
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        print("   ✓ Config loaded successfully.")
        # Print key settings
        print(f"     - Cascade Enabled (for seeding): {cfg.get('cascade', {}).get('enabled', False)}")
        print(f"     - Num Seed Faults: {cfg.get('cascade', {}).get('seed', {}).get('n_seeds', 0)}")
        print(f"     - Cascade Lambda0 (should be 0): {cfg.get('cascade', {}).get('lambda0', -1.0)}")
        print(f"     - DER Unavailability Enabled: {cfg.get('der_unavailability', {}).get('enabled', False)}")
        print(f"     - DER Deficit Range: {cfg.get('der_unavailability', {}).get('p_deficit_frac', [])}")
    except Exception as e:
        print(f"   ✗ Failed to load config: {e}")
        return False

    # Create network (same as basic test)
    print("\n2. Creating network (with ties)...")
    base = pn.case33bw()
    net, meta = augment_case33(
        base,
        keep_slack=False, # Use internal slack
        add_switches=True,
        add_tie_switches=True,
        run_pf_after=False,
    )
    print(f"   ✓ Network created")

    # Create environment with the fault config
    print("\n3. Creating environment with Fault/DER config...")
    # Use an empty scenario dict initially, faults are handled by config's cascade section
    env = MicrogridControlEnv(net, cfg, scenario={})
    print(f"   ✓ Environment created")

    # Test reset and log initial state
    print("\n4. Testing reset and logging initial state...")
    try:
        obs, info = env.reset(seed=42)
        print(f"\n   ✓ Reset successful")
        print(f"     - Initial PF success: {info.get('powerflow_success', False)}")
        print(f"     - Initial Served total: {info.get('served_total_frac', 0):.3f}")

        # Log detailed initial state
        log_gen_status(env, prefix="     Initial ")
        log_line_status(env, prefix="     Initial ")

        if not np.all(np.isfinite(obs)):
            print(f"   ✗ ERROR: Observation contains NaN/Inf at reset!")
            return False

    except Exception as e:
        print(f"   ✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Simulate an episode, logging state upon fault events
    print("\n5. Simulating episode ({} steps)...".format(cfg.get("max_steps", 20)))
    total_reward = 0
    fault_events_detected = 0
    max_steps = cfg.get("max_steps", 20) # Use max_steps from config or default

    for step in range(max_steps):
        action = env.action_space.sample() # Use random actions for testing
        try:
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            pf_success = info.get('powerflow_success', False)

            print(f"\n   --- Step {step+1}/{max_steps} ---")
            print(f"     Action: {action}, Reward: {reward:.3f}, PF Success: {pf_success}")

            # Check for fault events
            events = info.get("events_applied", [])
            fault_occurred_this_step = False
            if events:
                print(f"     Events this step: {', '.join(map(str, events))}")
                for event_str in events:
                    # Check if it's a line trip event (either seed or cascade, though cascades are disabled)
                    if isinstance(event_str, str) and event_str.startswith("line:") and "trip" in event_str:
                         fault_events_detected += 1
                         fault_occurred_this_step = True
                         print(f"     FAULT DETECTED: {event_str}")

            # Log detailed state *if* a fault occurred OR if PF failed
            if fault_occurred_this_step or not pf_success:
                print(f"     --- State after event/failure ---")
                log_gen_status(env, prefix="       ")
                log_line_status(env, prefix="       ")

            if not np.all(np.isfinite(obs)):
                print(f"   ✗ ERROR: Invalid observation at step {step+1}")
                log_gen_status(env, prefix="       Failed ")
                log_line_status(env, prefix="       Failed ")
                return False

            if not np.isfinite(reward):
                print(f"   ✗ ERROR: Invalid reward at step {step+1}: {reward}")
                return False

            if done or trunc:
                print(f"\n   ✓ Episode ended at step {step + 1} (Done={done}, Truncated={trunc})")
                break

        except Exception as e:
            print(f"   ✗ Step {step + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n   --- End of Episode ---")
    print(f"     - Total Steps: {step + 1}")
    print(f"     - Total Reward: {total_reward:.3f}")
    print(f"     - Fault Events Detected: {fault_events_detected}")
    print(f"     - Final Served Total: {info.get('served_total_frac', 0):.3f}")
    print(f"     - Final PF Success: {info.get('powerflow_success', False)}")
    # Log final state
    log_gen_status(env, prefix="     Final ")
    log_line_status(env, prefix="     Final ")

    if fault_events_detected == 0 and cfg.get('cascade', {}).get('seed', {}).get('n_seeds', 0) > 0:
        print("\n   ⚠ WARNING: Expected fault events based on config, but none were detected in the info dict.")
        print("     Check event logging logic in MicrogridControlEnv.step() if faults should have occurred.")

    print("\n✓ Fault & DER scenario test completed.")
    return True


# --- Keep original compatibility tests ---
def test_training_compatibility():
    """Test that environment works with SB3."""
    # This test can use the simpler config
    print("\n" + "=" * 80)
    print("TESTING STABLE-BASELINES3 COMPATIBILITY")
    print("=" * 80)

    try:
        from stable_baselines3.common.env_checker import check_env

        # Load config
        config_path = "microgrid_safe_rl/configs/env_case33.yaml" # Simple config is fine
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Create network
        base = pn.case33bw()
        net, _ = augment_case33(base, keep_slack=False, add_switches=True, add_tie_switches=True)

        # Create environment
        env = MicrogridControlEnv(net, cfg, scenario={})

        print("\nRunning SB3 environment checker...")
        check_env(env, warn=True)
        print("✓ Environment passes SB3 checks")

        return True

    except Exception as e:
        print(f"✗ SB3 compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_stability():
    """Test that observations remain finite throughout episode."""
    # This test can use the simpler config
    print("\n" + "=" * 80)
    print("TESTING OBSERVATION STABILITY")
    print("=" * 80)

    config_path = "microgrid_safe_rl/configs/env_case33.yaml" # Simple config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base = pn.case33bw()
    net, _ = augment_case33(base, keep_slack=False, add_switches=True, add_tie_switches=True)
    env = MicrogridControlEnv(net, cfg, scenario={})

    # Run multiple episodes
    num_episodes = 3 # Reduced for speed
    max_steps_per_ep = 30
    print(f"\nRunning {num_episodes} episodes (max {max_steps_per_ep} steps each)...")
    for ep in range(num_episodes):
        obs, info = env.reset(seed=100 + ep)

        if not np.all(np.isfinite(obs)):
            print(f"✗ Episode {ep}: Invalid observation at reset")
            return False

        for step in range(max_steps_per_ep):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)

            if not np.all(np.isfinite(obs)):
                print(f"✗ Episode {ep}, Step {step}: Invalid observation")
                return False

            if not np.isfinite(reward):
                print(f"✗ Episode {ep}, Step {step}: Invalid reward: {reward}")
                return False

            if done or trunc:
                break # Episode ended early

        print(f"✓ Episode {ep}: All observations finite ({step + 1} steps)")

    print("\n✓ Observation stability test passed")
    return True


if __name__ == "__main__":
    import sys
    # Configure logging level for the test script itself if needed
    logging.basicConfig(level=logging.INFO) # Show INFO logs from the env during testing

    # Run tests
    results = {}
    print("Starting Test Suite...\n")

    results["basic"] = test_basic_functionality()
    results["fault_der"] = test_fault_and_der_scenario() # Run the new detailed test
    results["stability"] = test_observation_stability()
    results["sb3_compat"] = test_training_compatibility()

    print("\n" + "="*40 + " TEST SUMMARY " + "="*40)
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"Test: {name:<25} | Status: {status}")
        if not passed:
            all_passed = False
    print("="*94)


    if all_passed:
        print("\n" + "🎉" * 40)
        print("ALL TESTS PASSED - Environment ready for training!")
        print("🎉" * 40)
        sys.exit(0)
    else:
        print("\n" + "❌" * 40)
        print("ONE OR MORE TESTS FAILED - Fix issues before training")
        print("❌" * 40)
        sys.exit(1)