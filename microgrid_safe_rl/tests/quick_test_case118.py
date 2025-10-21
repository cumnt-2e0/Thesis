#!/usr/bin/env python3
"""
Test Case118 using ACTUAL MicrogridControlEnv with correct config structure.
Includes observation stability, SB3 compatibility, and detailed logging for diagnostics.
"""

import sys
import logging
from pathlib import Path
import yaml
import numpy as np

# Make sure the main package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from microgrid_safe_rl.augmentation.case118 import augment_case118
from microgrid_safe_rl.envs.microgrid_control_env import MicrogridControlEnv
from stable_baselines3.common.env_checker import check_env

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
LOG = logging.getLogger("CASE118_ENV_TEST")


def create_test_config(enable_der_deficit=True, enable_cascade_faults=False):
    """Load the training config and selectively enable/disable disturbances for testing."""
    config_path = "microgrid_safe_rl/configs/env_case118.yaml" # Use the actual training config
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

            # --- Apply overrides for specific test scenarios ---
            cfg['der_unavailability']['enabled'] = enable_der_deficit
            cfg['cascade']['enabled'] = enable_cascade_faults
            cfg['stress']['local_on_fault']['enabled'] = enable_cascade_faults # Only stress if cascades are on

            # Shorter episodes for quicker tests
            cfg['max_steps'] = 60

            # Use lower fault patience for tests if needed
            # cfg['pf_failure_patience'] = 2

            # Ensure shedding is enabled for deficit tests
            if enable_der_deficit:
                cfg['enable_setpoint_shedding'] = True

            LOG.info(f"Loaded config: DER Deficit={'ON' if enable_der_deficit else 'OFF'}, "
                     f"Cascades={'ON' if enable_cascade_faults else 'OFF'}")
            return cfg
    except FileNotFoundError:
        LOG.error(f"Config file not found at {config_path}. Cannot run test.")
        sys.exit(1)


def run_der_deficit_test():
    """Test 1: Run episode focusing on DER deficit and load shedding."""
    LOG.info("=" * 80)
    LOG.info("TEST 1: DER Deficit & Load Shedding Diagnostic")
    LOG.info("=" * 80)

    # 1. Augment network
    LOG.info("\n1. Augmenting network...")
    net, info = augment_case118()
    LOG.info(f"   ✓ {info['n_buses']} buses, {info['n_loads']} loads ({info['n_critical_loads']} critical)")

    # 2. Create environment with ONLY DER deficit enabled
    LOG.info("\n2. Creating environment (DER Deficit ONLY)...")
    config = create_test_config(enable_der_deficit=True, enable_cascade_faults=False)
    env = MicrogridControlEnv(net, config)
    LOG.info(f"   ✓ Environment created (Action Space: {env.action_space.n})")

    # 3. Run episode with random actions, logging details
    LOG.info("\n3. Running diagnostic episode (random actions)...")
    obs, info_dict = env.reset(seed=42)
    initial_bound_frac = env._der_bound_frac_ep # Get the bound for this episode
    LOG.info(f"   Initial State:")
    LOG.info(f"     - DER Bound Frac: {initial_bound_frac:.3f}")
    LOG.info(f"     - Served Total: {info_dict.get('served_total_frac', 0):.3f}")
    LOG.info(f"     - Served Critical: {info_dict.get('served_crit_frac', 0):.3f}")

    total_reward = 0
    shed_actions_taken = 0
    pf_failures = 0
    bound_violations_count = 0
    max_steps_run = env.max_steps

    for step in range(max_steps_run):
        action = env.action_space.sample() # Random actions for now
        # action = env.action_space.n - 1 # Force max shedding?
        # action = env._switch_count # Force no-op?
        try:
            obs, reward, done, truncated, info_dict = env.step(action)
            total_reward += reward

            # Detailed Log Per Step
            reward_comps = info_dict.get("reward_components", {})
            bound_penalty = reward_comps.get("bound_violation", 0.0)
            shed_frac_avg = np.mean(env.net.load['shed_frac']) if 'shed_frac' in env.net.load else 0.0

            log_msg = (f"   Step {step:2d}: Act={action:3d}, R={reward:6.2f}, "
                       f"PF={'OK' if info_dict['powerflow_success'] else 'FAIL'}, "
                       f"Served={info_dict.get('served_total_frac', 0):.3f}, "
                       f"Bound P={bound_penalty:5.2f}, Shed%={shed_frac_avg:.1%}")
            LOG.info(log_msg)

            # Check if it was a shedding action
            if action > 2 * env._switch_count:
                shed_actions_taken += 1

            if not info_dict['powerflow_success']:
                pf_failures += 1

            if bound_penalty < -0.1: # Count steps with significant bound violation
                 bound_violations_count +=1

            if done or truncated:
                LOG.info(f"\n   Episode ended at step {step}")
                break

        except Exception as e:
            LOG.error(f"   ✗ ERROR during step {step}: {e}")
            import traceback
            traceback.print_exc()
            return False # Indicate failure

    # 4. Summary
    LOG.info("\n" + "=" * 80)
    LOG.info("RESULTS - DER Deficit Test")
    LOG.info("=" * 80)
    LOG.info(f"Total Reward: {total_reward:.2f}")
    LOG.info(f"Shed Actions Taken (Random): {shed_actions_taken}")
    LOG.info(f"Power Flow Failures: {pf_failures}")
    LOG.info(f"Steps with Bound Violation: {bound_violations_count} / {step+1}")
    LOG.info(f"Final Served Total: {info_dict.get('served_total_frac', 0):.3f}")
    LOG.info(f"Final Shed Fraction (Avg): {np.mean(env.net.load.get('shed_frac', [0])):.1%}")
    LOG.info("=" * 80)

    # Success criteria can be defined (e.g., no PF failures, some reduction in bound violations)
    # For now, just check if it ran without Python errors.
    if pf_failures > max_steps_run // 2:
         LOG.warning("⚠ High number of PF failures during episode.")
         # return False # Optionally fail the test

    LOG.info("\n✓ DER Deficit test completed.")
    return True


def test_observation_stability():
    """Test 2: Check observations remain finite over multiple episodes."""
    LOG.info("\n" + "=" * 80)
    LOG.info("TEST 2: Observation Stability")
    LOG.info("=" * 80)

    net, _ = augment_case118()
    # Use a config that might stress the system (e.g., cascades enabled)
    config = create_test_config(enable_der_deficit=True, enable_cascade_faults=True)
    env = MicrogridControlEnv(net, config)

    # Run multiple episodes
    for ep in range(5):
        try:
            obs, info = env.reset(seed=100 + ep)
            if not np.all(np.isfinite(obs)):
                LOG.error(f"✗ Episode {ep}: Invalid observation at reset")
                return False

            max_steps_run = env.max_steps
            for step in range(max_steps_run):
                action = env.action_space.sample()
                obs, reward, done, trunc, info = env.step(action)

                if not np.all(np.isfinite(obs)):
                    LOG.error(f"✗ Episode {ep}, Step {step}: Invalid observation")
                    LOG.error(f"  Action: {action}, PF Success: {info.get('powerflow_success')}")
                    LOG.error(f"  Reward: {reward}, Components: {info.get('reward_components')}")
                    return False

                if not np.isfinite(reward):
                    LOG.error(f"✗ Episode {ep}, Step {step}: Invalid reward: {reward}")
                    LOG.error(f"  Components: {info.get('reward_components')}")
                    return False

                if done or trunc:
                    break
            LOG.info(f"✓ Episode {ep}: All observations finite ({step + 1} steps)")
        except Exception as e:
            LOG.error(f"✗ ERROR during Episode {ep}: {e}")
            import traceback
            traceback.print_exc()
            return False

    LOG.info("\n✓ Observation stability test passed")
    LOG.info("=" * 80)
    return True


def test_training_compatibility():
    """Test 3: Check compatibility with Stable Baselines3."""
    LOG.info("\n" + "=" * 80)
    LOG.info("TEST 3: Stable Baselines3 Compatibility")
    LOG.info("=" * 80)
    try:
        net, _ = augment_case118()
        # Use the standard training config for this check
        config = create_test_config(enable_der_deficit=True, enable_cascade_faults=False)
        env = MicrogridControlEnv(net, config)
        LOG.info("\nRunning SB3 environment checker...")
        check_env(env, warn=True)
        LOG.info("✓ Environment passes SB3 checks")
        LOG.info("=" * 80)
        return True
    except Exception as e:
        LOG.error(f"✗ SB3 compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        LOG.info("=" * 80)
        return False


if __name__ == "__main__":
    test1_success = run_der_deficit_test()
    test2_success = test_observation_stability()
    test3_success = test_training_compatibility()

    overall_success = test1_success and test2_success and test3_success

    if overall_success:
        print("\n" + "🎉" * 40)
        print("ALL TESTS PASSED - Case118 Environment appears stable!")
        print("🎉" * 40)
        sys.exit(0)
    else:
        print("\n" + "❌" * 40)
        print("TESTS FAILED - Review errors and warnings above")
        print("❌" * 40)
        sys.exit(1)