#!/usr/bin/env python3
"""Quick test to verify case33 environment is working for training."""

import numpy as np
import yaml
import pandapower.networks as pn
from microgrid_safe_rl.augmentation.case33 import augment_case33
from microgrid_safe_rl.envs.microgrid_control_env import MicrogridControlEnv

def test_basic_functionality():
    """Test basic environment creation and stepping."""
    print("=" * 80)
    print("QUICK TEST: case33 Environment")
    print("=" * 80)
    
    # Load config
    config_path = "microgrid_safe_rl/configs/test_env_case33.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Create network with switches
    print("\n1. Creating network...")
    base = pn.case33bw()
    net, meta = augment_case33(
        base,
        keep_slack=False,
        add_switches=True,  # CRITICAL: Enable switches
        run_pf_after=True,
    )
    
    print(f"   ✓ Network created")
    print(f"     - Buses: {meta['buses']}")
    print(f"     - Lines: {meta['num_lines']}")
    print(f"     - Switches: {meta['num_switches']}")  # Should be > 0 now
    print(f"     - Loads: {meta['loads']}")
    print(f"     - Total load: {meta['total_load_mw']:.2f} MW")
    print(f"     - Has slack: {meta['has_slack']}")
    
    if meta['num_switches'] == 0:
        print("   ✗ ERROR: No switches added! Check augment_case33()")
        return False
    
    # Create environment
    print("\n2. Creating environment...")
    env = MicrogridControlEnv(net, cfg)
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
            print(f"     Invalid values at: {np.where(~np.isfinite(obs))}")
            return False
            
    except Exception as e:
        print(f"   ✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test stepping with different actions
    print("\n4. Testing actions...")
    actions_to_test = [
        (env.action_space.n // 2, "no-op"),
        (0, "open switch 0"),
        (env._switch_count + 1, "close switch 0"),
    ]
    
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
    
    # Test episode with cascade
    print("\n5. Testing episode with cascade...")
    obs, info = env.reset(seed=43)
    
    total_reward = 0
    cascades_seen = 0
    
    for step in range(20):
        # Random valid action
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        
        total_reward += reward
        
        # Check for events
        events = info.get("events_applied", [])
        if events:
            cascades_seen += sum(1 for e in events if "cascade" in str(e).lower())
            print(f"     Step {step}: {', '.join(map(str, events))}")
        
        if not np.all(np.isfinite(obs)):
            print(f"   ✗ ERROR: Invalid observation at step {step}")
            return False
        
        if done or trunc:
            print(f"   Episode ended at step {step}")
            break
    
    print(f"\n   ✓ Episode complete")
    print(f"     - Steps: {step + 1}")
    print(f"     - Total reward: {total_reward:.2f}")
    print(f"     - Cascade trips seen: {cascades_seen}")
    print(f"     - Final served: {info.get('served_total_frac', 0):.3f}")
    
    # Test load shedding actions
    print("\n6. Testing load shedding...")
    if env.enable_setpoint_shedding and len(env.load_ids) > 0:
        obs, info = env.reset(seed=44)
        
        # Try shedding a load
        shed_action = 2 * env._switch_count + 1  # First shed action
        obs, reward, done, trunc, info = env.step(shed_action)
        
        print(f"   ✓ Load shedding action executed")
        print(f"     - Observation valid: {np.all(np.isfinite(obs))}")
        print(f"     - PF success: {info.get('powerflow_success', False)}")
    else:
        print("   ⚠ Load shedding disabled or no loads")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    return True


def test_training_compatibility():
    """Test that environment works with SB3."""
    print("\n" + "=" * 80)
    print("TESTING STABLE-BASELINES3 COMPATIBILITY")
    print("=" * 80)
    
    try:
        from stable_baselines3.common.env_checker import check_env
        
        # Load config
        config_path = "microgrid_safe_rl/configs/test_env_case33.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # Create network
        base = pn.case33bw()
        net, _ = augment_case33(base, keep_slack=False, add_switches=True)
        
        # Create environment
        env = MicrogridControlEnv(net, cfg)
        
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
    print("\n" + "=" * 80)
    print("TESTING OBSERVATION STABILITY")
    print("=" * 80)
    
    config_path = "microgrid_safe_rl/configs/test_env_case33.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    base = pn.case33bw()
    net, _ = augment_case33(base, keep_slack=False, add_switches=True)
    env = MicrogridControlEnv(net, cfg)
    
    # Run multiple episodes
    for ep in range(5):
        obs, info = env.reset(seed=100 + ep)
        
        if not np.all(np.isfinite(obs)):
            print(f"✗ Episode {ep}: Invalid observation at reset")
            print(f"  NaN at indices: {np.where(np.isnan(obs))}")
            print(f"  Inf at indices: {np.where(np.isinf(obs))}")
            return False
        
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            
            if not np.all(np.isfinite(obs)):
                print(f"✗ Episode {ep}, Step {step}: Invalid observation")
                print(f"  Action: {action}")
                print(f"  PF success: {info.get('powerflow_success', False)}")
                print(f"  Events: {info.get('events_applied', [])}")
                return False
            
            if not np.isfinite(reward):
                print(f"✗ Episode {ep}, Step {step}: Invalid reward: {reward}")
                return False
            
            if done or trunc:
                break
        
        print(f"✓ Episode {ep}: All observations finite ({step + 1} steps)")
    
    print("\n✓ Observation stability test passed")
    return True


if __name__ == "__main__":
    import sys
    
    # Run tests
    success = True
    
    success = success and test_basic_functionality()
    
    if success:
        success = success and test_observation_stability()
    
    if success:
        success = success and test_training_compatibility()
    
    if success:
        print("\n" + "🎉" * 40)
        print("ALL TESTS PASSED - Environment ready for training!")
        print("🎉" * 40)
        sys.exit(0)
    else:
        print("\n" + "❌" * 40)
        print("TESTS FAILED - Fix issues before training")
        print("❌" * 40)
        sys.exit(1)