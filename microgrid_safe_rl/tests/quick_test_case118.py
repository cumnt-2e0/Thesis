#!/usr/bin/env python3
"""
Test Case118 using ACTUAL MicrogridControlEnv with correct config structure.
"""

import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from microgrid_safe_rl.augmentation.case118 import augment_case118
from microgrid_safe_rl.envs.microgrid_control_env import MicrogridControlEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
LOG = logging.getLogger("CASE118_ENV_TEST")


def create_cascade_config():
    """Create config that matches MicrogridControlEnv's expected structure."""
    
    return {
        'step_seconds': 5.0,
        'max_steps': 120,
        'voltage_limits': [0.95, 1.05],
        
        # DER UNAVAILABILITY (forces load shedding)
        'der_unavailability': {  # NOT 'der_degradation'!
            'enabled': True,
            'p_deficit_frac': [0.10, 0.20],  # 30-40% deficit
            'random_outage_prob': 0.3,
            'max_disabled': 3,
            'max_der_size_mw': 50.0,
            'scaling_range': [0.4, 0.8],
        },
        
        # CASCADE CONFIGURATION
        'cascade': {
            'enabled': True,
            
            # Seed faults
            'seed': {
                'n_seeds': 1,
                'step_window_frac': [0.10, 0.15],  # Trigger early
                'exclude_bridges': False,
            },
            
            # Gate parameters (CRITICAL for triggering)
            'overload_min_pct': 4.0,  
            'min_hold_steps': 2,  # Just 1 step hold
            'hysteresis_drop': 3.0,
            
            # Probability model
            'lambda0': 3.0,  # Higher base rate
            'rho0': 0.0,  # Lower threshold for fL
            'alpha': 1.0,  # Less aggressive exponent
            'betaV': 0.4,
            'zone_boost': 2.5,
            'd0': 4.0,
            
            # Per-wave limits
            'per_wave_max': 1,
            'max_additional': 3,
            'min_gap_steps': 2,
            
            # Topology
            'hop_limit': 2,
            'neighbor_sample_k': 50,
            'exclude_bridges': True,
            
            # Loading-based probability boost
            'prob_loading_map': [
                [4.5, 1.5],   # 80%+ → 80% of lambda
                [5.5, 2.0],  # 100%+ → 100% of lambda
                [7, 3.0],  # 115%+ → 150% of lambda
            ],
        },
        
        # STRESS CONFIGURATION (CRITICAL!)
        'stress': {
            # Local stress around faults
            'local_on_fault': {
                'enabled': True,
                'on_every_trip': False,  # Apply to seed AND cascade trips
                'hops': 2,
                'duration_steps': 10,  # Persist longer
                'accumulate': False,  # Stack stress from multiple trips
                'gate_margin_pct': 1.0,  # Target 90% (80 + 10)
            },

            'global': {                    # optional but helpful
                'load_scale_p': 1.15,
                'line_rating_scale': 0.9
            },
        },
        
        # REWARD
        'reward_weights': {
            'tier1': 10.0,
            'tier2': 3.0,
            'tier3': 1.0,
            'volt_violation': -5.0,
            'thermal_violation': -3.0,
            'switch_cost': -0.05,
            'restore_tier1_delta': 8.0,
            'pf_failure': -50.0,
            'bound_violation': -20.0,  # Penalty for violating DER bound
        },
        
        # SAFETY
        'safety': {
            'max_closed_ties': 1,
            'voltage_margin_pu': 0.02,
            'loading_margin_pct': 10.0,
            'per_step_switch_budget': 2,
            'min_dwell_steps': 2,
        },
        
        # CONTROLS
        'controls': {
            'allow_switching': True,
            'allow_tie_closure': True,
            'allow_load_shedding': False,  
            'allow_generation_control': False,
            'allow_storage_control': False,
        },
        
        # Feature flags
        'enable_setpoint_shedding': True,
        'pf_failure_patience': 5,
    }


def run_test():
    """Run comprehensive test."""
    
    LOG.info("=" * 80)
    LOG.info("CASE118 ENVIRONMENT INTEGRATION TEST")
    LOG.info("=" * 80)
    
    # 1. Augment network
    LOG.info("\n1. Augmenting network...")
    net, info = augment_case118()
    LOG.info(f"   ✓ {info['n_buses']} buses, {info['n_loads']} loads ({info['n_critical_loads']} critical)")
    
    # 2. Create environment with correct config
    LOG.info("\n2. Creating environment with cascade config...")
    config = create_cascade_config()
    
    LOG.info(f"   Config highlights:")
    LOG.info(f"     - DER deficit: {config['der_unavailability']['p_deficit_frac']}")
    LOG.info(f"     - Cascade gate: {config['cascade']['overload_min_pct']}% (lowered from 110%)")
    LOG.info(f"     - Lambda0: {config['cascade']['lambda0']}")
    LOG.info(f"     - Local stress: {config['stress']['local_on_fault']['enabled']}")
    
    env = MicrogridControlEnv(net, config)
    LOG.info(f"   ✓ Environment created")
    
    # 3. Run episode
    LOG.info("\n3. Running episode...")
    obs, info_dict = env.reset(seed=42)
    
    LOG.info(f"   Initial: crit={info_dict.get('served_crit_frac', 0):.1%}, "
            f"total={info_dict.get('served_total_frac', 0):.1%}")
    
    cascade_count = 0
    der_triggered = False
    critical_lost = False
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info_dict = env.step(action)
        
        # Check for DER unavailability
        if hasattr(env, '_der_bound_frac_ep') and env._der_bound_frac_ep < 0.99:
            if not der_triggered:
                LOG.info(f"\n   Step {step}: ✓ DER UNAVAILABILITY ACTIVE")
                LOG.info(f"     Bound fraction: {env._der_bound_frac_ep:.3f}")
                der_triggered = True
        
        # Check for cascades
        if 'events_applied' in info_dict:
            events = info_dict['events_applied']
            cascade_events = [e for e in events if 'cascade' in str(e).lower()]
            if cascade_events:
                cascade_count += len(cascade_events)
                LOG.info(f"\n   Step {step}: ✓ CASCADE! {len(cascade_events)} trips")
                for ev in cascade_events:
                    LOG.info(f"     - {ev}")
        
        # Check critical loads
        crit_served = info_dict.get('served_crit_frac', 1.0)
        if crit_served < 0.95 and not critical_lost:
            LOG.info(f"\n   Step {step}: ⚠ CRITICAL LOADS AFFECTED")
            LOG.info(f"     Critical served: {crit_served:.1%}")
            critical_lost = True
        
        if step % 20 == 0:
            LOG.info(f"   Step {step:3d}: reward={reward:7.2f}, "
                    f"crit={info_dict.get('served_crit_frac', 0):.1%}")
        
        if done or truncated:
            LOG.info(f"\n   Episode ended at step {step}")
            break
    
    # 4. Summary
    LOG.info("\n" + "=" * 80)
    LOG.info("RESULTS")
    LOG.info("=" * 80)
    LOG.info(f"DER unavailability: {'✓ YES' if der_triggered else '✗ NO'}")
    LOG.info(f"Cascade trips: {cascade_count} total")
    LOG.info(f"Critical loads affected: {'✓ YES' if critical_lost else '✗ NO'}")
    LOG.info(f"Final critical served: {info_dict.get('served_crit_frac', 0):.1%}")
    LOG.info("=" * 80)
    
    success = (cascade_count > 0 or der_triggered)
    
    if not der_triggered:
        LOG.warning("\n⚠ DER unavailability did not activate")
        LOG.warning("  Check: env._der_bound_frac_ep and _apply_der_unavailability()")
    
    if cascade_count == 0:
        LOG.warning("\n⚠ No cascade trips occurred")
        LOG.warning("  Possible causes:")
        LOG.warning("    1. Local stress not reaching gate threshold (80%)")
        LOG.warning("    2. Lambda0 too low")
        LOG.warning("    3. Hold time not being met")
    else:
        LOG.info(f"\n✓ SUCCESS: {cascade_count} cascade trips triggered!")
        LOG.info("  Environment cascade logic is working correctly")
    
    return success


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)