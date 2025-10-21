# microgrid_safe_rl/envs/factory.py
import copy
import logging

from microgrid_safe_rl.feeders.registry import get_feeder
from microgrid_safe_rl.augmentation.case145 import augment_case145
from microgrid_safe_rl.augmentation.case33 import augment_case33
from microgrid_safe_rl.augmentation.case118 import augment_case118
from .microgrid_control_env import MicrogridControlEnv

log = logging.getLogger("EnvFactory")
log.setLevel(logging.INFO)


def make_env(env_id: str, config: dict, scenario: dict):
    """Factory for creating microgrid environments."""
    log.info("EnvFactory.make_env called with env_id=%r", env_id)
    
    # Get base network
    build = get_feeder(env_id)
    net = build()
    
    # Augment based on network type
    env_id_lower = str(env_id).lower()
    
    if env_id_lower.startswith("case33"):
        log.info("AUG: calling augment_case33")
        net, info = augment_case33(net)
        log.info(f"AUG.OK: case33 augmented - {info['buses']} buses, {info['loads']} loads")
        
    elif env_id_lower.startswith("case145"):
        log.info("AUG: calling augment_case145")
        net, info = augment_case145(net)
        log.info("AUG.OK: case145 augmented")

    elif env_id_lower.startswith("case118"):
        log.info("AUG: calling augment_case118")
        net, info = augment_case118(net)
        log.info(f"AUG.OK: case118 augmented - {info['n_buses']} buses, {info['n_loads']} loads "
                 f"({info['n_critical_loads']} critical)")
        
    else:
        log.info(f"No augmentation for env_id={env_id}")
        info = {}
    
    # Create environment
    env = MicrogridControlEnv(net=net, config=config, scenario=scenario)
    
    # Force the env's reset template to the augmented copy
    env.net0 = copy.deepcopy(net)
    env.net = copy.deepcopy(env.net0)
    env.prev = copy.deepcopy(env.net0)
    
    return env