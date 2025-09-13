# microgrid_safe_rl/envs/factory.py
from microgrid_safe_rl.feeders.registry import get_feeder
from microgrid_safe_rl.augmentation.common import augment_full
from .microgrid_control_env import MicrogridControlEnv

def make_env(env_id: str, config: dict, scenario: dict):
    build = get_feeder(env_id)
    net = build()
    net = augment_full(net, feeder=env_id)  # feeder-aware augmentation
    return MicrogridControlEnv(net, config, scenario)
