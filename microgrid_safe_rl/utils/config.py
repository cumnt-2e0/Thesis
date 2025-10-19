# microgrid_safe_rl/utils/config.py
"""Configuration utilities for loading YAML files."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary of configuration values
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is invalid YAML
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    return config


def save_yaml(data: Dict[str, Any], filepath: str) -> None:
    """
    Save a dictionary to a YAML file.
    
    Args:
        data: Dictionary to save
        filepath: Output path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dicts
        
    Returns:
        Merged configuration
    """
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result