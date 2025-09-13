import os, yaml
from importlib.resources import files

def load_yaml(path: str):
    # If a filesystem path exists, load it; otherwise try package resources under microgrid_safe_rl/configs
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    try:
        res = files("microgrid_safe_rl.configs").joinpath(path)
        return yaml.safe_load(res.read_text(encoding="utf-8"))
    except Exception as e:
        raise FileNotFoundError(f"Could not find config '{path}' on disk or in package resources: {e}")
