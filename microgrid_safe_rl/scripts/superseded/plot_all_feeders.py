#!/usr/bin/env python3
"""
Render all target feeders to PNGs under artifacts/nets/.
"""
import argparse
from pathlib import Path
import subprocess
import sys

DEFAULTS = ["case33", "case145", "case300", "case1888"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_cfg", default="env.yaml")
    ap.add_argument("--envs", nargs="*", default=DEFAULTS)
    args = ap.parse_args()

    outdir = Path("artifacts/nets")
    outdir.mkdir(parents=True, exist_ok=True)

    for env_id in args.envs:
        out = outdir / f"{env_id}.png"
        print(f"[plot] {env_id} -> {out}")
        cmd = [
            sys.executable, "scripts/plot_feeder.py",
            "--env_id", env_id,
            "--env_cfg", args.env_cfg,
            "--out", str(out),
        ]
        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
