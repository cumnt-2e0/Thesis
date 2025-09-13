#!/usr/bin/env python3
"""
Manual baseline: open a given line (simulate isolation) and evaluate served fractions.
"""
import argparse
from pathlib import Path
import copy
import json

import numpy as np
import pandapower as pp

from microgrid_safe_rl.envs.factory import make_env
from microgrid_safe_rl.utils.config import load_yaml


def served_fractions(net):
    if not len(net.load):
        return dict(crit=1.0, imp=1.0, total=1.0)
    loads = net.load.copy()
    if "p_base_mw" not in loads.columns:
        loads["p_base_mw"] = loads["p_mw"].astype(float)
    if "priority" not in loads.columns:
        from microgrid_safe_rl.augmentation import assign_priorities
        assign_priorities(net)
        loads = net.load

    base = loads["p_base_mw"].astype(float)
    cur  = loads["p_mw"].astype(float)
    prio = loads.get("priority", 0)

    def frac(mask):
        denom = float(base[mask].sum())
        num   = float(cur[mask].sum())
        return (num / denom) if denom > 1e-9 else 1.0

    f2 = frac(prio == 2)  # critical
    f1 = frac(prio == 1)  # important
    ft = frac(np.ones(len(loads), dtype=bool))  # total
    return dict(crit=f2, imp=f1, total=ft)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", required=True)
    ap.add_argument("--env_cfg", default="env.yaml")
    ap.add_argument("--line_id", type=int, required=True, help="line index to open (isolate)")
    ap.add_argument("--out", default=None, help="save JSON summary")
    args = ap.parse_args()

    env_cfg = load_yaml(args.env_cfg)
    env = make_env(args.env_id, env_cfg, scenario={"enabled": False})
    net = copy.deepcopy(env.net0)

    # open target line
    if args.line_id not in net.line.index:
        raise SystemExit(f"Line {args.line_id} not found in {args.env_id}")
    net.line.at[int(args.line_id), "in_service"] = False

    # optional: if there are line switches on that line, open them too (visual realism)
    sw_mask = (net.switch.et == "l") & (net.switch.element == int(args.line_id))
    if sw_mask.any():
        net.switch.loc[sw_mask, "closed"] = False

    # run power flow
    ok = False
    try:
        pp.runpp(net, enforce_q_lims=True)
        ok = True
    except Exception:
        try:
            pp.runpp(net, algorithm="bfsw", enforce_q_lims=False, init="flat", tolerance_mva=1e-5)
            ok = True
        except Exception:
            try:
                pp.runpp(net, algorithm="bfsw", enforce_q_lims=False, init="dc", tolerance_mva=1e-5)
                ok = True
            except Exception:
                ok = False

    fracs = served_fractions(net) if ok else dict(crit=0.0, imp=0.0, total=0.0)
    res = {
        "env_id": args.env_id,
        "line_id": int(args.line_id),
        "pf_success": bool(ok),
        "served": fracs,
        "buses": int(len(net.bus)),
        "lines": int(len(net.line)),
        "switches": int(len(net.switch)),
    }
    print(json.dumps(res, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
