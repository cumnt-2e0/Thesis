# path: microgrid_safe_rl/feeders/case145_load.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.converter.matpower.from_mpc import from_mpc as pp_from_mpc

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# ---------- minimal MATPOWER parser (bus/baseKV + branch/RATE_A,ratio) ----------
_MPC_BLOCK_RE = re.compile(r"mpc\.(?P<name>bus|branch)\s*=\s*\[\s*(?P<body>.*?)\s*\];", re.S)

def _parse_numeric_rows(block: str, expected_cols: int) -> np.ndarray:
    rows = []
    for line in block.strip().splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if ";" in s:
            s = s.split(";")[0]
        parts = re.split(r"\s+", s.strip())
        if not parts or parts == [""]:
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        if len(vals) < expected_cols:
            vals += [0.0] * (expected_cols - len(vals))
        rows.append(vals[:expected_cols])
    return np.array(rows, dtype=float) if rows else np.zeros((0, expected_cols), dtype=float)

def parse_mpc_minimal(mfile: str | Path) -> Dict[str, np.ndarray]:
    txt = Path(mfile).read_text()
    blocks = {m.group("name"): m.group("body") for m in _MPC_BLOCK_RE.finditer(txt)}
    if "bus" not in blocks or "branch" not in blocks:
        raise ValueError("Could not find mpc.bus or mpc.branch blocks in .m file")
    bus = _parse_numeric_rows(blocks["bus"], expected_cols=13)       # baseKV col 10 (idx 9)
    branch = _parse_numeric_rows(blocks["branch"], expected_cols=13) # RATE_A col 6 (idx 5), ratio col 9 (idx 8)
    return {"bus": bus, "branch": branch}

# ---------- sanitizer (strict, AC-friendly) ----------
_SAN_NUM_LINE = re.compile(
    r"^\s*[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+"
    r"[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+"
    r"[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s*;\s*$"
)

def _fmt(x: float) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    return str(int(xf)) if xf.is_integer() else f"{xf:.10g}"

def _sanitize_matpower_case(
    mpath: str | Path,
    *,
    force_ratio_to_one: bool = False,
    enforce_min_rateA_on_trafos: bool = True,
    min_rateA: float = 10000.0,
    fix_negative_impedance: bool = True,
) -> Tuple[bool, str]:
    """
    Strict sanitizer for mpc.branch:
      - status := 1
      - r,x := abs(r),abs(x), floors r>=1e-4, x>=1e-2 (avoid near-zero series Z)
      - clamp shunt b into [-5, 5]
      - for trafos (ratio != 0): enforce RATE_A >= min_rateA, optionally ratio := 1
    """
    p = Path(mpath)
    original = p.read_text()
    in_branch, changed = False, False

    def fix_row(raw_line: str) -> str:
        nonlocal changed
        line = raw_line.rstrip()
        if not _SAN_NUM_LINE.match(line):
            return raw_line
        parts = re.split(r"\s+", line.strip().rstrip(";"))
        if len(parts) < 13:
            return raw_line

        try:
            fbus = int(float(parts[0])); tbus = int(float(parts[1]))
            r = float(parts[2]); x = float(parts[3]); b = float(parts[4])
            rateA = float(parts[5]); rateB = float(parts[6]); rateC = float(parts[7])
            ratio = float(parts[8]); angle = float(parts[9])
            status = float(parts[10]); angmin = float(parts[11]); angmax = float(parts[12])
        except Exception:
            return raw_line

        # force in-service
        if status != 1.0:
            status = 1.0; changed = True

        # impedance cleanup + floors (stronger floors)
        if fix_negative_impedance:
            nr, nx = abs(r), abs(x)
            if nr != r or nx != x:
                changed = True
            r, x = nr, nx
        if r < 1e-4:  # was 1e-5
            r = 1e-4; changed = True
        if x < 1e-2:  # was 1e-3
            x = 1e-2; changed = True

        # clamp shunt
        if not math.isfinite(b) or abs(b) > 5.0:
            b = max(-5.0, min(5.0, b if math.isfinite(b) else 0.0)); changed = True

        # transformer handling (ratio != 0)
        if ratio != 0.0:
            if enforce_min_rateA_on_trafos and rateA == 0.0:
                rateA = float(min_rateA); changed = True
            if force_ratio_to_one and ratio != 1.0:
                ratio = 1.0; changed = True

        fixed = (
            f"\t{fbus}\t{tbus}\t{_fmt(r)}\t{_fmt(x)}\t{_fmt(b)}\t{_fmt(rateA)}\t"
            f"{_fmt(rateB)}\t{_fmt(rateC)}\t{_fmt(ratio)}\t{_fmt(angle)}\t"
            f"{_fmt(status)}\t{_fmt(angmin)}\t{_fmt(angmax)};\n"
        )
        if fixed != raw_line:
            changed = True
        return fixed

    out = []
    for raw in original.splitlines(keepends=True):
        s = raw.strip()
        if s.startswith("mpc.branch = ["):
            in_branch = True; out.append(raw); continue
        if in_branch and s.startswith("];"):
            in_branch = False; out.append(raw); continue
        if in_branch and s.endswith(";"):
            out.append(fix_row(raw)); continue
        out.append(raw)

    new_text = "".join(out)
    if changed:
        p.write_text(new_text)
        return True, "mpc.branch sanitized (status=1, |r|>=1e-4, |x|>=1e-2, shunt b clamped, trafo RATE_A patched)"
    return False, "no changes applied to mpc.branch"

def _fix_generator_limits(net: pp.pandapowerNet, q_abs_max: float = 1e6) -> None:
    """Make generator limits solver-friendly on first AC run."""
    if not len(getattr(net, "gen", [])) or not len(net.gen):
        return
    g = net.gen

    # Ensure min_p <= p_mw <= max_p, widen caps if missing
    if "min_p_mw" not in g.columns:
        g["min_p_mw"] = -1e6
    if "max_p_mw" not in g.columns:
        g["max_p_mw"] =  1e6
    g["min_p_mw"] = np.minimum(g["min_p_mw"].astype(float), g["p_mw"].astype(float))
    g["max_p_mw"] = np.maximum(g["max_p_mw"].astype(float), g["p_mw"].astype(float))

    # Ensure massive Q range for the *initial* solve
    if "min_q_mvar" not in g.columns:
        g["min_q_mvar"] = -q_abs_max
    else:
        g["min_q_mvar"] = np.minimum(g["min_q_mvar"].astype(float), -q_abs_max)
    if "max_q_mvar" not in g.columns:
        g["max_q_mvar"] = +q_abs_max
    else:
        g["max_q_mvar"] = np.maximum(g["max_q_mvar"].astype(float), +q_abs_max)


def _ensure_one_slack(net: pp.pandapowerNet) -> None:
    """
    Ensure the network has at least one ext_grid (slack).
    If missing, promote the largest-P generator's bus to a synthetic slack.
    """
    if len(getattr(net, "ext_grid", [])) and len(net.ext_grid):
        return
    if len(getattr(net, "gen", [])) and len(net.gen):
        g = net.gen.copy()
        if "p_mw" in g:
            gid = int(g["p_mw"].fillna(0.0).astype(float).idxmax())
        else:
            gid = int(g.index[0])
        bus = int(net.gen.at[gid, "bus"])
        vm = 1.0
        if "vm_pu" in net.gen and np.isfinite(float(net.gen.at[gid, "vm_pu"])):  # best-effort inherit
            vm = float(net.gen.at[gid, "vm_pu"])
        pp.create_ext_grid(net, bus=bus, vm_pu=vm, va_degree=0.0, name="synthetic_slack")
    else:
        # fallback: first bus
        pp.create_ext_grid(net, bus=int(net.bus.index.min()), vm_pu=1.0, va_degree=0.0, name="synthetic_slack")


def _inflate_gen_q_limits(net: pp.pandapowerNet, q_abs_max: float = 1e6) -> Tuple[pd.Series, pd.Series]:
    """
    Temporarily expand generator Q-limits massively to avoid infeasibility due to tight Q caps.
    Returns (original_min_q, original_max_q) so we can restore.
    """
    if not len(getattr(net, "gen", [])) or not len(net.gen):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    qmin0 = net.gen["min_q_mvar"].copy() if "min_q_mvar" in net.gen else pd.Series(np.full(len(net.gen), -1e9))
    qmax0 = net.gen["max_q_mvar"].copy() if "max_q_mvar" in net.gen else pd.Series(np.full(len(net.gen), +1e9))
    net.gen["min_q_mvar"] = -abs(q_abs_max)
    net.gen["max_q_mvar"] = +abs(q_abs_max)
    return qmin0, qmax0

# ---------- RATE_A (MVA) -> line.max_i_ka (kA) ----------
def _compute_line_max_i_ka_from_m(mfile: str | Path) -> np.ndarray:
    mpc = parse_mpc_minimal(mfile); branch, bus = mpc["branch"], mpc["bus"]
    if branch.size == 0 or bus.size == 0:
        return np.zeros((0,), dtype=float)
    fbus = branch[:, 0].astype(int) - 1
    tbus = branch[:, 1].astype(int) - 1
    rate_a = branch[:, 5]
    ratio  = branch[:, 8]
    is_line = (ratio == 0.0)
    base_kv = bus[:, 9]
    vkv = base_kv[fbus[is_line]]
    bad = ~np.isfinite(vkv) | (vkv <= 0)
    if np.any(bad):
        vkv[bad] = base_kv[tbus[is_line]][bad]
    ika = np.full(np.sum(is_line), np.nan, dtype=float)
    ok = np.isfinite(rate_a[is_line]) & (rate_a[is_line] > 0) & np.isfinite(vkv) & (vkv > 0)
    ika[ok] = rate_a[is_line][ok] / (math.sqrt(3.0) * vkv[ok])
    return ika

def _assign_max_i_ka(net: pp.pandapowerNet, ika_for_lines_ordered: np.ndarray, default_if_all_nan: Optional[float] = 1.0):
    if len(net.line) != len(ika_for_lines_ordered):
        raise RuntimeError(
            f"Line count mismatch: net.line={len(net.line)} vs parsed-lines={len(ika_for_lines_ordered)}. "
            "Ensure group_parallel_lines=False."
        )
    net.line["max_i_ka"] = ika_for_lines_ordered
    if np.all(~np.isfinite(ika_for_lines_ordered)) and default_if_all_nan is not None:
        net.line["max_i_ka"] = float(default_if_all_nan)

# ---------- AC-only power flow ladder (strict) ----------
def _cap_vm_setpoints(net: pp.pandapowerNet, lo: float = 0.98, hi: float = 1.02):
    if len(net.ext_grid) and "vm_pu" in net.ext_grid:
        net.ext_grid["vm_pu"] = net.ext_grid["vm_pu"].fillna(1.0).clip(lower=lo, upper=hi)
    if len(net.gen) and "vm_pu" in net.gen:
        net.gen["vm_pu"] = net.gen["vm_pu"].fillna(1.0).clip(lower=lo, upper=hi)

def _relax_q_limits(net: pp.pandapowerNet, expand: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    if not len(net.gen): return pd.Series(dtype=float), pd.Series(dtype=float)
    qmin0 = net.gen["min_q_mvar"].copy() if "min_q_mvar" in net.gen else pd.Series(np.full(len(net.gen), -1e9))
    qmax0 = net.gen["max_q_mvar"].copy() if "max_q_mvar" in net.gen else pd.Series(np.full(len(net.gen), +1e9))
    if "min_q_mvar" in net.gen: net.gen["min_q_mvar"] = qmin0 * expand
    if "max_q_mvar" in net.gen: net.gen["max_q_mvar"] = qmax0 * expand
    return qmin0, qmax0

def _restore_q_limits(net: pp.pandapowerNet, qmin: pd.Series, qmax: pd.Series):
    if len(qmin): net.gen["min_q_mvar"] = qmin
    if len(qmax): net.gen["max_q_mvar"] = qmax

def _run_ac_pf_strict(net: pp.pandapowerNet, *, init: str = "results", enforce_q: bool = True, tol: float = 1e-7, it: int = 80, algo: str = "nr") -> None:
    pp.runpp(net, algorithm=algo, init=init, enforce_q_lims=enforce_q,
             calculate_voltage_angles=True, tolerance_mva=tol, max_iteration=it, numba=False)

def _ac_powerflow_ladder(net: pp.pandapowerNet) -> bool:
    """
    Try increasingly permissive AC solves; return True if solved, else False.
    Uses distributed_slack in later stages to help balance active power.
    """
    # Stage 0: baseline prep
    for tab in ("bus", "line", "trafo", "gen", "load", "sgen", "ext_grid"):
        df = getattr(net, tab, None)
        if df is not None and len(df) and "in_service" in df:
            df["in_service"] = True
    if "switch" in net and len(net.switch):
        net.switch["closed"] = True
    _ensure_one_slack(net)
    _cap_vm_setpoints(net, 0.98, 1.02)

    # Stage 1: NR (flat -> results), Q-lims off then on
    try:
        pp.runpp(net, algorithm="nr", init="flat",    enforce_q_lims=False,
                 calculate_voltage_angles=True, tolerance_mva=1e-6, max_iteration=60, numba=False)
        pp.runpp(net, algorithm="nr", init="results", enforce_q_lims=True,
                 calculate_voltage_angles=True, tolerance_mva=1e-7, max_iteration=80, numba=False)
        return True
    except Exception:
        pass

    # Stage 2: DC warm start -> NR
    try:
        pp.rundcpp(net)
        if hasattr(net, "res_bus") and len(net.res_bus):
            net.res_bus["vm_pu"] = 1.0
        pp.runpp(net, algorithm="nr", init="results", enforce_q_lims=False,
                 calculate_voltage_angles=True, tolerance_mva=1e-6, max_iteration=60, numba=False)
        pp.runpp(net, algorithm="nr", init="results", enforce_q_lims=True,
                 calculate_voltage_angles=True, tolerance_mva=1e-7, max_iteration=80, numba=False)
        return True
    except Exception:
        pass

    # Stage 3: BFSW / FDXB variants
    try:
        pp.runpp(net, algorithm="bfsw", init="flat", enforce_q_lims=False,
                 calculate_voltage_angles=True, tolerance_mva=1e-5, max_iteration=100, numba=False)
        pp.runpp(net, algorithm="nr",   init="results", enforce_q_lims=True,
                 calculate_voltage_angles=True, tolerance_mva=1e-6, max_iteration=100, numba=False)
        return True
    except Exception:
        pass
    try:
        pp.runpp(net, algorithm="fdxb", init="results", enforce_q_lims=False,
                 calculate_voltage_angles=True, tolerance_mva=5e-6, max_iteration=100, numba=False)
        pp.runpp(net, algorithm="fdxb", init="results", enforce_q_lims=True,
                 calculate_voltage_angles=True, tolerance_mva=1e-6, max_iteration=120, numba=False)
        return True
    except Exception:
        pass

    # Stage 4: greatly relax Q limits + distributed slack
    qmin0, qmax0 = _inflate_gen_q_limits(net, q_abs_max=1e6)
    try:
        pp.runpp(net, algorithm="nr", init="flat",    enforce_q_lims=False,
                 calculate_voltage_angles=True, tolerance_mva=5e-6, max_iteration=120, numba=False,
                 distributed_slack=True)
        pp.runpp(net, algorithm="nr", init="results", enforce_q_lims=False,
                 calculate_voltage_angles=True, tolerance_mva=1e-6, max_iteration=150, numba=False,
                 distributed_slack=True)
        # Tighten once stabilized
        pp.runpp(net, algorithm="nr", init="results", enforce_q_lims=True,
                 calculate_voltage_angles=True, tolerance_mva=1e-6, max_iteration=120, numba=False,
                 distributed_slack=True)
        return True
    except Exception:
        _restore_q_limits(net, qmin0, qmax0)

    # Stage 5 (last-ditch): distributed slack, Q-limits off, BFSW (very robust)
    try:
        pp.runpp(net, algorithm="bfsw", init="flat", enforce_q_lims=False,
                 calculate_voltage_angles=True, tolerance_mva=1e-5, max_iteration=200, numba=False,
                 distributed_slack=True)
        return True
    except Exception:
        return False


# ---------- diagnostics ----------
def _brief_diag(net: pp.pandapowerNet) -> str:
    try:
        p_load = float(net.load["p_mw"].sum()) if len(net.load) and "p_mw" in net.load else 0.0
        p_gen = 0.0
        for tb in ("ext_grid", "gen", "sgen", "storage"):
            df = getattr(net, tb, None)
            if df is not None and len(df) and "p_mw" in df:
                p_gen += float(df["p_mw"].sum())
        lines = f"buses={len(net.bus)} lines={len(net.line)} trafos={len(net.trafo)} loads={len(net.load)} gens={len(net.gen)}"
        return f"{lines}; Pload={p_load:.3f} MW, Pgen≈{p_gen:.3f} MW"
    except Exception:
        return "diag-unavailable"

# ---------- builder (strict: must return solved net) ----------
def load_case145_pp(
    path_to_case_m: str | Path,
    casename: str = "case145",
    hz: float = 60.0,
    *,
    strict_ac: bool = True,
):
    """
    Build case145 and RETURN A SOLVED NET.
    If AC doesn't converge after the ladder, raise (strict_ac=True by default).
    No global stress is applied here; only data sanitization and AC feasibility aids.
    """
    path_to_case_m = str(path_to_case_m)
    LOG.info("Loading %s from %s (f=%.1f Hz)", casename, path_to_case_m, hz)

    # 1) Sanitize raw MATPOWER branch data (status, impedance floors, etc.)
    changed, msg = _sanitize_matpower_case(
        path_to_case_m,
        force_ratio_to_one=False,
        enforce_min_rateA_on_trafos=True,
        min_rateA=10000.0,
        fix_negative_impedance=True,
    )
    LOG.info("Sanitize: %s (%s)", "changed" if changed else "unchanged", msg)

    # 2) Convert to pandapower net (no grouping to preserve branch ordering)
    net = pp_from_mpc(
        path_to_case_m,
        casename_mpc=casename,
        f_hz=hz,
        validate_conversion=False,
        group_parallel_lines=False,
        group_parallel_transformers=False,
    )

    _fix_generator_limits(net) 

    # 3) Default/guard columns
    if "max_loading_percent" in net.line.columns:
        net.line["max_loading_percent"] = net.line["max_loading_percent"].fillna(200.0)
    else:
        net.line["max_loading_percent"] = 200.0

    # 4) Convert MATPOWER RATE_A (MVA) -> pandapower max_i_ka for lines
    ika = _compute_line_max_i_ka_from_m(path_to_case_m)
    _assign_max_i_ka(net, ika, default_if_all_nan=1.0)

    # 5) Ensure a slack exists and cap setpoints to a narrow band for the first AC
    _ensure_one_slack(net)
    _cap_vm_setpoints(net, 0.98, 1.02)

    # 6) Solve AC — must succeed
    LOG.info("Attempting AC power flow ladder ...")
    ac_ok = _ac_powerflow_ladder(net)
    if not ac_ok:
        diag = _brief_diag(net)
        msg = (
            "AC PF did not converge in loader — refusing to return an unsolved net. "
            "Please fix feeder data/sanitizer. "
            f"Quick diag: {diag}"
        )
        LOG.error(msg)
        if strict_ac:
            raise RuntimeError(msg)

    # 7) Sanity: check finite %loading
    if len(getattr(net, "res_line", [])) and "loading_percent" in net.res_line:
        if not np.isfinite(net.res_line["loading_percent"].to_numpy()).all():
            LOG.warning(
                "AC PF finished but res_line.loading_percent has non-finite values; "
                "values will be recomputed on next AC solve."
            )

    # 8) Log a quick top-load snapshot
    if ac_ok and len(net.res_line):
        top = net.res_line.loading_percent.sort_values(ascending=False).head(10)
        LOG.info("Top line loads (AC): %s", "; ".join(f"{i}:{v:.1f}%%" for i, v in top.items()))

    LOG.info(
        "pandapower net: %d buses, %d lines, %d trafos, %d loads, %d gens",
        len(net.bus), len(net.line), len(net.trafo), len(net.load), len(net.gen)
    )
    return net


if __name__ == "__main__":
    import sys
    default_path = Path(__file__).with_name("case145.m")
    mm = sys.argv[1] if len(sys.argv) > 1 else str(default_path)
    net = load_case145_pp(mm, strict_ac=True)
    print("OK:", len(net.bus), "buses,", len(net.line), "lines,", len(net.trafo), "trafos")
