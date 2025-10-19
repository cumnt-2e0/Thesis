# microgrid_safe_rl/augmentation/case33.py
# Augment IEEE 33-bus (case33bw) for microgrid PF + diagnostics.
# IMPORTANT: No cascade logic here. We only ensure PF health and useful line ratings.

from __future__ import annotations
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

LOG = logging.getLogger("augment.case33")
LOG.setLevel(logging.INFO)


def augment_case33(
    net: Optional[pp.pandapowerNet] = None,
    *,
    keep_slack: bool = False,
    force_radial: bool = True,
    target_pv_frac: float = 0.22,
    target_bess_p_frac: float = 0.10,
    target_bess_e_hours: float = 1.0,
    vm_pu_slack: float = 1.00,
    line_rating_mva: float = 5.0,
    run_pf_after: bool = True,
) -> Tuple[pp.pandapowerNet, Dict[str, Any]]:
    """
    Prepare IEEE 33-bus (case33bw) for PF & diagnostics.
    """
    if net is None:
        net = pn.case33bw()
        LOG.info("Loaded pn.case33bw()")

    # Ensure gen table has necessary columns
    if hasattr(net, "gen") and len(net.gen) and "slack" not in net.gen.columns:
        net.gen["slack"] = False
    
    _normalise_load_units_if_needed(net)
    _prepare_switches_case33(net, force_radial=force_radial)
    _force_lines_in_service(net)
    _fix_degenerate_line_params(net)
    _ensure_line_ratings(net, default_mva=line_rating_mva)

    if keep_slack:
        _ensure_slack(net, vm_pu=vm_pu_slack)
    else:
        _remove_slack_if_present(net)
        _ensure_internal_slack(net, vm_pu_slack=vm_pu_slack)

    _add_ders(
        net,
        pv_frac=target_pv_frac,
        bess_p_frac=target_bess_p_frac,
        bess_e_hours=target_bess_e_hours,
    )

    # Verify we have a slack after all operations
    has_slack_gen = False
    if hasattr(net, "gen") and len(net.gen) and "slack" in net.gen.columns:
        has_slack_gen = net.gen["slack"].any()
    has_ext_grid = hasattr(net, "ext_grid") and len(net.ext_grid) > 0
    
    if not has_slack_gen and not has_ext_grid:
        LOG.error("CRITICAL: No slack source after augmentation!")
        _ensure_internal_slack(net, vm_pu_slack=vm_pu_slack)
    
    meta = {
        "buses": int(len(net.bus)) if hasattr(net, "bus") else 0,
        "loads": int(len(net.load)) if hasattr(net, "load") else 0,
        "total_load_mw": float(net.load.p_mw.sum()) if len(net.load) else 0.0,
        "num_lines": int(len(net.line)) if hasattr(net, "line") else 0,
        "num_switches": int(len(net.switch)) if hasattr(net, "switch") else 0,
        "num_pv": int(len(net.sgen)) if hasattr(net, "sgen") else 0,
        "num_storage": int(len(net.storage)) if hasattr(net, "storage") else 0,
        "islanded": not keep_slack,
        "has_slack": has_slack_gen or has_ext_grid,  # Add this for verification
    }

    if run_pf_after:
        ok = _sanity_pf(net)
        if not ok:
            LOG.error("Power-flow sanity check failed after augmentation.")
        else:
            _log_top_loaded_lines(net, topk=10)

    return net, meta


# ------------------------------ Internals ---------------------------------- #

def _pick_internal_slack_bus(net: pp.pandapowerNet) -> int:
    """
    Choose a reasonable PCC for the island's reference:
    1) If there's any existing gen, use its bus.
    2) Else if there's a BESS or PV sgen, use the first one's bus.
    3) Else fall back to bus 0.
    """
    try:
        if hasattr(net, "gen") and len(net.gen):
            return int(net.gen.bus.iloc[0])
    except Exception:
        pass
    try:
        if hasattr(net, "sgen") and len(net.sgen):
            return int(net.sgen.bus.iloc[0])
    except Exception:
        pass
    return 0


def _ensure_internal_slack(net: pp.pandapowerNet, vm_pu_slack: float = 1.0) -> None:
    """
    Ensure exactly one *internal* slack (a gen with slack=True) for islanded PF.
    """
    # Remove any ext_grid first
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        net.ext_grid.drop(net.ext_grid.index, inplace=True)
    
    # Check if we already have a slack gen
    if hasattr(net, "gen") and len(net.gen) and "slack" in net.gen.columns:
        slack_mask = net.gen["slack"].astype(bool)
        if slack_mask.any():
            # Keep only the first slack and ensure it's in service
            first_idx = net.gen.index[slack_mask][0]
            net.gen["slack"] = False
            net.gen.at[first_idx, "slack"] = True
            net.gen.at[first_idx, "in_service"] = True  # CRITICAL: ensure it's on!
            if "vm_pu" in net.gen.columns:
                net.gen.at[first_idx, "vm_pu"] = float(vm_pu_slack)
            LOG.info(f"Using existing slack gen at index {first_idx}, set in_service=True")
            return
    
    # Create a new slack gen
    bus_idx = _pick_internal_slack_bus(net)
    
    # Size it to handle the total load + losses
    total_load = float(net.load["p_mw"].sum()) if len(net.load) else 10.0
    gen_size = total_load * 1.2  # 20% margin for losses
    
    pp.create_gen(
        net,
        bus=bus_idx,
        p_mw=0.0,  # Slack will balance power
        vm_pu=float(vm_pu_slack),
        slack=True,
        min_p_mw=-gen_size,
        max_p_mw=gen_size,
        controllable=True,
        name="internal_slack_gen",
        in_service=True  # CRITICAL: must be True!
    )
    
    LOG.info(f"Created internal slack gen at bus {bus_idx} with capacity ±{gen_size:.1f} MW, in_service=True")
    
def _normalise_load_units_if_needed(net: pp.pandapowerNet) -> None:
    if len(net.load) == 0:
        return
    mean_load = float(net.load.p_mw.mean())
    # Heuristic: if mean per-load > 5 MW on a 33-bus feeder, it's likely kW entered as MW.
    if mean_load > 5.0:
        LOG.warning(f"Loads look like kW (avg per-load {mean_load:.1f}); converting /1000 to MW")
        net.load["p_mw"] = net.load["p_mw"] / 1000.0
        if "q_mvar" in net.load.columns:
            net.load["q_mvar"] = net.load["q_mvar"] / 1000.0
    LOG.info(f"Total load: {float(net.load.p_mw.sum()):.2f} MW across {len(net.load)} points")


def _prepare_switches_case33(net: pp.pandapowerNet, *, force_radial: bool) -> None:
    """
    Close line-end switches and open bus-bus ties if present. case33bw typically
    has no switches, so this is usually a no-op.
    """
    if not hasattr(net, "switch") or len(net.switch) == 0:
        LOG.info("No switches in case33bw (expected); proceeding without changes.")
        return

    sw = net.switch
    if "et" not in sw.columns:
        LOG.warning("Switch table missing 'et'; not forcing tie behaviour.")
        # If we can't distinguish, best-effort: close switches pointing to line elements
        line_ids = set(net.line.index.tolist()) if hasattr(net, "line") else set()
        mask_line_guess = sw["element"].isin(line_ids)
        n = int((~sw.loc[mask_line_guess, "closed"]).sum())
        if n:
            LOG.info(f"Closing {n} inferred line-end switches (heuristic).")
            sw.loc[mask_line_guess, "closed"] = True
        return

    line_sw_mask = (sw["et"] == "l")
    busbus_mask = (sw["et"] == "b")

    n_line_to_close = int((~sw.loc[line_sw_mask, "closed"]).sum())
    if n_line_to_close > 0:
        LOG.info(f"Closing {n_line_to_close} line-end switches")
        sw.loc[line_sw_mask, "closed"] = True

    if force_radial:
        n_busbus_to_open = int(sw.loc[busbus_mask, "closed"].sum())
        if n_busbus_to_open > 0:
            LOG.info(f"Opening {n_busbus_to_open} bus-bus ties (radial base)")
            sw.loc[busbus_mask, "closed"] = False


def _force_lines_in_service(net: pp.pandapowerNet) -> None:
    if hasattr(net, "line") and "in_service" in net.line.columns:
        n_off = int((~net.line.in_service).sum())
        if n_off:
            LOG.warning(f"Forcing {n_off} lines in_service=True for baseline PF visibility")
            net.line.loc[:, "in_service"] = True


def _fix_degenerate_line_params(net: pp.pandapowerNet) -> None:
    """Ensure positive R/X/length to avoid degenerate branches."""
    if not hasattr(net, "line") or len(net.line) == 0:
        return

    r_def, x_def, l_def = 0.0922, 0.0470, 1.0  # plausible defaults for 12.66 kV feeder modeling
    zfix = False

    if "r_ohm_per_km" in net.line.columns:
        mask = (net.line.r_ohm_per_km <= 0) | net.line.r_ohm_per_km.isna()
        if mask.any():
            net.line.loc[mask, "r_ohm_per_km"] = r_def
            zfix = True

    if "x_ohm_per_km" in net.line.columns:
        mask = (net.line.x_ohm_per_km <= 0) | net.line.x_ohm_per_km.isna()
        if mask.any():
            net.line.loc[mask, "x_ohm_per_km"] = x_def
            zfix = True

    if "length_km" in net.line.columns:
        mask = (net.line.length_km <= 0) | net.line.length_km.isna()
        if mask.any():
            net.line.loc[mask, "length_km"] = l_def
            zfix = True

    if zfix:
        LOG.warning("Line impedances/length were invalid; defaulted R/X and/or length to plausible values.")


def _ensure_line_ratings(net: pp.pandapowerNet, default_mva: float) -> None:
    """
    Provide a per-line rating used for diagnostics (% utilisation).
    If plausible max_i_ka is available (<5 kA), compute MVA = √3 * U_kV * I_kA.
    Otherwise, fall back to a uniform default (e.g., 5 MVA).
    """
    if not hasattr(net, "line") or len(net.line) == 0:
        return

    vn = net.bus.loc[net.line.from_bus, "vn_kv"].to_numpy(dtype=float)
    rating = np.full(len(net.line), default_mva, dtype=float)

    if "max_i_ka" in net.line.columns:
        ik = net.line.max_i_ka.to_numpy(dtype=float)
        plausible = np.isfinite(ik) & (ik > 1e-6) & (ik < 5.0)  # ignore absurd sentinels like 99999
        rating[plausible] = (np.sqrt(3.0) * vn[plausible] * ik[plausible])

    net.line["rating_mva"] = rating


def _ensure_slack(net: pp.pandapowerNet, *, vm_pu: float = 1.00) -> None:
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        net.ext_grid.loc[:, "vm_pu"] = vm_pu
        LOG.info(f"Using existing slack @ bus {int(net.ext_grid.bus.iloc[0])}, vm_pu={vm_pu:.3f}")
        return
    pp.create_ext_grid(net, bus=0, vm_pu=vm_pu, name="UPSTREAM")
    LOG.info("Created slack @ bus 0")


def _remove_slack_if_present(net: pp.pandapowerNet) -> None:
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        LOG.info("Removing ext_grid (fully islanded)")
        net.ext_grid.drop(net.ext_grid.index, inplace=True)


def _add_ders(net: pp.pandapowerNet, *, pv_frac: float, bess_p_frac: float, bess_e_hours: float) -> None:
    total_load = float(net.load.p_mw.sum()) if len(net.load) else 0.0
    if total_load <= 0.0:
        LOG.warning("No loads found; skipping DER sizing.")
        return

    pv_total = max(0.0, pv_frac) * total_load
    bess_p = max(0.0, bess_p_frac) * total_load
    bess_e = bess_p * max(0.1, bess_e_hours)

    # PV at two mid/late-feeder buses
    pv_buses = [8, 18] if len(net.bus) > 20 else [8, 16]
    pv_each = pv_total / max(1, len(pv_buses))
    for b in pv_buses:
        if 0 <= b < len(net.bus) and pv_each > 0:
            pp.create_sgen(net, bus=b, p_mw=pv_each, q_mvar=0.0, name=f"PV@{b}")

    # BESS near later bus
    bess_bus = 25 if len(net.bus) > 25 else max(0, len(net.bus) - 2)
    if bess_p > 0:
        pp.create_storage(
            net, bus=bess_bus, p_mw=0.0,
            sn_mva=max(0.1, bess_p),
            max_e_mwh=bess_e, min_e_mwh=0.0,
            name=f"BESS@{bess_bus}"
        )

    LOG.info(f"PV capacity: {pv_total:.2f} MW over {len(pv_buses)} units")
    LOG.info(f"BESS power/energy: {bess_p:.2f} MW / {bess_e:.2f} MWh")


def _set_pf_options(net: pp.pandapowerNet) -> None:
    pp.set_user_pf_options(
        net,
        init="flat",
        enforce_q_lims=True,
        calculate_voltage_angles=True,
        only_v_results=False,     # ensure branch results are computed
        check_connectivity=True,
        trafo_model="t",
        recycle=None,
    )


def _derived_line_loading_percent(net: pp.pandapowerNet) -> np.ndarray:
    """
    Compute loading % from PF p/q and rating_mva:
    S[MVA] = sqrt(P^2 + Q^2) with P in MW, Q in MVAr.
    """
    if not hasattr(net, "res_line") or len(net.res_line) == 0:
        return np.array([])

    p = net.res_line.get("p_from_mw", pd.Series(np.zeros(len(net.res_line)))).to_numpy(dtype=float)
    q = net.res_line.get("q_from_mvar", pd.Series(np.zeros(len(net.res_line)))).to_numpy(dtype=float)
    s_mva = np.sqrt(p**2 + q**2)

    rating = net.line.get("rating_mva", pd.Series(np.full(len(net.line), 5.0))).to_numpy(dtype=float)
    rating = np.where(rating > 1e-6, rating, 5.0)
    return (s_mva / rating) * 100.0


def _sanity_pf(net: pp.pandapowerNet) -> bool:
    _set_pf_options(net)
    try:
        pp.runpp(net)
    except Exception as e:
        LOG.error(f"PF failed: {e}")
        return False

    served = float(getattr(net, "res_load", pd.DataFrame()).p_mw.sum() if hasattr(net, "res_load") else 0.0)
    if served <= 1e-3:
        LOG.error("✗ CRITICAL: res_load ≈ 0 MW — loads not powered.")
        return False

    # Derived line loading should be non-trivial now
    if hasattr(net, "res_line") and len(net.res_line):
        derived = _derived_line_loading_percent(net)
        if derived.size == 0 or float(np.nanmax(derived)) < 0.05:
            LOG.error("✗ CRITICAL: Derived line loading ~0% — check ratings or topology.")
            return False

    # Info only
    if hasattr(net, "res_bus") and len(net.res_bus):
        vmin = float(net.res_bus.vm_pu.min())
        vmax = float(net.res_bus.vm_pu.max())
        LOG.info(f"Voltage window: {vmin:.3f}–{vmax:.3f} pu")

    return True


def _log_top_loaded_lines(net: pp.pandapowerNet, *, topk: int = 10) -> None:
    if not hasattr(net, "res_line") or len(net.res_line) == 0:
        return
    derived = _derived_line_loading_percent(net)
    if derived.size == 0:
        return
    order = np.argsort(derived)[::-1][:topk]
    msg = ", ".join([f"{int(i)}:{derived[i]:.1f}%" for i in order])
    LOG.info(f"Top loaded lines (derived): {msg}")
