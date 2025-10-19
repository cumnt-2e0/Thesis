# path: microgrid_safe_rl/envs/microgrid_control_env.py
from __future__ import annotations

import copy
import json
import logging
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top 
from gymnasium import spaces

from microgrid_safe_rl.augmentation.case145 import assign_priorities


# ---------------- utils ---------------- #

def _safe_bool_series(x, *, length: Optional[int] = None, default: bool = True) -> np.ndarray:
    if x is None:
        return np.full(length or 0, default, dtype=bool)
    if isinstance(x, bool):
        return np.full(length or 0, x, dtype=bool)
    try:
        return np.asarray(x, dtype=bool)
    except Exception:
        return np.full(length or 0, default, dtype=bool)


def _clip01(x):  # noqa: E302
    return np.asarray(np.clip(x, 0.0, 1.0), dtype=np.float32)


def _lookup_prob(prob_map: List[Tuple[float, float]], loading_pct: float) -> float:
    p = 0.0
    for th, val in prob_map:
        if loading_pct >= th:
            p = float(val)
        else:
            break
    return float(np.clip(p, 0.0, 1.0))


@dataclass
class ObsCfg:
    telemetry_latency_steps: int = 0
    include_async_masks: bool = False
    add_measurement_noise: bool = False
    noise_std_voltage: float = 0.002
    noise_std_loading: float = 0.01


@dataclass
class SafetyCfg:
    max_closed_ties: int = 1
    voltage_margin_pu: float = 0.01
    loading_margin_pct: float = 5.0
    per_step_switch_budget: int = 1
    min_dwell_steps: int = 0

DEFAULT_CTRL = {
    "allow_generation_control": False,   # agent cannot change P setpoints
    "allow_storage_control": False,      # freeze storage P too (optional)
}

DEFAULT_LOCAL_STRESS = {
    "enabled": True,
    "hops": 2,                 # hop radius from first tripped line
    "duration_steps": 3,       # TTL of the stress window
    "p_scale": 1.50,           # local load spike multiplier
    "q_scale": 1.00,           # leave Q unchanged unless you want added stress
    "line_rating_scale": 0.70, # temporary derating (max_i_ka *= 0.7)
}


class MicrogridControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, net, config: Dict[str, Any], scenario: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = config or {}

        # Scenario (JSON or dict). Kept minimal; merged with seeds.
        self.scenario = scenario or {"name": "default", "events": [], "repairs": []}
        if isinstance(self.scenario, str):
            with open(self.scenario, "r") as f:
                self.scenario = json.load(f)

        # Core knobs
        self.step_seconds = float(self.config.get("step_seconds", 5.0))
        self.v_limits: Tuple[float, float] = tuple(self.config.get("voltage_limits", (0.95, 1.05)))
        self.max_steps = int(self.config.get("max_steps", 60))
        self.failure_patience = int(self.config.get("pf_failure_patience", 3))

        # Reward
        rw = self.config.get("reward_weights", {}) or {}
        self.w_tier = (float(rw.get("tier1", 8.0)),
                    float(rw.get("tier2", 3.0)),
                    float(rw.get("tier3", 1.0)))
        self.w_volt = float(rw.get("volt_violation", -2.0))
        self.w_therm = float(rw.get("thermal_violation", -1.0))
        self.w_switch = float(rw.get("switch_cost", -0.02))
        self.w_restore_shape = float(rw.get("restore_tier1_delta", 5.0))
        self.pf_failure_penalty = float(rw.get("pf_failure", -30.0))
        self._prev_f1: Optional[float] = None

        # Observation realism
        ocfg = self.config.get("observation", {}) or {}
        noise_block = ocfg.get("noise_std", {}) if isinstance(ocfg.get("noise_std", {}), dict) else {}
        self.obs_cfg = ObsCfg(
            telemetry_latency_steps=int(ocfg.get("telemetry_latency_steps", 0)),
            include_async_masks=bool(ocfg.get("include_async_masks", False)),
            add_measurement_noise=bool(ocfg.get("add_measurement_noise", False)),
            noise_std_voltage=float(noise_block.get("voltage_pu", 0.002)),
            noise_std_loading=float(noise_block.get("loading_pct", 0.01)),
        )

        # Safety
        scfg = self.config.get("safety", {}) or {}
        self.safety = SafetyCfg(
            max_closed_ties=int(scfg.get("max_closed_ties", 1)),
            voltage_margin_pu=float(scfg.get("voltage_margin_pu", 0.01)),
            loading_margin_pct=float(scfg.get("loading_margin_pct", 5.0)),
            per_step_switch_budget=int(scfg.get("per_step_switch_budget", 1)),
            min_dwell_steps=int(scfg.get("min_dwell_steps", 0)),
        )

        # -------- SINGLE CASCADE ENGINE --------
        ccfg = self.config.get("cascade", {}) or {}
        self.cascade_enabled: bool = bool(ccfg.get("enabled", True))

        # Seeds (random)
        sc = ccfg.get("seed", {}) or {}
        self.seed_n_seeds: int = int(sc.get("n_seeds", 1))
        self.seed_step_window_frac: Tuple[float, float] = tuple(sc.get("step_window_frac", [0.05, 0.35]))
        self.seed_exclude_bridges: bool = bool(sc.get("exclude_bridges", True))

        # Candidate neighborhood
        self.cascade_hop_limit: int = int(ccfg.get("hop_limit", 3))
        self.cascade_neighbor_sample_k: int = int(ccfg.get("neighbor_sample_k", 0))  # 0 = no subsample

        # Overload hold (gate)
        self.cascade_overload_min: float = float(ccfg.get("overload_min_pct", 110.0))
        self.cascade_min_hold: int = int(ccfg.get("min_hold_steps", 1))
        self.cascade_hysteresis_drop: float = float(ccfg.get("hysteresis_drop", 3.0))

        # Probability model (per-step)
        self.lambda0: float = float(ccfg.get("lambda0", 0.12))
        self.rho0: float = float(ccfg.get("rho0", 0.90))
        self.alpha: float = float(ccfg.get("alpha", 2.0))
        self.betaV: float = float(ccfg.get("betaV", 1.0))
        self.zone_boost: float = float(ccfg.get("zone_boost", 1.4))
        self.d0: float = float(ccfg.get("d0", 3.0))
        self.per_wave_max: int = int(ccfg.get("per_wave_max", 2))
        self.max_additional: int = int(ccfg.get("max_additional", 5))
        self.min_gap_steps: int = int(ccfg.get("min_gap_steps", 1))
        self.exclude_bridges: bool = bool(ccfg.get("exclude_bridges", True))

        # Optional piecewise prob by loading
        self.prob_loading_map: List[Tuple[float, float]] = [
            (float(th), float(p)) for th, p in ccfg.get("prob_loading_map", [(115.0, 1.0)])
        ]
        # ---------------------------------------

        # Features
        self.enable_setpoint_shedding = bool(self.config.get("enable_setpoint_shedding", False))
        self.shed_step = float(self.config.get("shed_step", 0.1))

        # Controls (normalized view; _restore_frozen_generation will still read config directly)
        self.controls = {**DEFAULT_CTRL, **(self.config.get("controls") or {})}

        # Stress configuration
        st = self.config.get("stress", {}) or {}
        self.stress_global = (st.get("global", {}) or {})
        self.stress_local_cfg = {**DEFAULT_LOCAL_STRESS, **(st.get("local_on_fault", {}) or {})}
        self._local_stress_active: bool = False
        self._local_stress_expiry: Optional[int] = None
        self._local_stress_lines: Set[int] = set()

        # DER unavailability config
        self.der_cfg = self.config.get("der_unavailability", {})
        self._der_bound_frac_ep = 1.0  # Will be set in reset()
        
        # Bound violation penalty (for reward)
        self.bound_pen_w = float(self.config.get("reward_weights", {}).get("bound_violation", -20.0))

        # Prepare net(s)
        self.net0 = copy.deepcopy(net)
        self._prepare_net_inplace(self.net0)

        # Verify slack is still there after preparation
        if hasattr(self.net0, "gen") and len(self.net0.gen) and "slack" in self.net0.gen.columns:
            slack_count = self.net0.gen["slack"].sum()
            if slack_count == 0:
                self.log.warning("Slack was lost during net preparation!")
                # Find the first gen and make it slack
                if len(self.net0.gen) > 0:
                    self.net0.gen.at[self.net0.gen.index[0], "slack"] = True
                    self.log.info(f"Restored slack to gen at index {self.net0.gen.index[0]}")

        self.net = copy.deepcopy(self.net0)

        # IDs
        self.all_switch_ids = list(net.switch.index) if len(net.switch) else []
        self.load_ids = list(net.load.index) if len(net.load) else []

        # Tie switches (from pristine net)
        self._is_tie: Dict[int, bool] = {}
        if len(net.switch):
            mask = (net.switch.et == "l") & (~_safe_bool_series(net.switch.get("closed", True), length=len(net.switch)))
            for idx in net.switch.index:
                self._is_tie[int(idx)] = bool(mask.loc[idx]) if idx in mask.index else False

        # Action space
        K_cfg = int(self.config.get("max_switches", 0) or 0)
        self.K = K_cfg if (K_cfg > 0 and K_cfg <= len(self.all_switch_ids)) else len(self.all_switch_ids)
        self._switch_count = self.K
        self._shed_count = 2 * len(self.load_ids) if self.enable_setpoint_shedding else 0
        self.action_space = spaces.Discrete(2 * self._switch_count + 1 + self._shed_count)

        # Observation space
        n_bus = len(net.bus)
        n_line = len(net.line)
        n_gen = len(getattr(net, "gen", []))
        n_storage = len(getattr(net, "storage", []))
        scada_dim = (
            n_bus
            + n_line
            + len(self.all_switch_ids)
            + n_line
            + n_line
            + n_bus
            + n_gen
            + n_gen
            + 2 * n_gen
            + n_storage
            + 2 * n_storage
            + n_gen
            + n_bus
            + 3
            + 2
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(scada_dim,), dtype=np.float32)

        # ---------- Runtime state (episode-agnostic defaults) ----------
        self.prev = copy.deepcopy(self.net)
        self.current_step = 0
        self.wall_time_s = 0.0
        self._obs_queue: deque = deque(maxlen=max(1, self.obs_cfg.telemetry_latency_steps + 1))
        self._relay_trip = {int(i): False for i in self.net.line.index}
        self._switch_dwell = {int(i): 0 for i in self.all_switch_ids}
        self._events_by_step: Dict[int, List[dict]] = {}
        self.pf_fail_streak = 0

        # Baseline snapshots (filled in reset(); placeholders here)
        self._genP0 = None
        self._sgenP0 = None
        self._extP0 = None
        self._storP0 = None
        self._loadP0 = None
        self._loadQ0 = None

        # Cascade state
        self._seed_steps: List[int] = []
        self._seed_lines: List[int] = []
        self._tripped: Set[int] = set()
        self._hold: Dict[int, int] = {}
        self._additional_trips = 0
        self._last_trip_step = -10**9
        self._corridor: Set[int] = set()
        self._line_dist_cache: Dict[Tuple[int, int], float] = {}
        self._bridge_lines: Set[int] = set()

        # Optional line-state arrays some downstream code may expect
        try:
            nL = len(self.net.line)
        except Exception:
            nL = 0
        # Start everything "on" and healthy
        self.line_on = np.full(nL, True, dtype=bool)
        self.line_failed = np.full(nL, False, dtype=bool)
        self.line_tripped = np.full(nL, False, dtype=bool)
        self.line_forced_off = np.full(nL, False, dtype=bool)
        self.line_blocked = np.full(nL, False, dtype=bool)
        self._line_mask_cache = np.full(nL, False, dtype=bool)
        self.corridor_mask = np.ones(nL, dtype=bool)
        self.coverage_mask = np.ones(nL, dtype=bool)
        self.bridges_excluded_mask = np.ones(nL, dtype=bool)

        # Action mapping placeholders (rebuilt in reset())
        self.action_lines = np.arange(nL, dtype=int)
        self.n_actions = int(nL)
        self.allowed_line_ids = np.arange(nL, dtype=int)
        self.line_id_map = {int(i): int(i) for i in range(nL)}

        # Ensure metadata cols exist on the base net
        self._ensure_load_metadata(self.net0)


    # ------------ prep ------------- #

    def _build_obs_vector_from_current_state(self) -> np.ndarray:
        V = self._sense_voltages()
        I = self._sense_line_loading()
        sw = self._read_switches()
        br = self._read_breakers()
        rly = self._read_relays()
        live = self._derive_bus_energized()
        p_set, q_set, h_up, h_dn, der_avail = self._sense_der_channels()
        soc, bess_chg, bess_dis = self._sense_bess_channels()
        p_load_bus = self._loads_per_bus_scaled()
        tier = self._tier_onehot()
        t_step = _clip01(np.array([self.current_step / max(1.0, self.max_steps)], dtype=np.float32))
        wall = _clip01(np.array([self.wall_time_s / max(1.0, self.max_steps * self.step_seconds)], dtype=np.float32))
        
        parts = [V, I, sw, br, rly, live, p_set, q_set, h_up, h_dn, soc, bess_chg, bess_dis, der_avail, p_load_bus, tier, t_step, wall]
        obs = np.concatenate(parts).astype(np.float32)
        
        # CRITICAL: Validate observation
        if not np.all(np.isfinite(obs)):
            self.log.error(f"OBSERVATION CONTAINS NaN/Inf at step {self.current_step}")
            for i, part in enumerate(parts):
                if not np.all(np.isfinite(part)):
                    self.log.error(f"  Part {i} (shape={part.shape}) has invalid values: {part}")
            # Replace NaN/Inf with safe values
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        
        return obs

    def _ensure_slack_for_reset(self):
        """Ensure we have a valid slack reference for power flow."""
        # Check for slack generator
        has_slack = False
        if hasattr(self.net, "gen") and len(self.net.gen) and "slack" in self.net.gen.columns:
            has_slack = self.net.gen["slack"].any()
        
        # Check for ext_grid
        has_ext_grid = hasattr(self.net, "ext_grid") and len(self.net.ext_grid) > 0
        
        if not has_slack and not has_ext_grid:
            # Create an internal slack gen
            bus_idx = 0  # Use bus 0 as default
            if hasattr(self.net, "gen") and len(self.net.gen):
                bus_idx = int(self.net.gen.bus.iloc[0])
            
            total_load = float(self.net.load["p_mw"].sum()) if len(self.net.load) else 10.0
            gen_size = total_load * 1.2
            
            pp.create_gen(
                self.net,
                bus=bus_idx,
                p_mw=0.0,
                vm_pu=1.0,
                slack=True,
                min_p_mw=-gen_size,
                max_p_mw=gen_size,
                name="emergency_slack",
                in_service=True
            )
            self.log.warning(f"Created emergency slack generator at bus {bus_idx}")

    def _ensure_load_metadata(self, net):
        if len(net.load):
            if "p_base_mw" not in net.load.columns:
                net.load["p_base_mw"] = net.load["p_mw"].astype(float)
            if "q_base_mvar" not in net.load.columns and "q_mvar" in net.load.columns:
                net.load["q_base_mvar"] = net.load["q_mvar"].astype(float)
            if "shed_frac" not in net.load.columns:
                net.load["shed_frac"] = 0.0
            if "priority" not in net.load.columns:
                assign_priorities(net)

    def _prepare_net_inplace(self, net):
        self._ensure_load_metadata(net)
        for tb in ("gen", "sgen", "storage"):
            df = getattr(net, tb, None)
            if df is None or not len(df):
                continue
            if "p_base_mw" not in df.columns and "p_mw" in df.columns:
                df["p_base_mw"] = df["p_mw"].astype(float)
            if "in_service" not in df.columns:
                df["in_service"] = True

    # -------- scenario events (append-only) -------- #

    def _build_events_index_from_scenario(self):
        events = list(self.scenario.get("events", []))
        for ev in events:
            t_sec = float(ev.get("t", 0.0))
            step = int(round(t_sec / max(1e-9, self.step_seconds)))
            self._events_by_step.setdefault(step, []).append(ev)

    def _apply_event(self, ev: dict):
        tgt = ev.get("target", "")
        op = ev.get("op", "trip")
        if tgt.startswith("line:"):
            try:
                li = int(tgt.split(":")[1])
            except Exception:
                return


            if li in self.net.line.index and op == "trip":
                # Update pandapower state
                self.net.line.at[li, "in_service"] = False
                
                # Update tracking arrays
                self._relay_trip[li] = True
                self._tripped.add(int(li))
                
                # Update boolean state arrays 
                if hasattr(self, 'line_on') and li < len(self.line_on):
                    self.line_on[li] = False
                if hasattr(self, 'line_tripped') and li < len(self.line_tripped):
                    self.line_tripped[li] = True
                
                # Grow corridor for cascade neighbors
                self._corridor = self._compute_corridor(self._tripped)
                self._last_trip_step = self.current_step
                
                # Apply local stress on EVERY trip if configured
                if self.stress_local_cfg.get("enabled", False):
                    if self.stress_local_cfg.get("on_every_trip", True) or not self._local_stress_active:
                        self._apply_local_stress_around_fault([li])
                        self.log.info(f"LOCAL_STRESS applied around line {li} at step {self.current_step}")

    # -------- cascade helpers -------- #

    def _healthy_lines(self, net) -> List[int]:
        if not len(net.line):
            return []
        return [int(i) for i in net.line.index if bool(net.line.at[i, "in_service"])]

    def _lines_adjacent_by_hops(self, net, seed_line: int, hops: int) -> List[int]:
        G = top.create_nxgraph(net, include_switches=True)
        if seed_line not in net.line.index:
            return []
        a, b = int(net.line.at[seed_line, "from_bus"]), int(net.line.at[seed_line, "to_bus"])
        reach = set()
        for s in (a, b):
            try:
                reach |= set(nx.single_source_shortest_path_length(G, s, cutoff=hops).keys())
            except Exception:
                pass
        neigh = []
        for li, row in net.line.iterrows():
            u, v = int(row.from_bus), int(row.to_bus)
            if u in reach or v in reach:
                neigh.append(int(li))
        neigh = [seed_line] + [li for li in neigh if li != seed_line]
        return neigh

    def _lines_nearest_by_electrical(self, net, seed_line: int, k: int) -> List[int]:
        if seed_line not in net.line.index:
            return [seed_line]
        cands = [li for li in net.line.index if li != seed_line and bool(net.line.at[li, "in_service"])]
        dists = []
        for lj in cands:
            d = self._electrical_distance_between_lines(seed_line, int(lj))
            dists.append((int(lj), float(d)))
        dists.sort(key=lambda t: t[1])
        take = [seed_line] + [li for (li, _) in dists[:max(0, k - 1)]]
        return take

    def _pick_random_fault_cluster(
        self, net, *, n_faults: int, hops: int, use_electrical: bool, exclude_bridges: bool
    ) -> List[int]:
        healthy = self._healthy_lines(net)
        if not healthy:
            return []
        if exclude_bridges:
            if not self._bridge_lines:
                self._bridge_lines = self._compute_bridge_lines()
            healthy = [li for li in healthy if int(li) not in self._bridge_lines] or self._healthy_lines(net)
        seed = int(np.random.choice(healthy))
        if n_faults <= 1:
            return [seed]
        if use_electrical:
            cluster = self._lines_nearest_by_electrical(net, seed, n_faults)
        else:
            neigh = self._lines_adjacent_by_hops(net, seed, max(1, hops))
            neigh = [li for li in neigh if bool(net.line.at[li, "in_service"])]
            if len(neigh) > n_faults:
                others = [li for li in neigh if li != seed]
                picks = (
                    list(np.random.choice(others, size=n_faults - 1, replace=False))
                    if len(others) >= (n_faults - 1)
                    else others
                )
                cluster = [seed] + picks
            else:
                cluster = neigh[:n_faults]
        out, seen = [], set()
        for li in cluster:
            li = int(li)
            if li in seen:
                continue
            seen.add(li)
            if li in net.line.index and bool(net.line.at[li, "in_service"]):
                out.append(li)
            if len(out) >= n_faults:
                break
        return out if out else [seed]

    def _nx_graph(self) -> nx.Graph:
        return top.create_nxgraph(self.net, include_switches=True)

    def _line_endpoints(self, li: int) -> Optional[Tuple[int, int]]:
        if li not in self.net.line.index:
            return None
        return int(self.net.line.at[li, "from_bus"]), int(self.net.line.at[li, "to_bus"])

    def _line_electrical_length(self, li: int) -> float:
        try:
            L = float(self.net.line.at[li, "length_km"])
        except Exception:
            L = 1.0
        r = float(self.net.line.get("r_ohm_per_km", 0.2).get(li, 0.2))
        x = float(self.net.line.get("x_ohm_per_km", 0.3).get(li, 0.3))
        z = max(1e-6, (r * r + x * x) ** 0.5)
        return L * z

    def _electrical_distance_between_lines(self, li: int, lj: int, G=None) -> float:
        key = (min(li, lj), max(li, lj))
        if key in self._line_dist_cache:
            return self._line_dist_cache[key]
        if G is None:
            G = self._nx_graph()
        if G is None:
            self._line_dist_cache[key] = float("inf")
            return float("inf")
        ei, ej = self._line_endpoints(li), self._line_endpoints(lj)
        if not ei or not ej:
            self._line_dist_cache[key] = float("inf")
            return float("inf")
        for u, v, k, data in G.edges(keys=True, data=True):
            if isinstance(data, dict) and data.get("et") == "l":
                lij = int(data.get("element"))
                G[u][v][k]["weight"] = max(1e-6, self._line_electrical_length(lij))
            else:
                G[u][v][k]["weight"] = 1e-3
        try:
            cand = []
            for a in ei:
                for b in ej:
                    cand.append(nx.shortest_path_length(G, a, b, weight="weight"))
            dmin = float(min(cand)) if cand else float("inf")
        except Exception:
            dmin = float("inf")
        self._line_dist_cache[key] = dmin
        return dmin

    def _compute_bridge_lines(self) -> Set[int]:
        try:
            g0 = top.create_nxgraph(self.net0, include_switches=True)
            bridges = set()
            for a, b in nx.bridges(g0):
                for li, row in self.net0.line.iterrows():
                    u, v = int(row.from_bus), int(row.to_bus)
                    if {a, b} == {u, v}:
                        bridges.add(int(li))
            return bridges
        except Exception:
            return set()

    def _compute_corridor(self, cluster: Set[int]) -> Set[int]:
        G = self._nx_graph()
        if G is None or not len(self.net.line):
            return set()
        hop_set: set[int] = set()
        for li in cluster:
            ends = self._line_endpoints(li)
            if not ends:
                continue
            nodes = set()
            for end in ends:
                nodes |= set(nx.single_source_shortest_path_length(G, end, cutoff=self.cascade_hop_limit).keys())
            for u, v, k, data in G.edges(keys=True, data=True):
                if u in nodes or v in nodes:
                    if isinstance(data, dict) and data.get("et") == "l":
                        hop_set.add(int(data.get("element")))
        dist_set: set[int] = set()
        for lj in self.net.line.index:
            if not bool(self.net.line.at[lj, "in_service"]):
                continue
            dmin = min(self._electrical_distance_between_lines(lj, li, G) for li in cluster) if cluster else float("inf")
            if dmin <= (2.5 * self.d0):
                dist_set.add(int(lj))
        return hop_set | dist_set

    def _eligible_neighbors(self) -> List[int]:
        if not self._tripped:
            return []
        if not len(self.net.line):
            return []
        cands = [int(li) for li in self._corridor if bool(self.net.line.at[li, "in_service"]) and li not in self._tripped]
        if self.exclude_bridges:
            if not self._bridge_lines:
                self._bridge_lines = self._compute_bridge_lines()
            cands = [li for li in cands if li not in self._bridge_lines]
        if self.cascade_neighbor_sample_k and len(cands) > self.cascade_neighbor_sample_k:
            cands = list(np.random.choice(cands, size=self.cascade_neighbor_sample_k, replace=False))
        return sorted(set(cands))
    
    # -------- DER nUnavailability -------- # 

    def _apply_der_unavailability(self, seed=None):
        """Reduce DER generation to force load curtailment."""
        if not self.der_cfg or not self.der_cfg.get("enabled", False):
            return
        
        rng = np.random.default_rng(seed)
        
        # Sample the bound fraction for this episode
        lo, hi = self.der_cfg.get("p_deficit_frac", [0.1, 0.3])
        lo, hi = float(lo), float(max(hi, lo))
        deficit = rng.uniform(lo, hi)
        self._der_bound_frac_ep = float(np.clip(1.0 - deficit, 0.0, 1.0))
        
        self.log.info(f"DER: episode bound fraction = {self._der_bound_frac_ep:.3f} (deficit={deficit:.1%})")
        
        # Calculate target generation
        if len(self.net.load) and "p_base_mw" in self.net.load.columns:
            P_L = float(self.net.load["p_base_mw"].sum())
        else:
            P_L = float(self.net.load["p_mw"].sum()) if len(self.net.load) else 0.0
        
        P_target = self._der_bound_frac_ep * P_L
        
        # Random outages (optional)
        outage_p = float(self.der_cfg.get("random_outage_prob", 0.0))
        max_size = float(self.der_cfg.get("max_der_size_mw", 1e9))
        max_disabled = int(self.der_cfg.get("max_disabled", 2))
        
        disabled = []
        disabled_count = 0
        
        for tb in ("gen", "sgen"):
            if disabled_count >= max_disabled:
                break
            if not hasattr(self.net, tb):
                continue
            df = getattr(self.net, tb)
            if not len(df):
                continue
        
            is_slack = df.get("slack", False) if tb == "gen" and "slack" in df.columns else pd.Series(False, index=df.index)
            
            if rng.random() < outage_p:
                small = df[(df.in_service.astype(bool)) & (df.p_mw <= max_size) & (~is_slack)]
                if len(small):
                    idx = int(rng.choice(small.index))
                    disabled.append((tb, idx, float(df.at[idx, "p_mw"])))
                    df.at[idx, "in_service"] = False
                    disabled_count += 1
                    self.log.info(f"DER_OUTAGE: {tb}[{idx}] = {df.at[idx, 'p_mw']:.2f} MW")
        
        # Scale remaining DER to target
        s_lo, s_hi = self.der_cfg.get("scaling_range", [0.4, 0.9])
        s_lo, s_hi = float(s_lo), float(max(s_hi, s_lo))
        
        # Calculate current available generation
        P_base_avail = 0.0
        for tb in ("gen", "sgen"):
            if not hasattr(self.net, tb):
                continue
            df = getattr(self.net, tb)
            m = df.in_service.astype(bool)
            if "p_base_mw" in df.columns:
                P_base_avail += float(df.loc[m, "p_base_mw"].sum())
            else:
                P_base_avail += float(df.loc[m, "p_mw"].sum())
        
        if P_base_avail > 1e-9:
            common_scale = float(np.clip(P_target / P_base_avail, s_lo, s_hi))
            
            for tb in ("gen", "sgen"):
                if not hasattr(self.net, tb):
                    continue
                df = getattr(self.net, tb)
                m = df.in_service.astype(bool)
                if not m.any():
                    continue
                
                # Add per-unit jitter
                jitter = rng.uniform(s_lo, s_hi, size=int(m.sum()))
                
                if "p_base_mw" in df.columns:
                    df.loc[m, "p_mw"] = df.loc[m, "p_base_mw"].values * (common_scale * jitter)
                else:
                    df.loc[m, "p_mw"] = df.loc[m, "p_mw"].values * (common_scale * jitter)
        
        # Log result
        P_online = 0.0
        for tb in ("gen", "sgen"):
            if not hasattr(self.net, tb):
                continue
            df = getattr(self.net, tb)
            P_online += float(df.loc[df.in_service.astype(bool), "p_mw"].sum())
        
        self.log.info(f"DER_SCALED: P_load={P_L:.1f} MW, P_target={P_target:.1f} MW, "
                    f"P_online={P_online:.1f} MW, disabled={len(disabled)}")

    # -------- localized stress overlay -------- #

    def _snapshot_baselines(self):
        """Capture initial setpoints after global stress & before scenario events."""
        try:
            if hasattr(self.net, "gen") and len(self.net.gen):
                self._genP0 = self.net.gen["p_mw"].astype(float).copy()
            if hasattr(self.net, "sgen") and len(self.net.sgen):
                self._sgenP0 = self.net.sgen["p_mw"].astype(float).copy()
            if hasattr(self.net, "ext_grid") and len(self.net.ext_grid):
                self._extP0 = self.net.ext_grid.get("max_p_mw", None)  # optional
            if hasattr(self.net, "storage") and len(self.net.storage):
                # If your model uses p_mw for charge/discharge, snapshot it
                self._storP0 = self.net.storage.get("p_mw", None)
            if hasattr(self.net, "load") and len(self.net.load):
                self._loadP0 = self.net.load["p_mw"].astype(float).copy()
                if "q_mvar" in self.net.load:
                    self._loadQ0 = self.net.load["q_mvar"].astype(float).copy()
        except Exception:
            pass

    def _enforce_frozen_generation(self):
        """If controls disallow gen/storage manipulation, pin to baselines."""
        try:
            if not self.controls.get("allow_generation_control", False):
                if hasattr(self.net, "gen") and len(self.net.gen) and self._genP0 is not None:
                    # Align indexes defensively
                    common = self.net.gen.index.intersection(self._genP0.index)
                    self.net.gen.loc[common, "p_mw"] = self._genP0.loc[common].values
            if not self.controls.get("allow_storage_control", False):
                if hasattr(self.net, "storage") and len(self.net.storage) and self._storP0 is not None:
                    common = self.net.storage.index.intersection(self._storP0.index)
                    self.net.storage.loc[common, "p_mw"] = self._storP0.loc[common].values
        except Exception:
            pass
        
    def _buses_around_lines(self, net, lines: List[int], hops: int = 1) -> Set[int]:
        if not lines or hops <= 0:
            return set()
        G = top.create_nxgraph(net, include_switches=True)
        seeds = set()
        for li in lines:
            if li not in net.line.index:
                continue
            a, b = int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])
            seeds.add(a)
            seeds.add(b)
        seen: Set[int] = set()
        for s in seeds:
            try:
                reach = nx.single_source_shortest_path_length(G, s, cutoff=hops).keys()
                for v in reach:
                    if v in net.bus.index:
                        seen.add(int(v))
            except Exception:
                pass
        return seen

    def _scale_loads_at_buses(self, net, buses: Set[int], p_scale: float = 1.0, q_scale: float = 1.0):
        if not len(net.load) or not buses:
            return
        mask = net.load.bus.astype(int).isin(buses)
        if "p_mw" in net.load.columns and float(p_scale) != 1.0:
            net.load.loc[mask, "p_mw"] = net.load.loc[mask, "p_mw"].astype(float) * float(p_scale)
            if "p_base_mw" in net.load.columns:  # keep reward normalization consistent
                net.load.loc[mask, "p_base_mw"] = net.load.loc[mask, "p_base_mw"].astype(float) * float(p_scale)
        if "q_mvar" in net.load.columns and float(q_scale) != 1.0:
            net.load.loc[mask, "q_mvar"] = net.load.loc[mask, "q_mvar"].astype(float) * float(q_scale)
            if "q_base_mvar" in net.load.columns:
                net.load.loc[mask, "q_base_mvar"] = net.load.loc[mask, "q_base_mvar"].astype(float) * float(q_scale)

    def _scale_line_ratings(self, net, lines: Set[int], scale: float):
        if not len(net.line) or not lines or "max_i_ka" not in net.line.columns:
            return
        s = float(np.clip(scale, 0.2, 1.0))
        idx = [i for i in lines if i in net.line.index]
        if idx:
            net.line.loc[idx, "max_i_ka"] = np.maximum(1e-6, net.line.loc[idx, "max_i_ka"].astype(float) * s)
            if "max_loading_percent" in net.line:
                net.line.loc[idx, "max_loading_percent"] = np.maximum(net.line.loc[idx, "max_loading_percent"], 200.0)

    def _apply_local_stress_around_fault(self, seed_lines: List[int]):
        """Apply localized stress with GRADUAL ramping to avoid PF divergence."""
        cfg = self.stress_local_cfg
        hops = int(cfg.get("hops", 3))
        duration = int(cfg.get("duration_steps", 10))
        
        # Find neighborhood
        neigh_lines: Set[int] = set()
        for li in seed_lines:
            neigh_lines |= set(self._lines_adjacent_by_hops(self.net, int(li), hops=hops))
        neigh_lines |= set(seed_lines)
        buses = self._buses_around_lines(self.net, list(neigh_lines), hops=1)
        
        self.log.info(f"LOCAL_STRESS: targeting {len(neigh_lines)} lines, {len(buses)} buses")
        
        # CRITICAL: GRADUAL stress application
        target = float(self.cascade_overload_min) + float(cfg.get("gate_margin_pct", 5.0))
        max_attempts = 10
        
        # Start with MILD stress
        p_scale = 1.05  # Just 5% increase initially
        line_scale = 0.95  # Just 5% derating
        
        for attempt in range(max_attempts):
            # Apply incremental stress
            if attempt > 0:  # Don't double-apply on first iteration
                ratchet_p = 1.10  # Increase by 10% each time
                ratchet_line = 0.95  # Decrease rating by 5% each time
                
                self._scale_loads_at_buses(self.net, buses, p_scale=ratchet_p, q_scale=1.0)
                self._scale_line_ratings(self.net, neigh_lines, scale=ratchet_line)
                
                p_scale *= ratchet_p
                line_scale *= ratchet_line
            else:
                # First attempt: apply initial mild stress
                self._scale_loads_at_buses(self.net, buses, p_scale=p_scale, q_scale=1.0)
                self._scale_line_ratings(self.net, neigh_lines, scale=line_scale)
            
            # Try to run PF
            ok = self._runpf()
            if not ok:
                self.log.warning(f"STRESS attempt {attempt+1}: PF FAILED - backing off")
                # Back off the last increment
                if attempt > 0:
                    self._scale_loads_at_buses(self.net, buses, p_scale=1.0/ratchet_p, q_scale=1.0)
                    self._scale_line_ratings(self.net, neigh_lines, scale=1.0/ratchet_line)
                    p_scale /= ratchet_p
                    line_scale /= ratchet_line
                break
            
            # Check if we've achieved gate crossing
            lp = self.net.res_line.loading_percent.reindex(self.net.line.index).fillna(0.0).astype(float)
            corridor_max = float(lp.loc[list(neigh_lines)].max()) if neigh_lines else 0.0
            
            top5 = lp.loc[list(neigh_lines)].nlargest(5)
            self.log.info(f"STRESS attempt {attempt+1}: max={corridor_max:.1f}% (p_scale={p_scale:.2f}x, "
                        f"line_scale={line_scale:.2f}x) top5={[(i,f'{v:.1f}%') for i,v in top5.items()]}")
            
            if corridor_max >= target:
                self.log.info(f"GATE_ACHIEVED: {corridor_max:.1f}% >= {target:.1f}% ✓")
                break
            
            # Safety limits to prevent infinite escalation
            if p_scale >= 2.0 or line_scale <= 0.5:
                self.log.warning(f"STRESS: reached safety limits at {corridor_max:.1f}%")
                break
        
        # Bookkeeping
        if cfg.get("accumulate", True):
            self._local_stress_lines |= neigh_lines
        else:
            self._local_stress_lines = neigh_lines
        
        self._local_stress_active = True
        self._local_stress_expiry = (self.current_step + duration) if duration > 0 else None

    # ======================= DEBUG / DIAGNOSTICS ======================= #
    def _debug_dump_cascade_state(self, prefix: str = "") -> None:
        try:
            print(
                f"{prefix}DEBUG • tripped_cluster={sorted(map(int, self._tripped))} "
                f"last_trip_step={self._last_trip_step} step={self.current_step}"
            )
            corr = sorted(map(int, self._corridor)) if self._corridor else []
            print(f"{prefix}DEBUG • corridor_lines(count={len(corr)}): {corr[:50]}{' ...' if len(corr)>50 else ''}")
            has_res = hasattr(self.net, "res_line") and len(self.net.res_line)
            if not has_res:
                print(f"{prefix}DEBUG • NOTE: net.res_line missing; PF failed or not run yet.")
                return
            cands = self._eligible_neighbors()
            print(f"{prefix}DEBUG • eligible_neighbors(count={len(cands)}): {cands}")
            if self.exclude_bridges and not self._bridge_lines:
                self._bridge_lines = self._compute_bridge_lines()
            if self.exclude_bridges:
                filtered = [li for li in self._corridor if li in self._bridge_lines]
                if filtered:
                    print(
                        f"{prefix}DEBUG • excluded_as_bridges(count={len(filtered)}): "
                        f"{sorted(filtered)[:40]}{' ...' if len(filtered)>40 else ''}"
                    )
            loading = self.net.res_line.loading_percent.reindex(self.net.line.index).fillna(0.0).astype(float)
            fV = self._f_V()
            dt = self.step_seconds
            print(
                f"{prefix}DEBUG • gates: overload_min={self.cascade_overload_min:.1f}%, "
                f"min_hold={self.cascade_min_hold}, hysteresis_drop={self.cascade_hysteresis_drop}, "
                f"lambda0={self.lambda0:.3f}, rho0={self.rho0:.2f}, alpha={self.alpha:.2f}, fV={fV:.3f}"
            )
            if not cands:
                reasons = []
                if not self._tripped:
                    reasons.append("no_seed_tripped")
                if not self._corridor:
                    reasons.append("empty_corridor")
                else:
                    in_service_corr = [li for li in self._corridor if bool(self.net.line.at[li, "in_service"])]
                    if not in_service_corr:
                        reasons.append("corridor_all_out_of_service")
                print(f"{prefix}DEBUG • no candidates (reasons={reasons})")
                return
            rows = []
            for li in cands:
                if not bool(self.net.line.at[li, "in_service"]):
                    continue
                Lpct = float(loading.loc[li])
                hold = int(self._hold.get(li, 0))
                fL = self._f_load(li)
                prox = self._f_prox(li)
                zone = self._f_zone(li)
                dmin = min(self._electrical_distance_between_lines(li, sj) for sj in self._tripped) if self._tripped else float("inf")
                scale = _lookup_prob(self.prob_loading_map, Lpct)
                lam = self.lambda0 * fL * fV * prox * zone * scale
                p = 1.0 - float(np.exp(-lam * dt))
                rows.append((li, Lpct, hold, fL, prox, zone, dmin, scale, lam, p))
            rows.sort(key=lambda r: r[1], reverse=True)
            if not rows:
                print(f"{prefix}DEBUG • candidates present but all filtered by in_service / fL<=0.")
                return
            print(f"{prefix}DEBUG • candidate_stats:")
            for (li, Lpct, hold, fL, prox, zone, dmin, scale, lam, p) in rows[:40]:
                print(
                    f"{prefix}   L{li:>4} | load={Lpct:6.1f}% hold={hold:2d} "
                    f"fL={fL:6.3f} prox={prox:5.3f} zone={zone:4.2f} d={dmin:6.3f} "
                    f"scale={scale:4.2f} lam={lam:7.4f} p={p:6.3f}"
                )
            if len(rows) > 40:
                print(f"{prefix}   ... ({len(rows)-40} more)")
        except Exception as e:
            print(f"{prefix}DEBUG • error in _debug_dump_cascade_state: {e}")

    # -------- seed schedule -------- #

    _AUTO_RE = re.compile(r"^line:auto:(?P<mode>top_load|betweenness)(?::k=(?P<k>\d+))?(?:,?min=(?P<min>\d+))?$")

    def _choose_seed_lines(self, strategy: str, k: int, min_percent: float) -> list[int]:
        if not hasattr(self.net, "res_line") or not len(self.net.res_line):
            return []
        if strategy == "betweenness":
            g = top.create_nxgraph(self.net, include_switches=True)
            bc = nx.betweenness_centrality(g, normalized=True)
            scores = []
            for lid, row in self.net.line.iterrows():
                u, v = int(row.from_bus), int(row.to_bus)
                load = float(self.net.res_line.loading_percent.get(lid, 0.0)) if "loading_percent" in self.net.res_line else 0.0
                scores.append((int(lid), float(bc.get(u, 0) + bc.get(v, 0)), load))
            scores.sort(key=lambda t: (t[1], t[2]), reverse=True)
            return [lid for (lid, _, _) in scores[:max(0, int(k))]]
        s = self.net.res_line.loading_percent.copy().astype(float)
        s = s[np.isfinite(s)]
        s = s[s >= float(min_percent)]
        return [int(i) for i in s.sort_values(ascending=False).head(max(0, int(k))).index]

    def _expand_auto_events(self, events: list[dict]) -> list[dict]:
        out = []
        for ev in events:
            tgt = str(ev.get("target", ""))
            m = self._AUTO_RE.match(tgt)
            if not m:
                out.append(ev)
                continue
            mode = m.group("mode")
            k = int(m.group("k") or 1)
            mn = float(m.group("min") or 0.0)
            seeds = self._choose_seed_lines("betweenness" if mode == "betweenness" else "top_load", k, mn)
            step = int(ev.get("step") or round(float(ev.get("t", 0.0)) / max(1e-9, self.step_seconds)) or 3)
            for lid in seeds:
                out.append({"step": step, "t": step * self.step_seconds, "op": ev.get("op", "trip"), "target": f"line:{int(lid)}"})
        return out

    def _seed_fault_schedule(self):
        self._seed_steps.clear()
        self._seed_lines.clear()
        if not self.cascade_enabled or self.seed_n_seeds <= 0:
            return
        if self.exclude_bridges and not self._bridge_lines:
            self._bridge_lines = self._compute_bridge_lines()
        lo_f, hi_f = self.seed_step_window_frac
        lo = max(1, int(np.floor(self.max_steps * float(lo_f))))
        hi = max(lo, int(np.ceil(self.max_steps * float(hi_f))))
        steps = [int(np.random.randint(lo, max(lo + 1, hi + 1))) for _ in range(self.seed_n_seeds)]
        self._seed_steps = sorted(steps)
        healthy = [int(i) for i in self.net.line.index if bool(self.net.line.at[i, "in_service"])]
        if self.exclude_bridges:
            healthy = [i for i in healthy if i not in self._bridge_lines] or healthy
        np.random.shuffle(healthy)
        chosen = healthy[:len(self._seed_steps)]
        for stp, li in zip(self._seed_steps, chosen):
            self._events_by_step.setdefault(stp, []).append({"t": stp * self.step_seconds, "target": f"line:{li}", "op": "trip"})

    # -------- hazard/cascade tick -------- #

    def _f_load(self, li: int) -> float:
        if not hasattr(self.net, "res_line") or not len(self.net.res_line) or li not in self.net.res_line.index:
            return 0.0
        rho = float(self.net.res_line.at[li, "loading_percent"]) / 100.0
        return max(0.0, rho - self.rho0) ** self.alpha

    def _f_V(self) -> float:
        if not hasattr(self.net, "res_bus") or not len(self.net.res_bus):
            return 1.0
        V = np.nan_to_num(self.net.res_bus.vm_pu.values, nan=1.0)
        Vmin = float(self.v_limits[0])
        deficit = np.clip(Vmin - V, 0.0, None)
        return 1.0 + self.betaV * float(np.sum(deficit))

    def _f_prox(self, lj: int) -> float:
        if not self._tripped:
            return 0.0
        dmin = min(self._electrical_distance_between_lines(lj, li) for li in self._tripped)
        return float(np.exp(-dmin / max(1e-6, self.d0)))

    def _f_zone(self, lj: int) -> float:
        return self.zone_boost if (lj in self._corridor) else 1.0

    def _cascade_tick(self) -> List[int]:
        if not self.cascade_enabled:
            return []
        if not self._tripped:
            return []
        if (self.current_step - self._last_trip_step) < self.min_gap_steps:
            return []
        if not hasattr(self.net, "res_line") or not len(self.net.res_line):
            return []

        loading = self.net.res_line.loading_percent.reindex(self.net.line.index).fillna(0.0).astype(float)
        
        # Debug: show high-loaded lines
        high_loaded = [(i, load) for i, load in loading.items() if load > self.cascade_overload_min]
        if high_loaded:
            self.log.info(f"CASCADE_CHECK: {len(high_loaded)} lines above {self.cascade_overload_min}%: {high_loaded[:5]}")

        # Update hold counters
        for li in list(self._hold.keys()):
            if not bool(self.net.line.at[li, "in_service"]):
                self._hold.pop(li, None)

        eligible = self._eligible_neighbors()
        self.log.info(f"CASCADE_CHECK: {len(eligible)} eligible neighbors in corridor")
        
        for li in self._eligible_neighbors():
            L = float(loading.loc[li])
            prev = int(self._hold.get(li, 0))
            if L >= self.cascade_overload_min:
                self._hold[li] = prev + 1
            else:
                dec = int(math.ceil(self.cascade_hysteresis_drop))
                self._hold[li] = max(0, prev - dec)

        # Find candidates meeting hold threshold
        cands = [
            li
            for li, h in self._hold.items()
            if h >= self.cascade_min_hold and bool(self.net.line.at[li, "in_service"]) and li not in self._tripped
        ]

        self.log.info(f"CASCADE_CHECK: {len(cands)} candidates met hold threshold")
        
        if not cands:
            return []

        # Calculate probabilities
        fV = self._f_V()
        dt = self.step_seconds
        probs = []
        for li in cands:
            fL = self._f_load(li)
            if fL <= 0.0:
                continue
            lam = self.lambda0 * fL * fV * self._f_prox(li) * self._f_zone(li)
            Lpct = float(loading.loc[li])
            lam *= _lookup_prob(self.prob_loading_map, Lpct)
            p = 1.0 - float(np.exp(-lam * dt))
            probs.append((li, float(np.clip(p, 0.0, 1.0))))

        if not probs:
            return []

        # Limit trips per wave
        want = min(self.per_wave_max, self.max_additional - self._additional_trips)
        if want <= 0:
            return []
        
        picks = [li for (li, p) in probs if np.random.rand() < p]
        if len(picks) > want:
            picks = [li for (li, p) in sorted(probs, key=lambda x: x[1], reverse=True) if li in picks][:want]

        # Execute trips with proper state updates
        for li in picks:
            if bool(self.net.line.at[li, "in_service"]):
                # Update pandapower
                self.net.line.at[li, "in_service"] = False
                
                # Update tracking
                self._relay_trip[int(li)] = True
                self._tripped.add(int(li))
                self._additional_trips += 1
                
                # Update boolean arrays (CRITICAL FIX)
                if hasattr(self, 'line_on') and li < len(self.line_on):
                    self.line_on[li] = False
                if hasattr(self, 'line_tripped') and li < len(self.line_tripped):
                    self.line_tripped[li] = True
                
                self.log.info(f"CASCADE_TRIP: line {li} at step {self.current_step}, loading was {loading.loc[li]:.1f}%")

        if picks:
            # Recalculate corridor with new trips
            self._corridor = self._compute_corridor(self._tripped)
            self._last_trip_step = self.current_step
            
            # Apply local stress to new cascade trips if configured
            if self.stress_local_cfg.get("enabled", False) and self.stress_local_cfg.get("on_every_trip", True):
                self._apply_local_stress_around_fault(picks)
                self.log.info(f"LOCAL_STRESS applied to {len(picks)} cascade trips at step {self.current_step}")

        return picks
    
    def _diagnose_stress_effectiveness(self):
        """Call after stress to see if it worked."""
        if not hasattr(self.net, "res_line") or not len(self.net.res_line):
            self.log.warning("DIAGNOSE: No res_line available")
            return
        
        lp = self.net.res_line.loading_percent.fillna(0.0)
        
        # Overall stats
        self.log.info(f"DIAGNOSE: Line loading stats:")
        self.log.info(f"  - Max: {lp.max():.1f}%")
        self.log.info(f"  - Mean: {lp.mean():.1f}%")
        self.log.info(f"  - >100%: {int((lp > 100).sum())} lines")
        self.log.info(f"  - >110%: {int((lp > 110).sum())} lines")
        self.log.info(f"  - >120%: {int((lp > 120).sum())} lines")
        
        # Top 20 loaded lines
        top = lp.nlargest(20)
        self.log.info(f"  - Top 20: {[(i, f'{v:.1f}%') for i, v in top.items()]}")
        
        # Stressed neighborhood
        if self._local_stress_lines:
            stressed = lp.loc[list(self._local_stress_lines)]
            self.log.info(f"DIAGNOSE: Stressed neighborhood ({len(self._local_stress_lines)} lines):")
            self.log.info(f"  - Max: {stressed.max():.1f}%")
            self.log.info(f"  - Mean: {stressed.mean():.1f}%")
            top_stressed = stressed.nlargest(10)
            self.log.info(f"  - Top 10: {[(i, f'{v:.1f}%') for i, v in top_stressed.items()]}")
            
    # ------------- observation ------------- #

    def _sense_voltages(self) -> np.ndarray:
        n = len(self.net.bus)
        if hasattr(self.net, "res_bus") and len(self.net.res_bus):
            v = np.nan_to_num(self.net.res_bus.vm_pu.values, nan=1.0, posinf=2.0, neginf=0.0)
        else:
            v = np.ones(n)
        v_scaled = (v - 0.9) / 0.2
        if self.obs_cfg.add_measurement_noise:
            v_scaled = v_scaled + np.random.normal(0.0, self.obs_cfg.noise_std_voltage, size=v_scaled.shape)
        return _clip01(v_scaled)

    def _sense_line_loading(self) -> np.ndarray:
        n = len(self.net.line)
        if hasattr(self.net, "res_line") and len(self.net.res_line):
            load = np.nan_to_num(self.net.res_line.loading_percent.values, nan=0.0, posinf=1000.0, neginf=0.0)
        else:
            load = np.zeros(n)
        load = np.minimum(load, 150.0) / 150.0
        if self.obs_cfg.add_measurement_noise:
            load = load + np.random.normal(0.0, self.obs_cfg.noise_std_loading, size=load.shape)
        return _clip01(load)

    def _read_switches(self) -> np.ndarray:
        if not len(self.net.switch):
            return np.zeros(0, dtype=np.float32)
        closed = self.net.switch["closed"].astype(bool).reindex(self.all_switch_ids).fillna(False).values
        return _clip01(closed.astype(np.float32))

    def _read_breakers(self) -> np.ndarray:
        if not len(self.net.line):
            return np.zeros(0, dtype=np.float32)
        return _clip01(self.net.line.in_service.astype(bool).values.astype(np.float32))

    def _read_relays(self) -> np.ndarray:
        if not len(self.net.line):
            return np.zeros(0, dtype=np.float32)
        return _clip01(np.array([self._relay_trip.get(int(i), False) for i in self.net.line.index], dtype=np.float32))

    def _derive_bus_energized(self) -> np.ndarray:
        if not len(self.net.bus):
            return np.zeros(0, dtype=np.float32)
        slacks = self._source_buses_for(self.net)
        try:
            unsup = set(top.unsupplied_buses(self.net, slacks=list(slacks), respect_switches=True))
        except TypeError:
            g = top.create_nxgraph(self.net, include_switches=True)
            unsup = set(self.net.bus.index.tolist())
            for s in slacks:
                try:
                    unsup -= set(top.connected_component(g, s))
                except Exception:
                    pass
        m = np.array([int(b) not in unsup for b in self.net.bus.index], dtype=np.float32)
        return _clip01(m)

    def _sense_der_channels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self.net, "gen") or not len(self.net.gen):
            return (np.zeros(0, np.float32),) * 5
        g = self.net.gen
        p_base = g.get("p_base_mw", g.get("p_mw", 1.0)).astype(float).replace(0.0, 1.0)
        p_set = _clip01((g["p_mw"].astype(float) / p_base).values.astype(np.float32))
        q_set = _clip01(np.zeros_like(p_set))
        cap = np.maximum(1e-3, p_base.values.astype(float))
        head_up = _clip01((cap - g["p_mw"].astype(float).values) / cap)
        head_dn = _clip01(g["p_mw"].astype(float).values / cap)
        der_avail = _clip01(_safe_bool_series(g.get("in_service", True), length=len(g)).astype(np.float32))
        return p_set, q_set, head_up, head_dn, der_avail

    def _sense_bess_channels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self.net, "storage") or not len(self.net.storage):
            return (np.zeros(0, np.float32),) * 3
        st = self.net.storage
        soc = _clip01((st.get("soc_percent", 50.0).astype(float).values / 100.0).astype(np.float32))
        avail_chg = np.ones(len(st), dtype=np.float32)
        avail_dis = np.ones(len(st), dtype=np.float32)
        return soc, avail_chg, avail_dis

    def _loads_per_bus_scaled(self) -> np.ndarray:
        if not len(self.net.load):
            return np.zeros(len(self.net.bus), dtype=np.float32)
        df = self.net.load
        base_col = "p_base_mw" if "p_base_mw" in df.columns else "p_mw"
        by_bus = df.groupby("bus")["p_mw"].sum().reindex(self.net.bus.index).fillna(0.0).values
        by_bus_base = df.groupby("bus")[base_col].sum().reindex(self.net.bus.index).fillna(1.0).values
        return _clip01(np.divide(by_bus, np.maximum(1e-6, by_bus_base)))

    def _tier_onehot(self):
        """
        Return a normalized 3-vector with the fraction of loads in each priority bucket.
        Priority mapping is fixed: 0=critical, 1=important, 2=other.
        """

        if not hasattr(self.net, "load") or len(self.net.load) == 0:
            # Default to "all others" if no load table
            return np.array([0.0, 0.0, 1.0], dtype=float)

        # Pull priorities; if missing, default to 2 = other
        pr = self.net.load.get("priority", 2)
        try:
            pr = np.asarray(pr, dtype=int)
        except Exception:
            # If a scalar slipped in, coerce to array
            pr = np.asarray([int(pr)], dtype=int)

        t = np.zeros(3, dtype=float)  # [critical, important, other]
        # Count by bucket
        t[0] = float(np.sum(pr == 0))  # critical
        t[1] = float(np.sum(pr == 1))  # important
        t[2] = float(np.sum(pr == 2))  # other

        total = float(np.sum(t)) if np.isfinite(np.sum(t)) else 0.0
        if total <= 0.0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return t / total


    def _build_obs_vector_from_current_state(self) -> np.ndarray:
        V = self._sense_voltages()
        I = self._sense_line_loading()
        sw = self._read_switches()
        br = self._read_breakers()
        rly = self._read_relays()
        live = self._derive_bus_energized()
        p_set, q_set, h_up, h_dn, der_avail = self._sense_der_channels()
        soc, bess_chg, bess_dis = self._sense_bess_channels()
        p_load_bus = self._loads_per_bus_scaled()
        tier = self._tier_onehot()
        t_step = _clip01(np.array([self.current_step / max(1.0, self.max_steps)], dtype=np.float32))
        wall = _clip01(np.array([self.wall_time_s / max(1.0, self.max_steps * self.step_seconds)], dtype=np.float32))
        parts = [V, I, sw, br, rly, live, p_set, q_set, h_up, h_dn, soc, bess_chg, bess_dis, der_avail, p_load_bus, tier, t_step, wall]
        return np.concatenate(parts).astype(np.float32)

    def _obs(self) -> np.ndarray:
        cur = self._build_obs_vector_from_current_state()
        self._obs_queue.append(cur.copy())
        k = self.obs_cfg.telemetry_latency_steps
        if k > 0 and len(self._obs_queue) > k:
            return self._obs_queue[-(k + 1)]
        return cur

    # ------------- reward ------------- #

    def _served_mask_by_connectivity(self, net) -> np.ndarray:
        if not len(net.load):
            return np.zeros(0, dtype=bool)
        slacks = self._source_buses_for(net)
        try:
            unsup = set(top.unsupplied_buses(net, slacks=list(slacks), respect_switches=True))
        except TypeError:
            g = top.create_nxgraph(net, include_switches=True)
            unsup = set(net.bus.index.tolist())
            for s in slacks:
                try:
                    unsup -= set(top.connected_component(g, s))
                except Exception:
                    pass
        load_buses = net.load.bus.astype(int).values
        return np.array([int(b) not in unsup for b in load_buses], dtype=bool)

    def _source_buses_for(self, net) -> set[int]:
        src = set()
        if hasattr(net, "ext_grid") and len(net.ext_grid):
            try:
                df = net.ext_grid
                m = _safe_bool_series(df.get("in_service", True), length=len(df))
                src |= set(df.loc[m, "bus"].astype(int).tolist())
            except Exception:
                pass
        for tb in ("gen",):
            df = getattr(net, tb, None)
            if df is not None and len(df):
                try:
                    m = _safe_bool_series(df.get("in_service", True), length=len(df))
                    src |= set(df.loc[m, "bus"].astype(int).tolist())
                except Exception:
                    pass
        if not src and len(net.bus):
            src = {int(net.bus.index.min())}
        return src

    def _served_fractions_by_tier(self):
        """
        Compute served-fraction (Pserved / Ptotal) for loads grouped by priority.
        Output order: [critical, important, other] corresponding to priorities [0,1,2].
        Falls back to 1.0 for any group with zero baseline demand.
        """
        # If anything is missing, return all-ones (fully served)
        if not (hasattr(self.net, "load") and len(self.net.load)):
            return np.array([1.0, 1.0, 1.0], dtype=float)

        # Get priorities; default to 2=other if absent
        if "priority" in self.net.load.columns:
            pr = self.net.load["priority"].values
        else:
            pr = np.full(len(self.net.load), 2, dtype=int)
        
        # Ensure pr is always a numpy array
        if np.isscalar(pr):
            pr = np.full(len(self.net.load), int(pr), dtype=int)
        else:
            pr = np.asarray(pr, dtype=int)
        
        # Ensure we have the right length
        if len(pr) != len(self.net.load):
            pr = np.full(len(self.net.load), 2, dtype=int)

        # Baseline P and served P
        p_total = np.asarray(self.net.load.get("p_mw", pd.Series(np.zeros(len(self.net.load)))).values, dtype=float)
        
        if hasattr(self.net, "res_load") and len(self.net.res_load) and "p_mw" in self.net.res_load:
            p_served = np.asarray(self.net.res_load["p_mw"].values, dtype=float)
            # Align length defensively if needed
            if p_served.shape[0] != p_total.shape[0]:
                n = min(p_served.shape[0], p_total.shape[0])
                p_served = p_served[:n]
                p_total = p_total[:n]
                pr = pr[:n]
        else:
            # If we don't have results, assume fully served
            return np.array([1.0, 1.0, 1.0], dtype=float)

        fracs = []
        for priority_value in (0, 1, 2):
            mask = (pr == priority_value)
            tot = float(np.nansum(p_total[mask])) if np.any(mask) else 0.0
            if tot <= 0.0:
                fracs.append(1.0)  # no demand in this group → consider fully served
            else:
                served = float(np.nansum(p_served[mask]))
                fracs.append(max(0.0, min(1.0, served / tot)))
        
        return np.asarray(fracs, dtype=float)

    def _reward(self, switches_used: int) -> Tuple[float, Dict[str, float]]:
        """Compute reward with safety checks to prevent NaN/Inf."""
        
        # Get tier fractions with validation
        f1, f2, f3 = self._served_fractions_by_tier()
        
        # Validate fractions are finite
        if not (np.isfinite(f1) and np.isfinite(f2) and np.isfinite(f3)):
            self.log.error(f"Invalid tier fractions: f1={f1}, f2={f2}, f3={f3}")
            f1, f2, f3 = 0.0, 0.0, 0.0
        
        prev_f1 = f1 if (self._prev_f1 is None) else self._prev_f1
        
        comps: Dict[str, float] = {
            "service_t1": self.w_tier[0] * f1,
            "service_t2": self.w_tier[1] * f2,
            "service_t3": self.w_tier[2] * f3,
            "restore_tier1_delta": self.w_restore_shape * float(f1 - prev_f1),
        }
        
        # Voltage violations
        if hasattr(self.net, "res_bus") and len(self.net.res_bus):
            v = np.nan_to_num(self.net.res_bus.vm_pu.values, nan=1.0)
            vmin, vmax = self.v_limits
            viol = np.clip(vmin - v, 0, None) + np.clip(v - vmax, 0, None)
            volt_penalty = self.w_volt * float(np.sum(viol))
            if np.isfinite(volt_penalty):
                comps["volt_violation"] = volt_penalty
        
        # Thermal violations
        if hasattr(self.net, "res_line") and len(self.net.res_line):
            lp = np.nan_to_num(self.net.res_line.loading_percent.values, nan=0.0)
            thermal_penalty = self.w_therm * float(np.sum(np.clip(lp - 100.0, 0.0, None)) / 100.0)
            if np.isfinite(thermal_penalty):
                comps["thermal_violation"] = thermal_penalty
        
        # Switch operations
        if switches_used > 0:
            comps["switch_ops"] = self.w_switch * float(switches_used)
        
        # DER bound violation penalty
        if self.der_cfg.get("enabled", False) and self.bound_pen_w != 0.0:
            try:
                if len(self.net.load) and "p_base_mw" in self.net.load.columns:
                    mask = self._served_mask_by_connectivity(self.net)
                    served_total = float(self.net.load.loc[mask, "p_mw"].sum())
                    base_total = float(self.net.load["p_base_mw"].sum())
                    
                    if base_total > 1e-9:
                        served_frac = served_total / base_total
                        bound = float(np.clip(self._der_bound_frac_ep, 0.0, 1.0))
                        
                        if served_frac > (bound + 0.01):
                            excess = served_frac - bound
                            bound_penalty = self.bound_pen_w * excess
                            if np.isfinite(bound_penalty):
                                comps["bound_violation"] = bound_penalty
                                self.log.debug(f"DER_BOUND_VIOLATION: served={served_frac:.3f} > bound={bound:.3f}, "
                                            f"penalty={bound_penalty:.2f}")
            except Exception as e:
                self.log.debug(f"DER bound check failed: {e}")
        
        self._prev_f1 = float(f1)
        
        # Sum all components
        total_reward = float(sum(comps.values()))
        
        # CRITICAL: Validate and clip reward
        if not np.isfinite(total_reward):
            self.log.error(f"Non-finite reward computed: {total_reward}")
            self.log.error(f"  Components: {comps}")
            total_reward = -100.0  # Safe fallback
        
        # Reasonable bounds (adjust based on your reward scale)
        # You said rewards around 500, so let's allow -1000 to +1000
        total_reward = float(np.clip(total_reward, -1000.0, 1000.0))
        
        return total_reward, comps

    # ------------- safety ------------- #


    def _hypo_apply_switch(self, net: pp.pandapowerNet, sw_id: int, close: bool):
        if sw_id in net.switch.index:
            net.switch.at[int(sw_id), "closed"] = bool(close)

    def _count_ties_closed(self, net) -> int:
        if not len(net.switch):
            return 0
        closed = net.switch.closed.astype(bool)
        cnt = 0
        for i in net.switch.index:
            if closed.at[i] and self._is_tie.get(int(i), False):
                cnt += 1
        return int(cnt)

    def _violates_voltage_proxy(self, obs_V: np.ndarray) -> bool:
        lo, hi = self.v_limits
        V = 0.9 + 0.2 * obs_V
        return (np.any(V < (lo - self.safety.voltage_margin_pu)) or np.any(V > (hi + self.safety.voltage_margin_pu)))

    def _violates_thermal_proxy(self, obs_I: np.ndarray) -> bool:
        return np.any(150.0 * obs_I > (100.0 + self.safety.loading_margin_pct))

    def _safety_filter(self, intended_action: int, obs_delayed: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        info = {"mask_empty": False, "reasons": []}
        S = self._switch_count
        a = int(np.clip(int(intended_action), 0, self.action_space.n - 1))
        if a > 2 * S:
            return a, info
        if a == S:
            return a, info
        if a < S:
            sw_idx = a
            close = False
        else:
            sw_idx = a - (S + 1)
            close = True
        if sw_idx >= S or sw_idx >= len(self.all_switch_ids):
            return S, info
        sw_id = self.all_switch_ids[sw_idx]
        if self.safety.per_step_switch_budget <= 0:
            return S, info
        if self.safety.min_dwell_steps > 0 and self._switch_dwell.get(int(sw_id), 0) > 0:
            info["reasons"].append("dwell")
            return S, info
        hypo = copy.deepcopy(self.net)
        self._hypo_apply_switch(hypo, sw_id, close=close)
        if self._count_ties_closed(hypo) > int(self.safety.max_closed_ties):
            info["reasons"].append("ties")
            return S, info
        if not self._source_buses_for(hypo):
            info["reasons"].append("no_source")
            return S, info
        n_bus = len(self.net.bus)
        n_line = len(self.net.line)
        obs_V, obs_I = obs_delayed[:n_bus], obs_delayed[n_bus : n_bus + n_line]
        if self._violates_voltage_proxy(obs_V):
            info["reasons"].append("volt_proxy")
            return S, info
        if self._violates_thermal_proxy(obs_I):
            info["reasons"].append("therm_proxy")
            return S, info
        return a, info
    
    def _diagnose_pf_failure_once(self):
        try:
            self.log.warning("PF failure diagnose: lines_in_service=%d/%d, switches_closed(l)=%d",
                            int(self.net.line.in_service.sum()) if hasattr(self.net,"line") else -1,
                            len(self.net.line) if hasattr(self.net,"line") else -1,
                            int(self.net.switch.loc[self.net.switch.et=='l',"closed"].sum()) if hasattr(self.net,"switch") and "et" in self.net.switch else -1)
        except Exception:
            pass

    def _diagnose_pf_failure(self):
        """Detailed diagnostics when PF fails."""
        self.log.error("=== POWER FLOW FAILURE DIAGNOSTICS ===")
        
        # Check for slack
        has_slack = False
        if hasattr(self.net, "gen") and len(self.net.gen):
            self.log.error(f"Generators: {len(self.net.gen)}")
            if "slack" in self.net.gen.columns:
                slack_gens = self.net.gen[self.net.gen["slack"]]
                self.log.error(f"Slack generators: {len(slack_gens)}")
                for idx, row in slack_gens.iterrows():
                    self.log.error(f"  - Gen {idx}: bus={row.bus}, in_service={row.get('in_service', True)}, p_mw={row.p_mw}")
                has_slack = len(slack_gens) > 0
        
        if hasattr(self.net, "ext_grid") and len(self.net.ext_grid):
            self.log.error(f"Ext grids: {len(self.net.ext_grid)}")
            for idx, row in self.net.ext_grid.iterrows():
                self.log.error(f"  - ExtGrid {idx}: bus={row.bus}, in_service={row.get('in_service', True)}")
        
        if not has_slack and not (hasattr(self.net, "ext_grid") and len(self.net.ext_grid)):
            self.log.error("NO SLACK SOURCE FOUND!")
        
        # Check connectivity
        try:
            if hasattr(self.net, "line"):
                in_service_lines = int(self.net.line["in_service"].sum())
                self.log.error(f"Lines in service: {in_service_lines}/{len(self.net.line)}")
        except Exception as e:
            self.log.error(f"Error checking lines: {e}")
        
        # Check loads
        try:
            if hasattr(self.net, "load"):
                total_load = float(self.net.load["p_mw"].sum())
                self.log.error(f"Total load: {total_load:.2f} MW")
        except Exception as e:
            self.log.error(f"Error checking loads: {e}")


    # ------------- gym api ------------- #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # Copy base network
        self.net = copy.deepcopy(self.net0)
        
        # CRITICAL: Ensure slack generator stays in service
        if hasattr(self.net, "gen") and len(self.net.gen):
            if "slack" in self.net.gen.columns:
                slack_mask = self.net.gen["slack"].astype(bool)
                if slack_mask.any():
                    # Force slack generators to be in service
                    self.net.gen.loc[slack_mask, "in_service"] = True
                    self.log.debug(f"Forced {slack_mask.sum()} slack generator(s) to in_service=True")
        
        # Double-check we have a valid slack
        has_slack = False
        if hasattr(self.net, "gen") and len(self.net.gen) and "slack" in self.net.gen.columns:
            slack_in_service = self.net.gen[self.net.gen["slack"] & self.net.gen["in_service"]]
            has_slack = len(slack_in_service) > 0
            if has_slack:
                self.log.debug(f"Found {len(slack_in_service)} in-service slack generator(s)")
        
        has_ext_grid = hasattr(self.net, "ext_grid") and len(self.net.ext_grid) > 0
        
        if not has_slack and not has_ext_grid:
            # Emergency: create a slack generator
            self.log.warning("No in-service slack found after reset - creating emergency slack")
            bus_idx = 0
            if hasattr(self.net, "gen") and len(self.net.gen):
                # Try to use an existing generator bus
                bus_idx = int(self.net.gen.bus.iloc[0])
            
            total_load = float(self.net.load["p_mw"].sum()) if len(self.net.load) else 10.0
            gen_size = total_load * 1.5  # 50% margin
            
            pp.create_gen(
                self.net,
                bus=bus_idx,
                p_mw=0.0,
                vm_pu=1.0,
                slack=True,
                min_p_mw=-gen_size,
                max_p_mw=gen_size,
                controllable=True,
                name="emergency_slack_gen",
                in_service=True  # MUST be True!
            )
            self.log.info(f"Created emergency slack at bus {bus_idx}, in_service=True")
        
        # Only close switches that should be closed at baseline
        if hasattr(self.net, "switch") and not self.net.switch.empty:
            if hasattr(self.net0, "switch"):
                self.net.switch["closed"] = self.net0.switch["closed"].copy()

        # Rest of your reset code remains the same...
        # Initialize line state arrays to match current topology
        nL = len(self.net.line) if hasattr(self.net, "line") else 0
        if nL > 0:
            in_service = self.net.line["in_service"].to_numpy()
            self.line_on = in_service.copy()
            self.line_failed = np.zeros(nL, dtype=bool)
            self.line_tripped = np.zeros(nL, dtype=bool)
            self.line_forced_off = np.zeros(nL, dtype=bool)
            self.line_blocked = np.zeros(nL, dtype=bool)
        
        # Reset runtime state
        self.prev = copy.deepcopy(self.net)
        self.current_step = 0
        self.wall_time_s = 0.0
        self._obs_queue.clear()
        self._relay_trip = {int(i): False for i in self.net.line.index}
        self._switch_dwell = {int(i): 0 for i in self.all_switch_ids}
        self._events_by_step.clear()
        self.pf_fail_streak = 0
        self._prev_f1 = None

        # Reset cascade state
        self._seed_steps.clear()
        self._seed_lines.clear()
        self._tripped.clear()
        self._hold.clear()
        self._additional_trips = 0
        self._last_trip_step = -10**9
        self._corridor.clear()
        self._line_dist_cache.clear()
        self._bridge_lines = set()

        # Reset local stress
        self._local_stress_active = False
        self._local_stress_expiry = None
        self._local_stress_lines = set()

        # Schedule random seed faults
        self._seed_fault_schedule()

        # Apply global stress (loads/ratings)
        stress_cfg = self.stress_global or {}
        if stress_cfg:
            self._apply_global_stress(stress_cfg)
        
        # Apply DER unavailability BEFORE baseline snapshot
        if self.der_cfg.get("enabled", False):
            self._apply_der_unavailability(seed=seed)
        else:
            self._der_bound_frac_ep = 1.0

        # CRITICAL: Don't let DER unavailability turn off slack!
        if hasattr(self.net, "gen") and len(self.net.gen) and "slack" in self.net.gen.columns:
            slack_mask = self.net.gen["slack"].astype(bool)
            self.net.gen.loc[slack_mask, "in_service"] = True

        # Snapshot baselines AFTER stress
        self._snapshot_baselines()
        self._enforce_frozen_generation()

        # Debug: log slack status before PF
        self.log.debug("Pre-PF slack check: gen=%s (in_service=%s), ext_grid=%s", 
                    self.net.gen["slack"].any() if hasattr(self.net, "gen") and "slack" in self.net.gen.columns else False,
                    self.net.gen[self.net.gen["slack"]]["in_service"].all() if hasattr(self.net, "gen") and "slack" in self.net.gen.columns and self.net.gen["slack"].any() else False,
                    len(self.net.ext_grid) > 0 if hasattr(self.net, "ext_grid") else False)

        # Initial power flow
        ok = self._runpf()
        if not ok:
            # More detailed error logging
            self._diagnose_pf_failure()
            raise RuntimeError("Initial AC power flow failed at RESET")

        # Expand auto-events (needs PF results)
        scen = self.scenario or {"events": [], "repairs": []}
        raw_events = list(scen.get("events", []))
        if raw_events:
            concrete = self._expand_auto_events(raw_events)
            for ev in concrete:
                t_sec = float(ev.get("t", 0.0))
                step = int(round(t_sec / max(1e-9, self.step_seconds))) if "t" in ev else int(ev.get("step", 0))
                self._events_by_step.setdefault(step, []).append(ev)

        # Run PF again after any adjustments
        self._runpf()

        # Log initial state
        if hasattr(self.net, "res_line") and len(self.net.res_line):
            top = self.net.res_line.loading_percent.sort_values(ascending=False).head(10)
            self.log.info("RESET top line loads: %s", "; ".join(f"{i}:{v:.1f}%" for i, v in top.items()))

        obs = self._obs()
        f1, _, _ = self._served_fractions_by_tier()
        self._prev_f1 = float(f1)
        info = self._info_dict(pf_ok=True)

        # Add diagnostic counts
        live_by_status = int(np.count_nonzero(self.net.line.in_service.to_numpy()))
        info["live_lines"] = live_by_status
        
        self.log.info(f"RESET complete: live_lines={live_by_status}/{nL}, cascade_enabled={self.cascade_enabled}")

        return obs, info

    def step(self, action):
        self.prev = copy.deepcopy(self.net)
        self.current_step += 1
        self.wall_time_s += self.step_seconds

        # Optional expiry of local stress overlay
        if self._local_stress_active and self._local_stress_expiry is not None and self.current_step >= self._local_stress_expiry:
            # Persisting overlay is simpler; skipping automatic revert to keep behavior deterministic under cascades.
            self._local_stress_expiry = None  # stop checking further

        delayed_obs = self._obs_queue[-1] if len(self._obs_queue) else self._build_obs_vector_from_current_state()
        filtered_action, mask_info = self._safety_filter(action, delayed_obs)

        switches_used = 0
        S = self._switch_count
        if filtered_action <= 2 * S:
            if filtered_action < S:
                sw_idx = filtered_action
                if sw_idx < len(self.all_switch_ids):
                    sw_id = self.all_switch_ids[sw_idx]
                    self.net.switch.at[int(sw_id), "closed"] = False
                    switches_used = 1
                    self._switch_dwell[int(sw_id)] = self.safety.min_dwell_steps
            elif filtered_action == S:
                pass
            else:
                sw_idx = filtered_action - (S + 1)
                if sw_idx < len(self.all_switch_ids):
                    sw_id = self.all_switch_ids[sw_idx]
                    self.net.switch.at[int(sw_id), "closed"] = True
                    switches_used = 1
                    self._switch_dwell[int(sw_id)] = self.safety.min_dwell_steps
        else:
            if self.enable_setpoint_shedding and len(self.load_ids):
                rel = filtered_action - (2 * S + 1)
                if 0 <= rel < 2 * len(self.load_ids):
                    lid = self.load_ids[rel // 2]
                    direction = +1 if (rel % 2 == 0) else -1
                    if "shed_frac" not in self.net.load.columns:
                        self.net.load["shed_frac"] = 0.0
                    if "p_base_mw" not in self.net.load.columns:
                        self.net.load["p_base_mw"] = self.net.load["p_mw"].astype(float)
                    if "q_mvar" in self.net.load.columns and "q_base_mvar" not in self.net.load.columns:
                        self.net.load["q_base_mvar"] = self.net.load["q_mvar"].astype(float)
                    sf = float(self.net.load.at[lid, "shed_frac"])
                    sf = float(np.clip(sf + direction * self.shed_step, 0.0, 1.0))
                    self.net.load.at[lid, "shed_frac"] = sf
                    p0 = float(self.net.load.at[lid, "p_base_mw"])
                    self.net.load.at[lid, "p_mw"] = p0 * (1.0 - sf)
                    if "q_mvar" in self.net.load.columns:
                        q0 = float(self.net.load.at[lid, "q_base_mvar"])
                        self.net.load.at[lid, "q_mvar"] = q0 * (1.0 - sf)

        for k in list(self._switch_dwell.keys()):
            self._switch_dwell[k] = max(0, int(self._switch_dwell[k]) - 1)

        # Apply any scheduled event
        new_trips_applied: List[int] = []
        for ev in self._events_by_step.get(self.current_step, []):
            tgt = ev.get("target", "")
            if tgt.startswith("line:"):
                try:
                    lid = int(tgt.split(":")[1])
                except Exception:
                    lid = None
            else:
                lid = None
            prev_tripped = set(self._tripped)
            self._apply_event(ev)
            if lid is not None and lid in self._tripped and lid not in prev_tripped:
                new_trips_applied.append(lid)

        self._enforce_frozen_generation()

        pf_ok = self._runpf()

        # Perform cascade tick (probabilistic subfaults)
        if pf_ok:
            new_trips = self._cascade_tick()
            if new_trips:
                pf_ok = self._runpf()
        else:
            new_trips = []

        if not pf_ok:
            self.pf_fail_streak += 1
        else:
            self.pf_fail_streak = 0
        reward, comps = self._reward(switches_used=switches_used)
        if not pf_ok:
            comps["pf_fail"] = float(self.pf_failure_penalty)
        obs = self._obs()
        done = (self.current_step >= self.max_steps) or (self.pf_fail_streak >= self.failure_patience)
        info = self._info_dict(pf_ok=pf_ok)
        info.update(mask_info)
        info["reward_components"] = comps

        # Debug only (for sanity.py)
        applied = [f"{ev.get('target','?')} {ev.get('op','?')}" for ev in self._events_by_step.get(self.current_step, [])]
        if new_trips:
            applied.extend([f"line:{li} trip(cascade)" for li in new_trips])
        if applied:
            info["events_applied"] = applied

        return obs, reward, bool(done), False, info

    # ------------- power flow ------------- #

    def _runpf(self) -> bool:
        # AC ladder with DC warm start. Keep robust; failures are reported, not fatal.
        try:
            pp.runpp(
                self.net,
                algorithm="nr",
                init="flat",
                enforce_q_lims=False,
                calculate_voltage_angles=True,
                tolerance_mva=1e-6,
                max_iteration=60,
            )
            pp.runpp(
                self.net,
                algorithm="nr",
                init="results",
                enforce_q_lims=True,
                calculate_voltage_angles=True,
                tolerance_mva=1e-7,
                max_iteration=80,
            )
            return True
        except Exception:
            pass
        try:
            pp.rundcpp(self.net)
            if hasattr(self.net, "res_bus") and len(self.net.res_bus):
                self.net.res_bus["vm_pu"] = 1.0
            pp.runpp(
                self.net,
                algorithm="nr",
                init="results",
                enforce_q_lims=False,
                calculate_voltage_angles=True,
                tolerance_mva=1e-6,
                max_iteration=60,
            )
            pp.runpp(
                self.net,
                algorithm="nr",
                init="results",
                enforce_q_lims=True,
                calculate_voltage_angles=True,
                tolerance_mva=1e-7,
                max_iteration=80,
            )
            return True
        except Exception:
            pass
        try:
            pp.runpp(
                self.net,
                algorithm="fdxb",
                init="results",
                enforce_q_lims=False,
                calculate_voltage_angles=True,
                tolerance_mva=1e-6,
                max_iteration=80,
            )
            pp.runpp(
                self.net,
                algorithm="fdxb",
                init="results",
                enforce_q_lims=True,
                calculate_voltage_angles=True,
                tolerance_mva=1e-7,
                max_iteration=80,
            )
            return True
        except Exception as e:
            self.log.warning("AC power flow failed after ladder: %s", e)
            return False

    # ------------- global stress ------------- #

    def _apply_global_stress(self, cfg: dict):
        if not cfg:
            return
        lp = float(cfg.get("load_scale_p", 1.0))
        lq = float(cfg.get("load_scale_q", lp))
        rs = float(cfg.get("line_rating_scale", 1.0))
        gc = float(cfg.get("gen_cap_scale", 1.0))
        if len(self.net.load):
            if "p_mw" in self.net.load:
                self.net.load["p_mw"] = self.net.load["p_mw"].astype(float) * lp
            if "q_mvar" in self.net.load:
                self.net.load["q_mvar"] = self.net.load["q_mvar"].astype(float) * lq
            if "p_base_mw" in self.net.load:
                self.net.load["p_base_mw"] = self.net.load["p_base_mw"].astype(float) * lp
            if "q_base_mvar" in self.net.load and "q_mvar" in self.net.load:
                self.net.load["q_base_mvar"] = self.net.load["q_base_mvar"].astype(float) * lq
        if len(self.net.line) and "max_i_ka" in self.net.line:
            s = max(0.2, min(1.0, rs))
            self.net.line["max_i_ka"] = np.maximum(1e-6, self.net.line["max_i_ka"].astype(float) * s)
            if "max_loading_percent" in self.net.line:
                self.net.line["max_loading_percent"] = np.maximum(self.net.line["max_loading_percent"], 200.0)
        if gc < 0.999 and len(self.net.gen) and "p_mw" in self.net.gen:
            self.net.gen["p_mw"] = np.minimum(self.net.gen["p_mw"], self.net.gen["p_mw"] * gc)

    # ------------- info ------------- #

    def _ensure_info_keys(self, info: dict) -> dict:
        # Total served (by connectivity; normalized to base)
        try:
            if not len(self.net.load):
                tot = 1.0
            else:
                df = self.net.load
                if "p_base_mw" not in df.columns:
                    df["p_base_mw"] = df["p_mw"].astype(float)
                mask = self._served_mask_by_connectivity(self.net)
                tot = float(df.loc[mask, "p_mw"].sum()) / float(max(1e-9, df["p_base_mw"].sum()))
        except Exception:
            tot = 0.0

        # Served by priority tiers (uses res_load if available)
        f1, f2, _ = self._served_fractions_by_tier()

        # Canonical keys
        info.setdefault("served_total_frac", float(tot))
        info.setdefault("served_crit_frac", float(f1))
        info.setdefault("served_imp_frac", float(f2))
        info.setdefault("powerflow_success", True)

        # Lines: status vs. actually carrying flow
        try:
            live_by_status = int(np.count_nonzero(self.net.line.in_service.to_numpy()))
        except Exception:
            live_by_status = -1
        try:
            lp = self.net.res_line.loading_percent.to_numpy()
            loaded_by_flow = int(np.count_nonzero(np.nan_to_num(lp, nan=0.0) > 0.0))  # “loaded” = any nonzero flow
        except Exception:
            loaded_by_flow = -1

        info["live_lines"] = live_by_status
        info["loaded_lines"] = loaded_by_flow

        # Energized buses (supplied by at least one source, not just in_service)
        try:
            slacks = self._source_buses_for(self.net)
            try:
                unsup = set(top.unsupplied_buses(self.net, slacks=list(slacks), respect_switches=True))
            except TypeError:
                g = top.create_nxgraph(self.net, include_switches=True)
                unsup = set(self.net.bus.index.tolist())
                for s in slacks:
                    try:
                        unsup -= set(top.connected_component(g, s))
                    except Exception:
                        pass
            energized_buses = int(len(self.net.bus) - len(unsup))
        except Exception:
            energized_buses = -1
        info["energized_buses"] = energized_buses

        return info



    def _info_dict(self, pf_ok: bool) -> dict:
        info = {
            "powerflow_success": bool(pf_ok),
            "scenario_name": str(self.scenario.get("name", "scenario")),
            "episode_step": int(self.current_step),
        }
    
        return self._ensure_info_keys(info)
