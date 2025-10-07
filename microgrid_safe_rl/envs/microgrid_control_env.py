# microgrid_safe_rl/envs/microgrid_control_env.py
from __future__ import annotations

import copy
import logging
from typing import Dict, Any, Optional, List, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandapower as pp

from microgrid_safe_rl.augmentation.common import assign_priorities


class MicrogridControlEnv(gym.Env):
    """
    RL environment for islanded microgrid switching & load shedding with:
      • live line fault isolation
      • optional load surges
      • overload-driven cascading trips (optionally live)
      • optional DER unavailability
      • switch-subset action space
      • detailed disturbance logs
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------ #
    # __init__
    # ------------------------------------------------------------------ #
    def __init__(self, net, config: Dict[str, Any], scenario: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.scenario = scenario or {"enabled": True, "modes": ["line", "load"], "severity": 0.8}

        # Limits and episode length
        self.v_limits: Tuple[float, float] = tuple(self.config.get("voltage_limits", (0.95, 1.05)))
        self.max_steps = int(self.config.get("max_steps", 50))

        # PF failure handling
        self.failure_patience = int(self.config.get("pf_failure_patience", 3))
        self.pf_failure_penalty = float(self.config.get("reward_weights", {}).get("pf_failure", -30.0))

        # Behavior toggles
        self.enable_setpoint_shedding = bool(self.config.get("enable_setpoint_shedding", False))
        self.mask_load_actions_during_live_fault = bool(
            self.config.get("mask_load_actions_during_live_fault", True)
        )
        self.invalid_live_action_penalty = float(
            self.config.get("reward_weights", {}).get("invalid_live_action", -2.0)
        )

        # Prepare & keep a pristine prepared net
        self._prepare_net_inplace(net)
        self.net0 = copy.deepcopy(net)

        # Working episode state
        self.net = copy.deepcopy(net)
        self.prev = copy.deepcopy(net)
        self.current_step = 0
        self.pf_fail_streak = 0

        # Fault bookkeeping
        self.faulted_line: Optional[int] = None
        self._fault_live = False
        self._fault_live_steps = 0
        self._fault_isolated_at: Optional[int] = None
        self.protection_trip_delay = int(self.config.get("protection_trip_delay", 3))
        self._fault_award_pending = False
        self._fault_awarded = False

        # DER availability
        self.der_cfg = self.config.get("der_unavailability", {})
        self.bound_pen_w = float(self.config.get("reward_weights", {}).get("bound_violation", -20.0))
        self._der_bound_frac_ep = 1.0

        # Cascading state
        self._active_faults: set[int] = set()
        self._overload_watch: Dict[int, int] = {}
        self._cascade_remaining = 0
        self._c_rt = None  # per-episode cascade runtime parameters

        # Full lists & fixed action sizes
        self.all_switch_ids = list(self.net.switch.index) if len(self.net.switch) else []
        self.load_ids = list(self.net.load.index) if len(self.net.load) else []

        # Fixed K size for switches
        K_cfg = int(self.config.get("max_switches", 0) or 0)
        self.K = K_cfg if (K_cfg > 0 and K_cfg <= len(self.all_switch_ids)) else len(self.all_switch_ids)

        # L loads if setpoint shedding enabled
        self.L = len(self.load_ids) if self.enable_setpoint_shedding else 0

        # Per-episode subset (rebuilt after we know the fault)
        self.switch_ids = self._select_switch_subset(faulted_line=None)

        # Spaces
        n_buses = len(self.net.bus)
        self.action_space = spaces.Discrete(1 + self.K + 2 * self.L)
        self.observation_space = spaces.Box(
            low=np.array([self.v_limits[0]] * n_buses + [0.0] * self.K + [0.0] * self.L, dtype=np.float32),
            high=np.array([self.v_limits[1]] * n_buses + [1.0] * self.K + [1.0] * self.L, dtype=np.float32),
            dtype=np.float32,
        )

    # -------------------------- cascade config -------------------------- #
    def _cascade_cfg(self) -> dict:
        """Read cascade sub-config with sane defaults even if missing/malformed."""
        c = self.scenario.get("cascade", {}) or {}
        if not isinstance(c, dict):
            c = {}  # guard against None/list/str
        return {
            "strategy":           str(c.get("strategy", "overload")),
            "max_additional":     int(c.get("max_additional", 0)),
            "per_wave_max":       int(c.get("per_wave_max", 1)),
            "hop_limit":          int(c.get("hop_limit", 2)),
            "overload_min_pct":   float(c.get("overload_min_pct", 115.0)),
            "min_hold_steps":     int(c.get("min_hold_steps", 2)),
            "hysteresis_drop":    float(c.get("hysteresis_drop", 3.0)),
            "prob_loading_map":   list(c.get("prob_loading_map", [(115.0, 0.15), (125.0, 0.45), (140.0, 0.85)])),
            "live_faults":        bool(c.get("live_faults", False)),
            "neighbor_sample_k":  int(c.get("neighbor_sample_k", 0)),
        }

    def _sample_cascade_runtime(self, rng):
        """Per-episode jitter of cascade severity."""
        c = self._cascade_cfg()
        def jit(x, frac, floor=0.0):
            return float(np.clip(rng.normal(x, frac * max(1e-6, x)), floor, 1e9))
        self._c_rt = {
            "overload_min_pct":   jit(c["overload_min_pct"], 0.05),
            "min_hold_steps":     max(1, int(round(jit(c["min_hold_steps"], 0.25)))),
            "per_wave_max":       max(1, int(round(jit(c["per_wave_max"], 0.25)))),
            "max_additional":     max(0, int(round(jit(c["max_additional"], 0.20)))),
            "hysteresis_drop":    max(0.5, jit(c["hysteresis_drop"], 0.20)),
            "hop_limit":          int(c["hop_limit"]),
            "live_faults":        bool(c["live_faults"]),
            "neighbor_sample_k":  int(c["neighbor_sample_k"]),
            "prob_scale":         float(rng.uniform(0.85, 1.25)),
            "prob_loading_map":   list(c["prob_loading_map"]),
            "strategy":           str(c["strategy"]),
        }
        self.log.debug("[CASCADE] Runtime cfg (episode): %s", self._c_rt)

    # --------------------- der toggle for scenario ---------------------- #
    def _der_is_enabled_for_this_episode(self) -> bool:
        return bool(self.scenario.get("der", False) and self.der_cfg.get("enabled", False))

    # --------------------------- net prepare ---------------------------- #
    def _prepare_net_inplace(self, net):
        if len(net.load):
            if "p_base_mw" not in net.load.columns:
                net.load["p_base_mw"] = net.load["p_mw"].astype(float)
            if "q_base_mvar" not in net.load.columns and "q_mvar" in net.load.columns:
                net.load["q_base_mvar"] = net.load["q_mvar"].astype(float)
            if "shed_frac" not in net.load.columns:
                net.load["shed_frac"] = 0.0
            if "priority" not in net.load.columns:
                assign_priorities(net)

        for tb in ("gen", "sgen"):
            df = getattr(net, tb, None)
            if df is not None and len(df):
                if "p_base_mw" not in df.columns and "p_mw" in df.columns:
                    df["p_base_mw"] = df["p_mw"].astype(float)
                if "in_service" not in df.columns:
                    df["in_service"] = True

    # ------------------------------ reset ------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Episode counters and flags
        self.current_step = 0
        self.pf_fail_streak = 0
        self.faulted_line = None
        self._fault_live = False
        self._fault_live_steps = 0
        self._fault_award_pending = False
        self._fault_awarded = False
        self._fault_isolated_at = None

        # disturbances log
        self._disturbance_log: List[dict] = []

        # Cascading state
        self._active_faults.clear()
        self._overload_watch.clear()
        self._cascade_remaining = 0
        self._c_rt = None

        # Grace controls
        self._grace_budget_max = int(self.config.get("grace_steps", 0))
        self._grace_budget = 0
        self._needs_grace = False

        retries = int(self.config.get("reset_retries", 3))
        ok = False
        rng = np.random.default_rng(seed)

        # per-episode cascade runtime params
        self._sample_cascade_runtime(rng)
        self._cascade_remaining = int(self._c_rt["max_additional"])

        for t in range(max(1, retries)):
            self.net = copy.deepcopy(self.net0)
            self._prepare_net_inplace(self.net)
            self.prev = copy.deepcopy(self.net)

            # Disturbance
            if self.scenario.get("enabled", True):
                self._inject_disturbance(seed=None if seed is None else seed + t)
                if self.faulted_line is not None:
                    on_fault = self._on_fault_switch_ids()
                    self.log.debug("[RESET] Primary live fault on line %s; on-fault switches: %s",
                                   self.faulted_line, on_fault)

            # DER (if scenario enables it)
            if bool(self.scenario.get("der", False)):
                self.der_cfg = self.config.get("der_unavailability", {})
                self._apply_der_unavailability(seed=None if seed is None else seed + 1337 + t)
            else:
                self._der_bound_frac_ep = 1.0

            # Rebuild K-sized switch subset AFTER we know the fault
            self.switch_ids = self._select_switch_subset(self.faulted_line)
            self.log.debug("[RESET] K-switch subset (K=%d): %s", self.K, self.switch_ids)

            ok = self._runpf()
            self.log.debug("[RESET] Initial PF ok=%s, fault_live=%s, cascade_remaining=%d",
                           ok, self._fault_live, self._cascade_remaining)

            if not ok and not self._fault_live:
                self._needs_grace = (self._grace_budget_max > 0)
                self._grace_budget = self._grace_budget_max
                if self._needs_grace:
                    self.log.debug("[RESET] PF infeasible -> start grace window (%d steps)", self._grace_budget)
                    ok = True
                break

            if ok or self._fault_live:
                break

        info = {
            "powerflow_success": ok,
            "faulted_line": self.faulted_line,
            "der_bound_frac": float(self._der_bound_frac_ep),
            "grace_budget": int(self._grace_budget),
            "needs_grace": bool(self._needs_grace),
            "disturbance_log": list(self._disturbance_log),
        }
        info = self._ensure_info_keys(info) 
        return self._obs(), info

    # ------------------------- fault selection -------------------------- #
    def _safe_fault_line(self) -> Optional[int]:
        try:
            import networkx as nx  # noqa: F401
            import pandapower.topology as top
            g = top.create_nxgraph(self.net, include_switches=True)
            bridges = set(tuple(sorted(e)) for e in nx.bridges(g))
            safe = []
            for li, row in self.net.line.iterrows():
                if not bool(row.in_service):
                    continue
                a, b = int(row.from_bus), int(row.to_bus)
                if tuple(sorted((a, b))) not in bridges:
                    safe.append(int(li))
            if not safe:
                safe = [int(i) for i, r in self.net.line.iterrows() if bool(r.in_service)]
            return None if not safe else int(np.random.default_rng().choice(safe))
        except Exception as e:
            self.log.debug("_safe_fault_line fallback due to: %s", e)
            if len(self.net.line):
                cands = [int(i) for i, r in self.net.line.iterrows() if bool(r.in_service)]
                return None if not cands else int(np.random.default_rng().choice(cands))
            return None

    def _baseline_loading_probs(self) -> Optional[np.ndarray]:
        """Return per-line probabilities ~ baseline loading on intact net (bias faults to stressed lines)."""
        if not len(self.net.line):
            return None
        tmp = copy.deepcopy(self.net0)
        try:
            pp.runpp(tmp, enforce_q_lims=True)
        except Exception:
            return None
        if not hasattr(tmp, "res_line") or not len(tmp.res_line):
            return None
        load = np.maximum(0.0, np.nan_to_num(tmp.res_line.loading_percent.values))
        if load.sum() <= 0:
            return None
        m = tmp.line.in_service.astype(bool).values
        p = load * m
        s = p.sum()
        return (p / s) if s > 0 else None

    def _pick_fault_line(self, rng, seed=None) -> Optional[int]:
        if not len(self.net.line):
            return None
        p = float(self.scenario.get("exclude_bridges_prob", 0.0))
        excl = bool(self.scenario.get("exclude_bridges", False) or (rng.random() < p))
        if excl:
            return self._safe_fault_line()
        probs = self._baseline_loading_probs()
        if probs is not None:
            idx = np.array(self.net.line.index)
            return int(rng.choice(idx, p=probs))
        return int(np.random.default_rng(seed).choice(self.net.line.index))

    # --------------------------- disturbance ---------------------------- #
    def _inject_disturbance(self, seed=None):
        rng = np.random.default_rng(seed)
        modes = self.scenario.get("modes", ["line", "load"])
        sev = float(self.scenario.get("severity", 0.8))
        mode = rng.choice(modes) if modes else None

        if mode == "line" and len(self.net.line):
            li = self._pick_fault_line(rng, seed)
            if li is not None:
                self.faulted_line = int(li)
                self._fault_live = True
                self._fault_live_steps = 0
                self.switch_ids = self._select_switch_subset(self.faulted_line)
                self._active_faults.add(int(li))
                self._disturbance_log.append({"t": 0, "type": "primary_line_fault", "line": int(li)})
                self.log.debug("[DISTURBANCE] Primary live fault on line %s", li)
            return

        if mode == "load" and len(self.net.load):
            li = int(rng.choice(self.net.load.index))
            f = float(np.clip(1.0 + 0.2 * sev + rng.uniform(0, 0.3 * sev), 1.0, 2.0))
            self.net.load.at[li, "p_mw"] = float(self.net.load.at[li, "p_mw"]) * f
            if "q_mvar" in self.net.load.columns:
                self.net.load.at[li, "q_mvar"] = float(self.net.load.at[li, "q_mvar"]) * f
            self._disturbance_log.append({"t": 0, "type": "load_surge", "load": int(li), "scale": float(f)})
            self.log.debug("[DISTURBANCE] Load surge on load %s scale %.2f", li, f)

    # -------------------------- DER unavailability ---------------------- #
    def _sample_der_bound_frac(self, rng) -> float:
        cfg = self.der_cfg or {}
        lo, hi = cfg.get("p_deficit_frac", [0.0, 0.0])
        lo = float(lo); hi = float(hi)
        hi = max(hi, lo)
        deficit = rng.uniform(lo, hi)
        return float(np.clip(1.0 - deficit, 0.0, 1.0))

    def _apply_der_unavailability(self, seed=None):
        if not self.der_cfg or not self.der_cfg.get("static_on_reset", False):
            rng = np.random.default_rng(seed)
            self._der_bound_frac_ep = self._sample_der_bound_frac(rng)
            self.log.debug("[DER] Ep bound frac=%.3f (no static rescale)", self._der_bound_frac_ep)
            return

        rng = np.random.default_rng(seed)
        cfg = self.der_cfg
        self._der_bound_frac_ep = self._sample_der_bound_frac(rng)

        outage_p = float(cfg.get("random_outage_prob", 0.0))
        s_lo, s_hi = cfg.get("scaling_range", [0.5, 1.0])
        s_lo = float(s_lo); s_hi = float(max(s_hi, s_lo))
        max_size = float(cfg.get("max_der_size_mw", 1e9))
        max_disabled = int(cfg.get("max_disabled", 1))

        if len(self.net.load):
            if "p_base_mw" not in self.net.load.columns:
                self.net.load["p_base_mw"] = self.net.load["p_mw"].astype(float)
            P_L = float(self.net.load["p_base_mw"].sum())
        else:
            P_L = 0.0
        P_target = self._der_bound_frac_ep * P_L

        der_tables = [tb for tb in ("gen", "sgen") if hasattr(self.net, tb) and len(getattr(self.net, tb))]
        if not der_tables:
            return

        disabled = []
        disabled_count = 0
        for tb in der_tables:
            if disabled_count >= max_disabled:
                break
            df = getattr(self.net, tb)
            if not len(df):
                continue
            if rng.random() < outage_p:
                small = df[(df.in_service.astype(bool)) & (df.p_mw <= max_size)]
                if len(small):
                    idx = int(rng.choice(small.index))
                    disabled.append((tb, idx, float(df.at[idx, "p_mw"])))
                    df.at[idx, "in_service"] = False
                    disabled_count += 1

        bases = []
        for tb in der_tables:
            df = getattr(self.net, tb)
            m = df.in_service.astype(bool)
            if "p_base_mw" in df.columns:
                bases.append(float(df.loc[m, "p_base_mw"].sum()))
            else:
                bases.append(float(df.loc[m, "p_mw"].sum()))
        P_base_avail = float(sum(bases))

        if P_base_avail > 1e-9:
            common_scale = float(np.clip(P_target / P_base_avail, s_lo, s_hi))
            for tb in der_tables:
                df = getattr(self.net, tb)
                m = df.in_service.astype(bool)
                if int(m.sum()) == 0:
                    continue
                jitter = rng.uniform(s_lo, s_hi, size=int(m.sum()))
                if "p_base_mw" in df.columns:
                    df.loc[m, "p_mw"] = df.loc[m, "p_base_mw"].values * (common_scale * jitter)
                else:
                    df.loc[m, "p_mw"] = df.loc[m, "p_mw"].values * (common_scale * jitter)

        P_online = 0.0
        for tb in der_tables:
            df = getattr(self.net, tb)
            P_online += float(df.loc[df.in_service.astype(bool), "p_mw"].sum())

        self.log.debug("[DER] bound_frac=%.3f | P_load=%.2f | P_target=%.2f | P_online=%.2f | disabled=%s",
                       self._der_bound_frac_ep, P_L, P_target, P_online, disabled or "None")

    # -------------------------- switch selection ------------------------ #
    def _on_fault_switch_ids(self) -> List[int]:
        if self.faulted_line is None or not len(self.net.switch):
            return []
        try:
            return list(self.net.switch.index[
                (self.net.switch["et"] == "l") &
                (self.net.switch["element"] == int(self.faulted_line))
            ])
        except Exception:
            return []

    def _select_switch_subset(self, faulted_line: Optional[int]):
        if not len(self.all_switch_ids) or self.K == 0:
            return []
        must_have = []
        if faulted_line is not None and len(self.net.switch):
            try:
                on_fault = self.net.switch.index[
                    (self.net.switch["et"] == "l") & (self.net.switch["element"] == int(faulted_line))
                ].tolist()
                must_have = [int(s) for s in on_fault]
            except Exception:
                must_have = []

        if self._fault_live and must_have:
            subset = list(dict.fromkeys(must_have))[: self.K]
            if len(subset) < self.K:
                for s in self.all_switch_ids:
                    if s not in subset:
                        subset.append(int(s))
                    if len(subset) >= self.K:
                        break
            return subset

        dist = {}
        g = None
        fault_end_buses = set()
        try:
            import pandapower.topology as top
            g = top.create_nxgraph(self.net, include_switches=True)
        except Exception:
            pass
        if faulted_line is not None and faulted_line in self.net.line.index:
            row = self.net.line.loc[faulted_line]
            fault_end_buses = {int(row.from_bus), int(row.to_bus)}
        if g is not None and fault_end_buses:
            import networkx as nx
            sp_maps = {b: nx.single_source_shortest_path_length(
                g, b, cutoff=int(self.config.get("near_fault_hops", 3))) for b in fault_end_buses}
            for sw in self.all_switch_ids:
                try:
                    bus = int(self.net.switch.at[sw, "bus"])
                    dist[sw] = min((sp_maps[b].get(bus, 1e9) for b in fault_end_buses))
                except Exception:
                    dist[sw] = 1e9
        else:
            for sw in self.all_switch_ids:
                dist[sw] = 9999

        is_tie = (self.net.switch["et"] == "l") & (~self.net.switch["closed"].astype(bool))
        tie_ids = [int(sw) for sw in self.all_switch_ids if int(sw) in self.net.switch.index and bool(is_tie.get(sw, False))]
        non_tie_ids = [int(sw) for sw in self.all_switch_ids if sw not in tie_ids]

        tie_sorted = sorted(tie_ids, key=lambda s: dist.get(s, 1e9))
        non_tie_sorted = sorted(non_tie_ids, key=lambda s: dist.get(s, 1e9))

        seen, subset = set(), []
        for s in (must_have + tie_sorted + non_tie_sorted):
            if s not in seen:
                subset.append(s); seen.add(s)
            if len(subset) >= self.K:
                break
        if len(subset) < self.K:
            for s in self.all_switch_ids:
                if s not in seen:
                    subset.append(int(s)); seen.add(int(s))
                if len(subset) >= self.K:
                    break
        return subset

    # ------------------------- cascade helpers -------------------------- #
    def _fault_cluster_buses(self) -> set:
        buses = set()
        if not self._active_faults:
            return buses
        for ln in self._active_faults:
            if ln in self.net.line.index:
                row = self.net.line.loc[ln]
                buses.add(int(row.from_bus)); buses.add(int(row.to_bus))
        return buses

    def _line_neighbors_within_hops(self, hop_limit: int, sample_k: int = 0) -> List[int]:
        import pandapower.topology as top
        if hop_limit <= 0 or not len(self.net.line):
            return [int(i) for i in self.net.line.index if bool(self.net.line.at[i, "in_service"]) and (i not in self._active_faults)]

        src_buses = self._fault_cluster_buses()
        if not src_buses:
            return [int(i) for i in self.net.line.index if bool(self.net.line.at[i, "in_service"]) and (i not in self._active_faults)]

        try:
            g = top.create_nxgraph(self.net, include_switches=True)
        except Exception:
            return [int(i) for i in self.net.line.index if bool(self.net.line.at[i, "in_service"]) and (i not in self._active_faults)]

        import networkx as nx
        reachable = set()
        for b in src_buses:
            try:
                sp = nx.single_source_shortest_path_length(g, b, cutoff=hop_limit)
                reachable.update(int(x) for x in sp.keys())
            except Exception:
                pass

        cands = []
        for i, row in self.net.line.iterrows():
            if not bool(row.in_service):
                continue
            if int(i) in self._active_faults:
                continue
            if (int(row.from_bus) in reachable) or (int(row.to_bus) in reachable):
                cands.append(int(i))

        if sample_k and len(cands) > sample_k:
            rng = np.random.default_rng()
            cands = list(rng.choice(cands, size=sample_k, replace=False))
        return cands

    def _trip_lines(self, line_ids: List[int], as_live_fault: bool, info: dict):
        tripped = []
        for ln in line_ids:
            if ln not in self.net.line.index or ln in self._active_faults:
                continue
            if as_live_fault:
                self.faulted_line = int(ln)
                self._fault_live = True
                self._fault_live_steps = 0
                self.switch_ids = self._select_switch_subset(self.faulted_line)
            else:
                self.net.line.at[int(ln), "in_service"] = False
            self._active_faults.add(int(ln))
            tripped.append(int(ln))
            self._overload_watch.pop(int(ln), None)

        if tripped:
            info.setdefault("cascade_tripped", []).extend(tripped)
            for tln in tripped:
                self._disturbance_log.append({
                    "t": self.current_step, "type": "cascade_trip",
                    "line": int(tln), "as_live": bool(as_live_fault)
                })
            self.log.debug("[CASCADE] Trip %s (as_live=%s); budget->%d",
                           tripped, as_live_fault, self._cascade_remaining - len(tripped))

    def _log_overload_watch(self, cand: List[int], load_series, min_pct, hold, drop):
        # Small helper to print watch counters & loadings
        rows = []
        for ln in cand:
            try:
                lpct = float(load_series.at[ln])
            except Exception:
                continue
            cnt = self._overload_watch.get(ln, 0)
            rows.append((ln, lpct, cnt))
        rows.sort(key=lambda t: -t[1])
        msg = ", ".join([f"L{ln}:{lpct:.1f}% (hold={cnt})" for ln, lpct, cnt in rows[:10]])
        self.log.debug("[CASCADE] Watch(min=%.1f, hold=%d, drop=%.1f): %s", min_pct, hold, drop, msg or "—")

    def _cascade_overload_step(self, pf_ok: bool, info: dict):
        cfg = self._c_rt if self._c_rt is not None else self._cascade_cfg()
        if cfg["strategy"] != "overload" or cfg["max_additional"] <= 0:
            return
        if not pf_ok:
            return
        if not hasattr(self.net, "res_line") or not len(self.net.res_line):
            return
        if self._cascade_remaining <= 0:
            return

        hop_limit = cfg["hop_limit"]
        cand = self._line_neighbors_within_hops(hop_limit, sample_k=cfg["neighbor_sample_k"])
        if not cand:
            return

        load = self.net.res_line.loading_percent
        min_pct = cfg["overload_min_pct"]
        hold = cfg["min_hold_steps"]
        drop = cfg["hysteresis_drop"]
        live_faults = cfg["live_faults"]
        per_wave_max = min(cfg["per_wave_max"], self._cascade_remaining)

        # Update and log overload watch counters
        eligible = []
        for ln in cand:
            try:
                lpct = float(load.at[ln])
            except Exception:
                continue
            if lpct >= (min_pct - 1e-6):
                self._overload_watch[ln] = self._overload_watch.get(ln, 0) + 1
            elif lpct <= (min_pct - drop):
                self._overload_watch.pop(ln, None)
            if self._overload_watch.get(ln, 0) >= hold and lpct >= min_pct:
                eligible.append((ln, lpct))
        self._log_overload_watch(cand, load, min_pct, hold, drop)

        if not eligible:
            return

        self.log.debug("[CASCADE] Eligible after hold≥%d: %s",
                       hold, [(ln, round(lpct, 1)) for ln, lpct in sorted(eligible, key=lambda t: -t[1])[:10]])

        pl = sorted((float(a), float(b) * float(cfg.get("prob_scale", 1.0)))
                    for a, b in cfg["prob_loading_map"])
        def prob_for_loading(x: float) -> float:
            p = 0.0
            for thr, val in pl:
                if x >= thr:
                    p = val
            return float(np.clip(p, 0.0, 1.0))

        rng = np.random.default_rng()
        chosen = []
        for ln, lpct in sorted(eligible, key=lambda t: -t[1]):
            if len(chosen) >= per_wave_max:
                break
            pr = prob_for_loading(lpct)
            r = rng.random()
            self.log.debug("[CASCADE] Consider L%s load=%.1f%% -> p=%.2f rand=%.2f", ln, lpct, pr, r)
            if r < pr:
                chosen.append(int(ln))

        if not chosen:
            self.log.debug("[CASCADE] No lines chosen this step (per_wave_max=%d)", per_wave_max)
            return

        if len(chosen) > self._cascade_remaining:
            chosen = chosen[: self._cascade_remaining]

        self._trip_lines(chosen, as_live_fault=live_faults, info=info)
        self._cascade_remaining -= len(chosen)
        self.log.debug("[CASCADE] Chosen=%s, live_faults=%s, remaining=%d",
                       chosen, live_faults, self._cascade_remaining)


    def _ensure_info_keys(self, info: dict) -> dict:
        """Guarantee all Monitor info_keywords are present with safe values."""
        try:
            tot, crit, imp = self._served_fractions()
        except Exception:
            tot = crit = imp = 0.0

        info.setdefault("served_total_frac", float(tot))
        info.setdefault("served_crit_frac",  float(crit))
        info.setdefault("served_imp_frac",   float(imp))

        info.setdefault("fault_live",        bool(self._fault_live))
        info.setdefault("isolation_happened", False)
        info.setdefault("powerflow_success", False)

        info.setdefault("cascade_remaining", int(getattr(self, "_cascade_remaining", 0)))
        info.setdefault("der_bound_frac",    float(getattr(self, "_der_bound_frac_ep", 1.0)))

        # MW-level convenience
        try:
            mask = self._served_mask_by_connectivity(self.net)
            served_mw = float(self.net.load.loc[mask, "p_mw"].sum()) if len(self.net.load) else 0.0
            base_mw   = float(self.net.load["p_base_mw"].sum()) if len(self.net.load) and "p_base_mw" in self.net.load else served_mw
        except Exception:
            served_mw = base_mw = 0.0
        info.setdefault("served_mw", served_mw)
        info.setdefault("der_bound_mw", base_mw * float(getattr(self, "_der_bound_frac_ep", 1.0)))

        return info


    # ------------------------------- step ------------------------------- #
    def step(self, action):
        try:
            self.prev = copy.deepcopy(self.net)
            self.current_step += 1
            info = {"step": self.current_step, "faulted_line": self.faulted_line}
            ok = False

            a = int(np.asarray(action).item() if isinstance(action, (np.ndarray, list)) else int(action))
            if a < 0 or a >= self.action_space.n:
                a = 0

            K, L = self.K, self.L
            local_mask_pen = 0.0

            # Decode action for logging
            act_txt = "noop"
            if 1 <= a <= K:
                act_txt = f"toggle_switch[{self.switch_ids[a-1] if self.switch_ids else 'NA'}]"
            elif a > K:
                if self.enable_setpoint_shedding and L > 0:
                    rel = a - (1 + K)
                    if 0 <= rel < 2 * L:
                        lid = self.load_ids[rel // 2]
                        direction = "down" if (rel % 2 == 0) else "up"
                        act_txt = f"shed_{direction}[load {lid}]"
            self.log.debug("[STEP %d] action=%s fault_live=%s faulty_line=%s",
                           self.current_step, act_txt, self._fault_live, self.faulted_line)

            # --------------- Grace path ---------------
            if self._needs_grace and self._grace_budget > 0:
                info["emergency"] = 1
                info["grace_left_before"] = int(self._grace_budget)

                if a == 0:
                    info["noop"] = True
                elif 1 <= a <= K:
                    sw = self.switch_ids[a - 1]
                    if self._fault_live:
                        on_fault = set(self._on_fault_switch_ids())
                        if len(on_fault) and sw not in on_fault:
                            info["masked_switch_away_from_fault"] = int(sw)
                            local_mask_pen = float(self.invalid_live_action_penalty)
                            self.log.debug("[STEP] Masked switch %s (away from fault)", sw)
                        else:
                            curr = bool(self.net.switch.at[sw, "closed"])
                            new = not curr
                            self.net.switch.at[sw, "closed"] = new
                            info["toggled_switch"] = int(sw)
                            self.log.debug("[STEP] toggled switch %s -> %s (fault_live=%s)",
                                           sw, "closed" if new else "open", self._fault_live)
                            if self.faulted_line is not None:
                                row = self.net.switch.loc[sw]
                                if (row.et == "l") and (int(row.element) == int(self.faulted_line)) and (new is False):
                                    self.net.line.at[int(self.faulted_line), "in_service"] = False
                                    self._fault_live = False
                                    info["isolation_happened"] = True
                                    self._fault_award_pending = True
                                    self._fault_isolated_at = self.current_step
                                    self.switch_ids = self._select_switch_subset(self.faulted_line)
                                    self.log.debug("[ISOLATION] Fault on line %s isolated at step %d by switch %s",
                                                   self.faulted_line, self.current_step, sw)
                    else:
                        curr = bool(self.net.switch.at[sw, "closed"])
                        self.net.switch.at[sw, "closed"] = (not curr)
                        info["toggled_switch"] = int(sw)
                        self.log.debug("[STEP] toggled switch %s -> %s", sw, "open" if curr else "closed")
                else:
                    if self.enable_setpoint_shedding and L > 0:
                        rel = a - (1 + K)
                        if 0 <= rel < 2 * L:
                            load_idx = rel // 2
                            direction = +1 if (rel % 2 == 0) else -1
                            lid = self.load_ids[load_idx]
                            if self._fault_live and self.mask_load_actions_during_live_fault:
                                info["masked_load_action"] = int(lid)
                                local_mask_pen = float(self.invalid_live_action_penalty)
                                self.log.debug("[STEP] Masked shedding on load %s (fault live)", lid)
                            else:
                                if "shed_frac" not in self.net.load.columns:
                                    self.net.load["shed_frac"] = 0.0
                                if "p_base_mw" not in self.net.load.columns:
                                    self.net.load["p_base_mw"] = self.net.load["p_mw"].astype(float)
                                if "q_mvar" in self.net.load.columns and "q_base_mvar" not in self.net.load.columns:
                                    self.net.load["q_base_mvar"] = self.net.load["q_mvar"].astype(float)
                                step_sz = float(self.config.get("shed_step", 0.1))
                                sf = float(self.net.load.at[lid, "shed_frac"])
                                sf = float(np.clip(sf + direction * step_sz, 0.0, 1.0))
                                self.net.load.at[lid, "shed_frac"] = sf
                                p0 = float(self.net.load.at[lid, "p_base_mw"])
                                self.net.load.at[lid, "p_mw"] = p0 * (1.0 - sf)
                                if "q_mvar" in self.net.load.columns:
                                    q0 = float(self.net.load.at[lid, "q_base_mvar"])
                                    self.net.load.at[lid, "q_mvar"] = q0 * (1.0 - sf)
                                info["shedding_load"] = int(lid)
                                self.log.debug("[STEP] Shedding load %s to frac=%.2f", lid, sf)
                    else:
                        info["noop"] = True

                ok = self._runpf()
                self._grace_budget -= 1
                info["grace_left_after"] = int(self._grace_budget)

                if ok:
                    self._cascade_overload_step(pf_ok=True, info=info)
                    info["cascade_remaining"] = int(self._cascade_remaining)

                if self._fault_live:
                    self._fault_live_steps += 1
                else:
                    self._fault_live_steps = 0

                _, comps = self._reward()
                if local_mask_pen:
                    comps["invalid_live_action"] = comps.get("invalid_live_action", 0.0) + float(local_mask_pen)
                if not ok:
                    if "pf_fail" in comps:
                        comps["pf_fail"] = 0.0
                    self._needs_grace = (self._grace_budget > 0)
                else:
                    self.pf_fail_streak = 0
                    self._needs_grace = False

                self._log_step_summary(ok, comps, info)
                done = (
                    (self.current_step >= self.max_steps)
                    or (self._fault_live and (self._fault_live_steps >= int(self.protection_trip_delay)))
                    or ((not ok) and (self._grace_budget == 0))
                )
                info.update({"reward_components": comps})
                if done and self._fault_live and (self._fault_live_steps >= int(self.protection_trip_delay)):
                    self.log.debug("[PROTECTION] Auto-trip: fault on line %s persisted %d steps",
                                   self.faulted_line, self._fault_live_steps)
                info["powerflow_success"] = bool(ok) 
                info = self._ensure_info_keys(info)
                return self._obs(), float(sum(comps.values())), bool(done), False, info

            # --------------- Normal path ---------------
            if a == 0:
                info["noop"] = True
            elif 1 <= a <= K:
                sw = self.switch_ids[a - 1]
                if self._fault_live:
                    on_fault = set(self._on_fault_switch_ids())
                    if len(on_fault) and sw not in on_fault:
                        info["masked_switch_away_from_fault"] = int(sw)
                        local_mask_pen = float(self.invalid_live_action_penalty)
                        self.log.debug("[STEP] Masked switch %s (away from fault)", sw)
                    else:
                        curr = bool(self.net.switch.at[sw, "closed"])
                        new = not curr
                        self.net.switch.at[sw, "closed"] = new
                        info["toggled_switch"] = int(sw)
                        if self.faulted_line is not None:
                            row = self.net.switch.loc[sw]
                            if (row.et == "l") and (int(row.element) == int(self.faulted_line)) and (new is False):
                                self.net.line.at[int(self.faulted_line), "in_service"] = False
                                self._fault_live = False
                                info["isolation_happened"] = True
                                self._fault_award_pending = True
                                self._fault_isolated_at = self.current_step
                                self.switch_ids = self._select_switch_subset(self.faulted_line)
                                self.log.debug("[ISOLATION] Fault on line %s isolated at step %d by switch %s",
                                               self.faulted_line, self.current_step, sw)
                else:
                    curr = bool(self.net.switch.at[sw, "closed"])
                    self.net.switch.at[sw, "closed"] = (not curr)
                    info["toggled_switch"] = int(sw)
                    self.log.debug("[STEP] toggled switch %s -> %s", sw, "open" if curr else "closed")
            else:
                if self.enable_setpoint_shedding and L > 0:
                    rel = a - (1 + K)
                    if 0 <= rel < 2 * L:
                        load_idx = rel // 2
                        direction = +1 if (rel % 2 == 0) else -1
                        lid = self.load_ids[load_idx]
                        if self._fault_live and self.mask_load_actions_during_live_fault:
                            info["masked_load_action"] = int(lid)
                            local_mask_pen = float(self.invalid_live_action_penalty)
                            self.log.debug("[STEP] Masked shedding on load %s (fault live)", lid)
                        else:
                            if "shed_frac" not in self.net.load.columns:
                                self.net.load["shed_frac"] = 0.0
                            if "p_base_mw" not in self.net.load.columns:
                                self.net.load["p_base_mw"] = self.net.load["p_mw"].astype(float)
                            if "q_mvar" in self.net.load.columns and "q_base_mvar" not in self.net.load.columns:
                                self.net.load["q_base_mvar"] = self.net.load["q_mvar"].astype(float)
                            step_sz = float(self.config.get("shed_step", 0.1))
                            sf = float(self.net.load.at[lid, "shed_frac"])
                            sf = float(np.clip(sf + direction * step_sz, 0.0, 1.0))
                            self.net.load.at[lid, "shed_frac"] = sf
                            p0 = float(self.net.load.at[lid, "p_base_mw"])
                            self.net.load.at[lid, "p_mw"] = p0 * (1.0 - sf)
                            if "q_mvar" in self.net.load.columns:
                                q0 = float(self.net.load.at[lid, "q_base_mvar"])
                                self.net.load.at[lid, "q_mvar"] = q0 * (1.0 - sf)
                            info["shedding_load"] = int(lid)
                            self.log.debug("[STEP] Shedding load %s to frac=%.2f", lid, sf)
                else:
                    info["noop"] = True

            ok = self._runpf()

            if ok:
                self._cascade_overload_step(pf_ok=True, info=info)
                info["cascade_remaining"] = int(self._cascade_remaining)

            if not ok and not self._fault_live and (self._grace_budget == 0) and (self._grace_budget_max > 0):
                self._needs_grace = True
                self._grace_budget = self._grace_budget_max
                self.log.debug("[STEP] PF infeasible -> starting grace window (%d steps)", self._grace_budget)

            if self._fault_live:
                self._fault_live_steps += 1
            else:
                self._fault_live_steps = 0

            _, comps = self._reward()
            if local_mask_pen:
                comps["invalid_live_action"] = comps.get("invalid_live_action", 0.0) + float(local_mask_pen)

            if not ok:
                if not self._fault_live:
                    self.pf_fail_streak += 1
                    comps["pf_fail"] = float(self.pf_failure_penalty)
            else:
                self.pf_fail_streak = 0

            # step summary & termination
            self._log_step_summary(ok, comps, info)
            done = (
                (self.current_step >= self.max_steps)
                or (self.pf_fail_streak >= self.failure_patience)
                or (self._fault_live and (self._fault_live_steps >= int(self.protection_trip_delay)))
            )

            if done and self._fault_live and (self._fault_live_steps >= int(self.protection_trip_delay)):
                self.log.debug("[PROTECTION] Auto-trip: fault on line %s persisted %d steps",
                               self.faulted_line, self._fault_live_steps)

            info.update({"reward_components": comps,
                         "grace_left": int(self._grace_budget),
                         "needs_grace": bool(self._needs_grace)})
            info["powerflow_success"] = bool(ok) 
            info = self._ensure_info_keys(info)
            return self._obs(), float(sum(comps.values())), bool(done), False, info

        except Exception as e:
            self.log.error("Error in step(): %s", e, exc_info=True)
            comps = {"pf_fail": float(self.pf_failure_penalty)}
            info = {
                "error": str(e),
                "reward_components": comps,
                "served_total_frac": 0.0,
                "served_crit_frac": 0.0,
                "served_imp_frac": 0.0,
                "fault_live": bool(self._fault_live),
                "cascade_remaining": int(self._cascade_remaining),
                "isolation_happened": False,
                "powerflow_success": False,
                "der_bound_frac": float(getattr(self, "_der_bound_frac_ep", 1.0)),
                "served_mw": 0.0,
                "der_bound_mw": 0.0,
                "disturbance_log": list(self._disturbance_log[-10:]) if hasattr(self, "_disturbance_log") else [],
            }

            return self._obs(), float(sum(comps.values())), True, False, info

    # ---------------------- observation / reward / PF ------------------- #
    def _source_buses_for(self, net) -> set[int]:
        src = set()
        if hasattr(net, "ext_grid") and len(net.ext_grid):
            try:
                df = net.ext_grid
                m = df.get("in_service", True)
                if isinstance(m, bool):
                    m = np.full(len(df), m, dtype=bool)
                src |= set(df.loc[m, "bus"].astype(int).tolist())
            except Exception:
                pass
        for tb in ("gen", "sgen"):
            df = getattr(net, tb, None)
            if df is not None and len(df):
                try:
                    m = df.get("in_service", True)
                    if isinstance(m, bool):
                        m = np.full(len(df), m, dtype=bool)
                    src |= set(df.loc[m, "bus"].astype(int).tolist())
                except Exception:
                    pass
        if not src and len(net.bus):
            try:
                if "in_service" in net.bus:
                    first = int(net.bus.index[net.bus.in_service.astype(bool)].min())
                else:
                    first = int(net.bus.index.min())
                src = {first}
            except Exception:
                pass
        return src

    def _served_mask_by_connectivity(self, net) -> np.ndarray:
        import pandapower.topology as top
        if not len(net.load):
            return np.zeros(0, dtype=bool)
        g = top.create_nxgraph(
            net, respect_switches=True,
            include_lines=True, include_trafos=True,
            include_impedances=True, include_out_of_service=False,
            include_switches=False,
        )
        slacks = list(self._source_buses_for(net))
        try:
            unsup = set(top.unsupplied_buses(net, slacks=slacks, respect_switches=True))
        except TypeError:
            unsup = set(net.bus.index.tolist())
            for s in slacks:
                try:
                    comp = set(top.connected_component(g, s))
                    unsup -= comp
                except Exception:
                    pass
        load_buses = net.load.bus.astype(int).values
        mask = np.array([int(b) not in unsup for b in load_buses], dtype=bool)
        if not mask.any():
            mask = net.load.in_service.astype(bool).values if "in_service" in net.load else np.ones(len(net.load), dtype=bool)
        return mask

    def _served_fractions(self) -> Tuple[float, float, float]:
        if not len(self.net.load):
            return 1.0, 1.0, 1.0
        if "p_base_mw" not in self.net.load.columns:
            self.net.load["p_base_mw"] = self.net.load["p_mw"].astype(float)

        base = self.net.load["p_base_mw"].astype(float)
        cur  = self.net.load["p_mw"].astype(float)
        prio = self.net.load.get("priority", 0)

        total = float(base.sum()) if len(base) else 0.0
        mask = self._served_mask_by_connectivity(self.net)

        served_total = float(cur[mask].sum()) if total > 1e-9 else 0.0
        crit_tot = float(base[prio == 2].sum()); crit_srv = float(cur[(prio == 2) & mask].sum()) if crit_tot > 1e-9 else 0.0
        imp_tot  = float(base[prio == 1].sum()); imp_srv  = float(cur[(prio == 1) & mask].sum()) if imp_tot  > 1e-9 else 0.0

        return (
            (served_total / total) if total > 1e-9 else 1.0,
            (crit_srv / crit_tot) if crit_tot > 1e-9 else 1.0,
            (imp_srv  / imp_tot)  if imp_tot  > 1e-9 else 1.0,
        )

    def _served_fractions_for(self, net) -> Tuple[float, float, float]:
        if not len(net.load):
            return 1.0, 1.0, 1.0
        if "p_base_mw" not in net.load.columns:
            net.load["p_base_mw"] = net.load["p_mw"].astype(float)

        base = net.load["p_base_mw"].astype(float)
        cur  = net.load["p_mw"].astype(float)
        prio = net.load.get("priority", 0)

        total = float(base.sum()) if len(base) else 0.0
        mask = self._served_mask_by_connectivity(net)

        served_total = float(cur[mask].sum()) if total > 1e-9 else 0.0
        crit_tot = float(base[prio == 2].sum()); crit_srv = float(cur[(prio == 2) & mask].sum()) if crit_tot > 1e-9 else 0.0
        imp_tot  = float(base[prio == 1].sum()); imp_srv  = float(cur[(prio == 1) & mask].sum()) if imp_tot  > 1e-9 else 0.0

        return (
            (served_total / total) if total > 1e-9 else 1.0,
            (crit_srv / crit_tot) if crit_tot > 1e-9 else 1.0,
            (imp_srv  / imp_tot)  if imp_tot  > 1e-9 else 1.0,
        )

    def _reward(self) -> Tuple[float, dict]:
        w = self.config.get("reward_weights", {})
        keep_c = float(w.get("keep_crit", 8.0))
        keep_i = float(w.get("keep_imp", 3.0))
        v_pen  = float(w.get("voltage", -2.0))
        shed_w = float(w.get("shed", -1.0))
        sw_w   = float(w.get("switch", -0.01))
        iso_w  = float(w.get("fault_isolated", 300.0))
        live_w = float(w.get("fault_live", -80.0))
        v_delta_w = float(w.get("volt_delta", 0.5))
        rest_w = float(w.get("restore_delta", 5.0))
        open_nf_cost = float(w.get("line_open_nonfault_live", 0.0))
        alive_w = float(w.get("alive", 0.12))

        comps = {"crit": 0.0, "imp": 0.0, "volt": 0.0, "shed": 0.0, "switch": 0.0,
                 "fault": 0.0, "fault_live": 0.0}

        if len(self.net.load):
            if "p_base_mw" not in self.net.load.columns:
                self.net.load["p_base_mw"] = self.net.load["p_mw"].astype(float)

            base = self.net.load["p_base_mw"].astype(float)
            cur  = self.net.load["p_mw"].astype(float)
            prio = self.net.load.get("priority", 0)

            p_tot = float(base.sum())
            mask = self._served_mask_by_connectivity(self.net)

            p_srv = float(cur[mask].sum()) if p_tot > 1e-9 else 0.0
            p2_tot = float(base[prio == 2].sum()); p2_srv = float(cur[(prio == 2) & mask].sum()) if p2_tot > 1e-9 else 0.0
            p1_tot = float(base[prio == 1].sum()); p1_srv = float(cur[(prio == 1) & mask].sum()) if p1_tot > 1e-9 else 0.0

            if p2_tot > 1e-9:
                comps["crit"] = keep_c * (p2_srv / p2_tot)
            if p1_tot > 1e-9:
                comps["imp"]  = keep_i * (p1_srv / p1_tot)

            # alive shaping
            if p_tot > 1e-9 and alive_w != 0.0:
                comps["alive"] = alive_w * (p_srv / p_tot)

            if not self._fault_live and p_tot > 1e-9:
                curtail_frac = float(1.0 - (p_srv / p_tot))
                comps["shed"] = shed_w * curtail_frac

        if hasattr(self.net, "res_bus") and len(self.net.res_bus):
            v = np.nan_to_num(self.net.res_bus.vm_pu.values, nan=0.0, posinf=2.0, neginf=0.0)
            vmin, vmax = self.v_limits
            dev = np.clip(vmin - v, 0, None) + np.clip(v - vmax, 0, None)
            comps["volt"] = v_pen * float(np.sum(dev))
            try:
                v_now = np.nan_to_num(self.net.res_bus.vm_pu.values, nan=1.0)
                v_prev = np.nan_to_num(self.prev.res_bus.vm_pu.values, nan=1.0) if hasattr(self.prev, "res_bus") and len(self.prev.res_bus) else v_now
                dev_now  = np.clip(self.v_limits[0] - v_now, 0, None) + np.clip(v_now - self.v_limits[1], 0, None)
                dev_prev = np.clip(self.v_limits[0] - v_prev, 0, None) + np.clip(v_prev - self.v_limits[1], 0, None)
                comps["volt"] += v_delta_w * float(np.sum(dev_prev) - np.sum(dev_now))
            except Exception:
                pass

        if len(self.net.switch) and len(self.prev.switch):
            prev_s = self.prev.switch["closed"].astype(int).values
            curr_s = self.net.switch["closed"].astype(int).values
            comps["switch"] = sw_w * int(np.sum(np.abs(prev_s - curr_s)))

        if self._fault_live:
            comps["fault_live"] = live_w

        if self._fault_award_pending:
            comps["fault"] = iso_w
            self._fault_award_pending = False
            self._fault_awarded = True

        if rest_w != 0.0 and (not self._fault_live):
            now_tot, _, _ = self._served_fractions()
            prev_tot, _, _ = self._served_fractions_for(self.prev) if hasattr(self, "prev") else (now_tot, 0, 0)
            comps["restore_delta"] = rest_w * float(now_tot - prev_tot)

        if not self._fault_live and self.bound_pen_w != 0.0:
            now_tot, _, _ = self._served_fractions()
            bound = float(np.clip(self._der_bound_frac_ep, 0.0, 1.0))
            if now_tot > bound + 1e-6:
                comps["bound_violation"] = self.bound_pen_w * float(now_tot - bound)

        # penalize opening non-fault line switches while the fault is live
        if open_nf_cost != 0.0 and self._fault_live and len(self.net.switch) and len(self.prev.switch):
            prev = self.prev.switch["closed"].astype(bool)
            curr = self.net.switch["closed"].astype(bool)
            new_opens = (~curr & prev)
            if new_opens.any():
                idx = self.net.switch.index[new_opens]
                nf = self.net.switch.loc[idx]
                nf = nf[(nf["et"] == "l") &
                        (nf["element"].astype(int) != int(self.faulted_line if self.faulted_line is not None else -1))]
                if len(nf):
                    comps["nonfault_live_open"] = open_nf_cost * float(len(nf))

        total = float(sum(comps.values()))
        return total, comps

    def _obs(self):
        n_buses = len(self.net.bus)
        v = np.ones(n_buses, dtype=np.float32)
        if hasattr(self.net, "res_bus") and len(self.net.res_bus):
            v = np.nan_to_num(self.net.res_bus.vm_pu.values.astype(np.float32),
                              nan=1.0, posinf=1.0, neginf=0.0)
        if len(self.switch_ids):
            sw = self.net.switch.loc[self.switch_ids, "closed"].astype(np.float32).values
        else:
            sw = np.zeros(0, dtype=np.float32)
        if self.enable_setpoint_shedding and len(self.load_ids):
            if "shed_frac" not in self.net.load.columns:
                self.net.load["shed_frac"] = 0.0
            sf = self.net.load.loc[self.load_ids, "shed_frac"].astype(np.float32).values
        else:
            sf = np.zeros(0, dtype=np.float32)
        return np.concatenate([v, sw, sf]).astype(np.float32)

    def _runpf(self) -> bool:
        if self._fault_live:
            self.log.debug("[PF] Skipped runpp because fault is live")
            return False
        try:
            pp.runpp(self.net, enforce_q_lims=True)
            self.log.debug("[PF] Success (NR+rqlims)")
            return True
        except Exception as e1:
            self.log.warning("Power flow failed (NR+rqlims): %s", e1)
        try:
            pp.runpp(self.net, algorithm="bfsw", enforce_q_lims=False, init="flat", tolerance_mva=1e-5)
            self.log.debug("[PF] Success (BFSW flat)")
            return True
        except Exception as e2:
            self.log.warning("Power flow failed (BFSW flat): %s", e2)
        try:
            pp.runpp(self.net, algorithm="bfsw", enforce_q_lims=False, init="dc", tolerance_mva=1e-5)
            self.log.debug("[PF] Success (BFSW dc)")
            return True
        except Exception as e3:
            self.log.warning("Power flow failed (BFSW dc): %s", e3)
            return False

    # ------------------------- logging helpers -------------------------- #
    def _log_step_summary(self, pf_ok: bool, comps: dict, info: dict):
        tot, crit, imp = info.get("served_total_frac"), info.get("served_crit_frac"), info.get("served_imp_frac")
        caserem = info.get("cascade_remaining", self._cascade_remaining)
        live = info.get("fault_live", self._fault_live)
        iso = info.get("isolation_happened", False)
        reward = float(sum(comps.values()))
        short = {k: round(v, 3) for k, v in comps.items() if abs(v) > 1e-6}
        self.log.debug(
            "[STEP %d] PF=%s live=%s iso=%s served(T/C/I)=%.2f/%.2f/%.2f reward=%.2f cascade_rem=%d comps=%s",
            self.current_step, pf_ok, live, iso, (tot or 0.0), (crit or 0.0), (imp or 0.0), reward, caserem, short
        )
