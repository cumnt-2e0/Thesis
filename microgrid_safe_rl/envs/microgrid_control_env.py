# microgrid_safe_rl/envs/microgrid_control_env.py
"""
MicrogridControlEnv — islanded microgrid switching control (no pruning)

Behavior summary
----------------
• The feeder is treated as an islanded microgrid.
• A disturbance may mark a line as FAULTED but still energized. The agent must OPEN a line
  switch on that line to isolate it. When it does:
    - that line is set out of service for the remainder of the episode,
    - a one-off isolation bonus is awarded,
    - the per-step live-fault penalty stops.
• While the fault is live, power flow is considered failed (we do not modify the net).
• After isolation, the agent can reconfigure (close ties) to restore supply to loads.
• No "prune islands": we never deactivate parts of the network for you; PF failures are
  penalized (except while the fault is live).

Action space (Discrete):
  0                -> no-op
  1..K             -> toggle one of K selected switches (subset rebuilt per episode)
  K+1..K+2L - 1    -> per-load curtailment +/- (only if enable_setpoint_shedding: true)

Observation: [bus voltages | K selected switch states | L load shed_frac]

Key config (env.yaml):
  voltage_limits: [0.95, 1.05]
  max_steps: 80
  pf_failure_patience: 3
  protection_trip_delay: 8
  reset_retries: 3
  max_switches: 20
  near_fault_hops: 2
  enable_setpoint_shedding: false
  mask_load_actions_during_live_fault: true
  shed_step: 0.1
  reward_weights:
    keep_crit: 8.0
    keep_imp:  3.0
    voltage:  -2.0
    volt_delta: 0.5
    shed:     -1.0        # FRACTION (0..1), applied only AFTER isolation
    switch:   -0.01
    pf_failure: -30.0
    fault_live: -80.0
    fault_isolated: 300.0
    invalid_live_action: -2.0
    restore_delta: 5.0    # reward increases in served_total after isolation
"""

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
        self.net0 = copy.deepcopy(net)  # pristine prepared template

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
        self.der_cfg = self.config.get("der_unavailability", {})  # outages + derating at reset
        self.bound_pen_w = float(self.config.get("reward_weights", {}).get("bound_violation", -20.0))
        self._der_bound_frac_ep = 1.0  # sampled per-episode “available supply” / base load


        # Full lists & fixed action sizes
        self.all_switch_ids = list(self.net.switch.index) if len(self.net.switch) else []
        self.load_ids = list(self.net.load.index) if len(self.net.load) else []

        # Fixed K (cap switches); if 0 or larger than available, use all
        K_cfg = int(self.config.get("max_switches", 0) or 0)
        self.K = K_cfg if (K_cfg > 0 and K_cfg <= len(self.all_switch_ids)) else len(self.all_switch_ids)

        # Effective L (only if we allow set-point curtailment)
        self.L = len(self.load_ids) if self.enable_setpoint_shedding else 0

        # Initially build a per-episode subset (rebuilt after we know the fault)
        self.switch_ids = self._select_switch_subset(faulted_line=None)

        # Spaces
        n_buses = len(self.net.bus)
        self.action_space = spaces.Discrete(1 + self.K + 2 * self.L)
        self.observation_space = spaces.Box(
            low=np.array([self.v_limits[0]] * n_buses + [0.0] * self.K + [0.0] * self.L, dtype=np.float32),
            high=np.array([self.v_limits[1]] * n_buses + [1.0] * self.K + [1.0] * self.L, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # Net preparation
    # ------------------------------------------------------------------ #
    def _prepare_net_inplace(self, net):
        """Ensure base columns & policy state columns exist on the given net."""
        if len(net.load):
            if "p_base_mw" not in net.load.columns:
                net.load["p_base_mw"] = net.load["p_mw"].astype(float)
            if "q_base_mvar" not in net.load.columns and "q_mvar" in net.load.columns:
                net.load["q_base_mvar"] = net.load["q_mvar"].astype(float)
            if "shed_frac" not in net.load.columns:
                net.load["shed_frac"] = 0.0
            if "priority" not in net.load.columns:
                assign_priorities(net)

        # DER base setpoints (for derating/outage math)
        for tb in ("gen", "sgen"):
            df = getattr(net, tb, None)
            if df is not None and len(df):
                if "p_base_mw" not in df.columns and "p_mw" in df.columns:
                    df["p_base_mw"] = df["p_mw"].astype(float)
                if "in_service" not in df.columns:
                    df["in_service"] = True

    # ------------------------------------------------------------------ #
    # Gym API: reset
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.pf_fail_streak = 0
        self.faulted_line = None
        self._fault_live = False
        self._fault_live_steps = 0
        self._fault_award_pending = False
        self._fault_awarded = False
        self._fault_isolated_at = None

        retries = int(self.config.get("reset_retries", 3))
        ok = False

        for t in range(max(1, retries)):
            # Restore pristine, prepared net
            self.net = copy.deepcopy(self.net0)
            self._prepare_net_inplace(self.net)
            self.prev = copy.deepcopy(self.net)

            # Disturbance
            if self.scenario.get("enabled", True):
                self._inject_disturbance(seed=None if seed is None else seed + t)

            # Apply DER outages/derating (static per episode, if configured)
            self._apply_der_unavailability(seed=None if seed is None else seed + 1337 + t)

            # Rebuild K-sized switch subset AFTER we know the fault
            self.switch_ids = self._select_switch_subset(self.faulted_line)

            # PF may be False while fault is live (by design)
            ok = self._runpf()
            if ok or self._fault_live:
                break

        return self._obs(), {
            "powerflow_success": ok,
            "faulted_line": self.faulted_line,
            "der_bound_frac": float(self._der_bound_frac_ep),
        }

    # ------------------------------------------------------------------ #
    # Disturbance helpers
    # ------------------------------------------------------------------ #
    def _safe_fault_line(self) -> Optional[int]:
        """
        Pick a non-bridge line (removing it would not disconnect the graph).
        Falls back to any in-service line if all are bridges or on error.
        """
        try:
            import networkx as nx
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
            if not safe:
                return None
            return int(np.random.default_rng().choice(safe))

        except Exception as e:
            self.log.debug(f"_safe_fault_line fallback due to: {e}")
            if len(self.net.line):
                candidates = [int(i) for i, r in self.net.line.iterrows() if bool(r.in_service)]
                return None if not candidates else int(np.random.default_rng().choice(candidates))
            return None

    def _pick_fault_line(self, rng, seed=None) -> Optional[int]:
        """Choose a line to mark as faulted; optionally avoid bridges for training stability."""
        if not len(self.net.line):
            return None
        p = float(self.scenario.get("exclude_bridges_prob", 0.0))
        excl = bool(self.scenario.get("exclude_bridges", False) or (rng.random() < p))
        if excl:
            return self._safe_fault_line()
        return int(np.random.default_rng(seed).choice(self.net.line.index))

    def _inject_disturbance(self, seed=None):
        rng = np.random.default_rng(seed)
        modes = self.scenario.get("modes", ["line", "load"])
        sev = float(self.scenario.get("severity", 0.8))
        mode = rng.choice(modes) if modes else None

        if mode == "line" and len(self.net.line):
            li = self._pick_fault_line(rng, seed)
            if li is not None:
                # LIVE FAULT: keep the line in service; agent must open a switch on it
                self.faulted_line = int(li)
                self._fault_live = True
                self._fault_live_steps = 0
                # Ensure on-fault switches are in the subset ASAP
                self.switch_ids = self._select_switch_subset(self.faulted_line)
            return

        if mode == "load" and len(self.net.load):
            # optional random load surge (independent of topology shedding)
            li = int(rng.choice(self.net.load.index))
            f = float(np.clip(1.0 + 0.2 * sev + rng.uniform(0, 0.3 * sev), 1.0, 2.0))
            self.net.load.at[li, "p_mw"] = float(self.net.load.at[li, "p_mw"]) * f
            if "q_mvar" in self.net.load.columns:
                self.net.load.at[li, "q_mvar"] = float(self.net.load.at[li, "q_mvar"]) * f

    # ------ DER unavailability helpers ------
    def _sample_der_bound_frac(self, rng) -> float:
        cfg = self.der_cfg or {}
        lo, hi = cfg.get("p_deficit_frac", [0.0, 0.0])
        lo = float(lo); hi = float(hi)
        hi = max(hi, lo)
        deficit = rng.uniform(lo, hi)
        # fraction of base load that is *available* to serve this episode
        return float(np.clip(1.0 - deficit, 0.0, 1.0))

    def _apply_der_unavailability(self, seed=None):
        """Optionally knock out and derate DERs to create a real supply shortfall.
        Also store a per-episode DER bound fraction for reward shaping/metrics.
        """
        if not self.der_cfg or not self.der_cfg.get("static_on_reset", False):
            # even if we don't modify the net, still produce a bound for shaping if user configured p_deficit_frac
            rng = np.random.default_rng(seed)
            self._der_bound_frac_ep = self._sample_der_bound_frac(rng)
            return

        rng = np.random.default_rng(seed)
        cfg = self.der_cfg
        self._der_bound_frac_ep = self._sample_der_bound_frac(rng)

        # Outage probability and scaling range for *remaining* DERs
        outage_p = float(cfg.get("random_outage_prob", 0.0))
        s_lo, s_hi = cfg.get("scaling_range", [0.5, 1.0])
        s_lo = float(s_lo); s_hi = float(max(s_hi, s_lo))

        # Base total load (for a target)
        if len(self.net.load):
            if "p_base_mw" not in self.net.load.columns:
                self.net.load["p_base_mw"] = self.net.load["p_mw"].astype(float)
            P_L = float(self.net.load["p_base_mw"].sum())
        else:
            P_L = 0.0

        # Target available power for the episode (soft target)
        P_target = self._der_bound_frac_ep * P_L

        # Build a list of DER tables to manipulate
        der_tables = [tb for tb in ("gen", "sgen") if hasattr(self.net, tb) and len(getattr(self.net, tb))]
        if not der_tables:
            return  # nothing to do; bound will still shape behavior

        # 1) Random outages
        disable_frac = float(cfg.get("disable_frac", 0.0))
        if disable_frac > 0 and len(der_tables):
            for tb in der_tables:
                df = getattr(self.net, tb)
                n_disable = int(np.floor(len(df) * disable_frac))
                if n_disable > 0:
                    to_disable = rng.choice(df.index, size=n_disable, replace=False)
                    df.loc[to_disable, "in_service"] = False
        
        # 2) Derate remaining DERs with a common scale * random jitter in [s_lo, s_hi]
        #    Try to aim the sum back to P_target (rough, but good enough to stress the agent).
        #    Use cached p_base_mw as the nominal capability.
        bases = []
        for tb in der_tables:
            df = getattr(self.net, tb)
            m = df.in_service.astype(bool)
            if "p_base_mw" in df.columns:
                bases.append(float(df.loc[m, "p_base_mw"].sum()))
            else:
                bases.append(float(df.loc[m, "p_mw"].sum()))
        P_base_avail = float(sum(bases))

        if P_base_avail > 1e-9 and len(der_tables):
            common_scale = float(np.clip(P_target / P_base_avail, s_lo, s_hi))
            for tb in der_tables:
                df = getattr(self.net, tb)
                m = df.in_service.astype(bool)
                if "p_base_mw" in df.columns:
                    jitter = rng.uniform(s_lo, s_hi, size=int(m.sum()))
                    df.loc[m, "p_mw"] = df.loc[m, "p_base_mw"].values * (common_scale * jitter)
                else:
                    jitter = rng.uniform(s_lo, s_hi, size=int(m.sum()))
                    df.loc[m, "p_mw"] = df.loc[m, "p_mw"].values * (common_scale * jitter)

        # Note: ext_grid (slack) may still cover any mismatch; the *bound penalty* (below)
        # forces the agent to shed if it serves above the bound.

    # ------------------------------------------------------------------ #
    # Switch subset selection (per-episode)
    # ------------------------------------------------------------------ #
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
        """
        Return a list of exactly self.K switch ids, prioritizing:
          - switches on the faulted line (must include, especially while live),
          - switches near the fault (by graph distance),
          - then remaining switches to fill to K.
        """
        if not len(self.all_switch_ids) or self.K == 0:
            return []

        # MUST-HAVE: switches physically on the faulted line
        must_have = []
        if faulted_line is not None and len(self.net.switch):
            try:
                on_fault = self.net.switch.index[
                    (self.net.switch["et"] == "l") & (self.net.switch["element"] == int(faulted_line))
                ].tolist()
                must_have = [int(s) for s in on_fault]
            except Exception:
                must_have = []

        # If a fault is live, restrict the subset to on-fault first (focus exploration)
        if self._fault_live and must_have:
            subset = list(dict.fromkeys(must_have))[: self.K]
            if len(subset) < self.K:
                for s in self.all_switch_ids:
                    if s not in subset:
                        subset.append(int(s))
                    if len(subset) >= self.K:
                        break
            return subset

        # Otherwise: prioritize by distance to fault endpoints (ties first)
        dist = {}
        g = None
        fault_end_buses = set()
        try:
            import networkx as nx
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

    # ------------------------------------------------------------------ #
    # Gym API: step
    # ------------------------------------------------------------------ #
    def step(self, action):
        try:
            # Keep a feasible rollback snapshot
            self.prev = copy.deepcopy(self.net)
            self.current_step += 1
            info = {"step": self.current_step, "faulted_line": self.faulted_line}

            # Parse & clip action
            a = int(np.asarray(action).item() if isinstance(action, (np.ndarray, list)) else int(action))
            if a < 0 or a >= self.action_space.n:
                a = 0

            K, L = self.K, self.L
            local_mask_pen = 0.0

            # Apply action
            if a == 0:
                info["noop"] = True

            elif 1 <= a <= K:
                sw = self.switch_ids[a - 1]

                # While fault is live: only allow on-fault switches to toggle; mask others
                if self._fault_live:
                    on_fault = set(self._on_fault_switch_ids())
                    if len(on_fault) and sw not in on_fault:
                        info["masked_switch_away_from_fault"] = int(sw)
                        local_mask_pen = float(self.invalid_live_action_penalty)
                    else:
                        curr = bool(self.net.switch.at[sw, "closed"])
                        new = not curr
                        self.net.switch.at[sw, "closed"] = new
                        info["toggled_switch"] = int(sw)
                        # Isolation if we OPEN a switch on the faulted line
                        if self.faulted_line is not None:
                            row = self.net.switch.loc[sw]
                            if (row.et == "l") and (int(row.element) == int(self.faulted_line)) and (new is False):
                                self.net.line.at[int(self.faulted_line), "in_service"] = False
                                self._fault_live = False
                                self._fault_award_pending = True
                                self._fault_isolated_at = self.current_step
                                # Rebuild general subset post-isolation
                                self.switch_ids = self._select_switch_subset(self.faulted_line)
                else:
                    # normal (post-isolation) switching
                    curr = bool(self.net.switch.at[sw, "closed"])
                    self.net.switch.at[sw, "closed"] = (not curr)
                    info["toggled_switch"] = int(sw)

            else:
                # Load curtailment +/- on a single load (only if enabled)
                if self.enable_setpoint_shedding and L > 0:
                    rel = a - (1 + K)
                    if 0 <= rel < 2 * L:
                        load_idx = rel // 2
                        direction = +1 if (rel % 2 == 0) else -1
                        lid = self.load_ids[load_idx]

                        if self._fault_live and self.mask_load_actions_during_live_fault:
                            info["masked_load_action"] = int(lid)
                            local_mask_pen = float(self.invalid_live_action_penalty)
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
                    else:
                        info["noop"] = True
                else:
                    # No set-point shedding in action space: defensive noop
                    info["noop"] = True

            # Solve PF (returns False while fault is live; we do NOT change the net)
            ok = self._runpf()

            # Fault protection timer bookkeeping
            if self._fault_live:
                self._fault_live_steps += 1
            else:
                self._fault_live_steps = 0

            # Reward
            total, comps = self._reward()

            # Add mask penalty if we ignored a bad live-fault action
            if local_mask_pen:
                comps["invalid_live_action"] = comps.get("invalid_live_action", 0.0) + float(local_mask_pen)

            # PF failure handling: do NOT count live-fault PF "failures" in the streak
            if not ok:
                if not self._fault_live:
                    self.pf_fail_streak += 1
                    comps["pf_fail"] = float(self.pf_failure_penalty)
            else:
                self.pf_fail_streak = 0

            # Served fractions (for eval; safe defaults while live)
            served_tot_frac, served_crit_frac, served_imp_frac = self._served_fractions()
            info.update({
                "served_total_frac": served_tot_frac,
                "served_crit_frac":  served_crit_frac,
                "served_imp_frac":   served_imp_frac,
                "der_bound_frac":    float(self._der_bound_frac_ep),  # <-- expose bound per step
                "fault_live":        bool(self._fault_live),
                "isolation_happened": (self._fault_isolated_at == self.current_step),
            })

            if len(self.net.load):
                base_mw = float(self.net.load["p_base_mw"].sum())
                mask = self._served_mask_by_connectivity(self.net)
                served_mw = float(self.net.load.loc[mask, "p_mw"].sum())
            else:
                base_mw, served_mw = 0.0, 0.0
            info.update({
                "der_bound_mw": base_mw * float(self._der_bound_frac_ep),
                "served_mw": served_mw,
            })

            # Termination
            done = (
                (self.current_step >= self.max_steps)
                or (self.pf_fail_streak >= self.failure_patience)
                or (self._fault_live and (self._fault_live_steps >= int(self.protection_trip_delay)))
            )

            info.update({"powerflow_success": ok, "reward_components": comps})
            return self._obs(), float(sum(comps.values())), bool(done), False, info

        except Exception as e:
            self.log.error(f"Error in step(): {e}", exc_info=True)
            comps = {"pf_fail": float(self.pf_failure_penalty)}
            return self._obs(), float(sum(comps.values())), True, False, {"error": str(e), "reward_components": comps}

    # ------------------------------------------------------------------ #
    # Reward / Observation / PF
    # ------------------------------------------------------------------ #
    def _source_buses_for(self, net) -> set[int]:
        """Buses treated as energized sources: ext_grid ∪ gen ∪ sgen (in_service)."""
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
        # fallback (masking only): if no explicit sources, assume first in-service bus
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
        """
        True if a load's bus is connected (through *closed* switches/branches) to any source bus.
        Uses pandapower.topology.unsupplied_buses with custom slacks (ext_grid/gen/sgen).
        """
        import pandapower.topology as top

        if not len(net.load):
            return np.zeros(0, dtype=bool)

        # Build a graph respecting switch states (open switches break paths)
        g = top.create_nxgraph(
            net,
            respect_switches=True,
            include_lines=True,
            include_trafos=True,
            include_impedances=True,
            include_out_of_service=False,
            include_switches=False,  # bus connectivity only
        )

        slacks = list(self._source_buses_for(net))

        # Get unsupplied buses, then invert mask for loads
        try:
            # Newer pp versions
            unsup = set(top.unsupplied_buses(net, slacks=slacks, respect_switches=True))
        except TypeError:
            # Older pp versions might not accept slacks kw; approximate via reachability
            unsup = set(net.bus.index.tolist())
            for s in slacks:
                try:
                    comp = set(top.connected_component(g, s))
                    unsup -= comp
                except Exception:
                    pass

        load_buses = net.load.bus.astype(int).values
        mask = np.array([int(b) not in unsup for b in load_buses], dtype=bool)

        # Safety fallback: if all-False, treat in-service loads as supplied for reward purposes
        if not mask.any():
            mask = net.load.in_service.astype(bool).values if "in_service" in net.load else np.ones(len(net.load), dtype=bool)
        return mask

    def _served_fractions(self) -> Tuple[float, float, float]:
        """(served_total_frac, served_crit_frac, served_imp_frac) for current net."""
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
        """(total_frac, crit_frac, imp_frac) for an arbitrary net snapshot."""
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
        shed_w = float(w.get("shed", -1.0))          # FRACTIONAL, post-isolation only
        sw_w   = float(w.get("switch", -0.01))
        iso_w  = float(w.get("fault_isolated", 300.0))
        live_w = float(w.get("fault_live", -80.0))
        v_delta_w = float(w.get("volt_delta", 0.5))
        rest_w = float(w.get("restore_delta", 5.0))

        comps = {"crit": 0.0, "imp": 0.0, "volt": 0.0, "shed": 0.0, "switch": 0.0,
                 "fault": 0.0, "fault_live": 0.0}

        # --- Connectivity-aware served / kept (scale-free) ---
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

            # Penalize fractional curtailment only AFTER isolation
            if not self._fault_live and p_tot > 1e-9:
                curtail_frac = float(1.0 - (p_srv / p_tot))
                comps["shed"] = shed_w * curtail_frac

        # Voltage penalty + shaping on improvement (available once PF succeeds)
        if hasattr(self.net, "res_bus") and len(self.net.res_bus):
            v = np.nan_to_num(self.net.res_bus.vm_pu.values, nan=0.0, posinf=2.0, neginf=0.0)
            vmin, vmax = self.v_limits
            dev = np.clip(vmin - v, 0, None) + np.clip(v - vmax, 0, None)
            comps["volt"] = v_pen * float(np.sum(dev))
            try:
                v_now = np.nan_to_num(self.net.res_bus.vm_pu.values, nan=1.0)
                v_prev = np.nan_to_num(self.prev.res_bus.vm_pu.values, nan=1.0) if hasattr(self.prev, "res_bus") and len(self.prev.res_bus) else v_now
                dev_now  = np.clip(vmin - v_now, 0, None) + np.clip(v_now - vmax, 0, None)
                dev_prev = np.clip(vmin - v_prev, 0, None) + np.clip(v_prev - vmax, 0, None)
                comps["volt"] += v_delta_w * float(np.sum(dev_prev) - np.sum(dev_now))
            except Exception:
                pass

        # Penalize switch churn (delta vs previous state)
        if len(self.net.switch) and len(self.prev.switch):
            prev_s = self.prev.switch["closed"].astype(int).values
            curr_s = self.net.switch["closed"].astype(int).values
            comps["switch"] = sw_w * int(np.sum(np.abs(prev_s - curr_s)))

        # Per-step penalty while the fault is energized
        if self._fault_live:
            comps["fault_live"] = live_w

        # One-off isolation award
        if self._fault_award_pending:
            comps["fault"] = iso_w
            self._fault_award_pending = False
            self._fault_awarded = True

        # Restoration shaping: reward increases in served TOTAL fraction after isolation
        if rest_w != 0.0 and (not self._fault_live):
            now_tot, _, _ = self._served_fractions()
            prev_tot, _, _ = self._served_fractions_for(self.prev) if hasattr(self, "prev") else (now_tot, 0, 0)
            comps["restore_delta"] = rest_w * float(now_tot - prev_tot)

        # DER bound penalty (only post-isolation so the agent fixes the fault first)
        if not self._fault_live and self.bound_pen_w != 0.0:
            now_tot, _, _ = self._served_fractions()         # 0..1
            bound = float(np.clip(self._der_bound_frac_ep, 0.0, 1.0))
            if now_tot > bound + 1e-6:
                comps["bound_violation"] = self.bound_pen_w * float(now_tot - bound)

        total = float(sum(comps.values()))
        return total, comps

    def _obs(self):
        # Bus voltages
        n_buses = len(self.net.bus)
        v = np.ones(n_buses, dtype=np.float32)
        if hasattr(self.net, "res_bus") and len(self.net.res_bus):
            v = np.nan_to_num(self.net.res_bus.vm_pu.values.astype(np.float32),
                              nan=1.0, posinf=1.0, neginf=0.0)

        # Selected switches only (stable K ordering)
        if len(self.switch_ids):
            sw = self.net.switch.loc[self.switch_ids, "closed"].astype(np.float32).values
        else:
            sw = np.zeros(0, dtype=np.float32)

        # Loads: append shed_frac ONLY if set-point shedding is enabled
        if self.enable_setpoint_shedding and len(self.load_ids):
            if "shed_frac" not in self.net.load.columns:
                self.net.load["shed_frac"] = 0.0
            sf = self.net.load.loc[self.load_ids, "shed_frac"].astype(np.float32).values
        else:
            sf = np.zeros(0, dtype=np.float32)

        return np.concatenate([v, sw, sf]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # PF (no pruning; strictly respects the current net)
    # ------------------------------------------------------------------ #
    def _runpf(self) -> bool:
        """Run PF; while the fault is live we treat PF as failed/unstable."""
        if self._fault_live:
            return False
        try:
            pp.runpp(self.net, enforce_q_lims=True)
            return True
        except Exception as e1:
            self.log.warning(f"Power flow failed (NR+rqlims): {e1}")
        try:
            # Try BFSW variants — still without modifying the network.
            pp.runpp(self.net, algorithm="bfsw", enforce_q_lims=False, init="flat", tolerance_mva=1e-5)
            return True
        except Exception as e2:
            self.log.warning(f"Power flow failed (BFSW flat): {e2}")
        try:
            pp.runpp(self.net, algorithm="bfsw", enforce_q_lims=False, init="dc", tolerance_mva=1e-5)
            return True
        except Exception as e3:
            self.log.warning(f"Power flow failed (BFSW dc): {e3}")
            return False
