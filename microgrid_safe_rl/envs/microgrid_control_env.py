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
  grace_steps: 0..N
  enable_setpoint_shedding: false
  mask_load_actions_during_live_fault: true
  shed_step: 0.1
  der_unavailability: {...}  # only applied when scenario.der: true
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

Scenario keys (phase configs):
  enabled: true/false
  modes: ["line", "load"]
  severity: 0.8
  der: false            # when true, env.der_unavailability is applied at reset
  cascading_line_faults: false
  cascade_prob: 0.0..1.0
  max_cascades: 1..N
  exclude_bridges: false
  exclude_bridges_prob: 0.0..1.0
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
    """
    RL environment for islanded microgrid switching & load shedding.
    This environment wraps a `pandapower` network to train an agent to:
    - isolate live line faults by opening an on-fault line switch,
    - reconfigure the network (close ties) to restore supply,
    - optionally curtail loads via set-point shedding,
    - deal with optional DER unavailability (outages/derating), and
    - (optionally) endure cascading line faults, to model real world outages.

    Notes
    -----
    * We never mutate the topology automatically (no pruning). PF failures are part of the task.
    * While a live fault exists, `_runpf()` intentionally returns `False` to produce a
    consistent learning signal; after isolation, PF attempts resume.
    * A small, fixed subset of switches (size K) is exposed to the agent to constrain the
    action space. The subset prioritizes switches on / near the (known) faulted line.
    """
    metadata = {"render_modes": []}

    # ------------------------------------------------------------------ #
    # __init__
    # ------------------------------------------------------------------ #
    def __init__(self, net, config: Dict[str, Any], scenario: Optional[Dict[str, Any]] = None):
        """
        Build the environment around a prepared pandapower net.

        Parameters
        ----------
        net : pandapowerNet
        The feeder/microgrid to control. We will add helper columns in-place
        (e.g., `p_base_mw`, `shed_frac`) via :func:`_prepare_net_inplace`.
        config : dict
        Environment configuration (limits, timing, reward weights, etc.).
        scenario : dict, optional
        Scenario toggles for disturbance injection (e.g., modes, severity,
        cascading flags). If None, sensible defaults are used.
        """

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

        # Working episode state (initialized in reset, but define here for clarity)
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

        # DER availability (sampled per episode when scenario.der is True)
        self.der_cfg = self.config.get("der_unavailability", {})
        self.bound_pen_w = float(self.config.get("reward_weights", {}).get("bound_violation", -20.0))
        self._der_bound_frac_ep = 1.0

        # Cascading state (overload-driven only)
        self._active_faults: set[int] = set()      # lines faulted/tripped this episode
        self._overload_watch: Dict[int, int] = {}  # line_id -> consecutive PF-success steps over threshold
        self._cascade_remaining = 0                # remaining extra trips allowed this episode

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
    
    def _cascade_cfg(self) -> dict:
        """Read cascade sub-config with sane defaults."""
        c = self.scenario.get("cascade", {}) or {}
        return {
            "strategy":           c.get("strategy", "overload"),
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
    
    # ------------------------------------------------------------------ #
    # _der_is_enabled_for_this_episode: gate DER logic by SCENARIO flag
    # ------------------------------------------------------------------ #
    def _der_is_enabled_for_this_episode(self) -> bool:
        """Return True if scenario enables DER unavailability AND env der config itself is enabled."""
        return bool(self.scenario.get("der", False) and self.der_cfg.get("enabled", False))

    # ------------------------------------------------------------------ #
    # _prepare_net_inplace
    # ------------------------------------------------------------------ #
    def _prepare_net_inplace(self, net):
        """
        Add/ensure helper columns required by the environment.

        Creates the following columns if missing:
        - **load**: `p_base_mw`, `q_base_mvar` (if `q_mvar` exists), `shed_frac`, `priority`
        - **gen/sgen**: `p_base_mw`, `in_service`

        Parameters
        ----------
        net : pandapowerNet
        Network object to be modified in-place.
        """

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
    # reset
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        """
        Start a new episode: inject disturbance, apply static DER limits, run PF.
        
        Steps
        -----
        1. Restore `net` from `net0` (both already prepared).
        2. Inject a disturbance per `scenario` (line fault *or* load surge).
        3. If `scenario.get('der', False)` is True, apply per-episode DER outages/derating.
        4. Rebuild the K-sized switch subset (now that the fault is known).
        5. Attempt an initial PF. If it fails (and no live fault), open a grace window so
        the agent can attempt recovery actions without immediate termination.

        Returns
        -------
        obs : np.ndarray
        Observation vector `[vm_pu | K switch states | L shed_frac]`.
        info : dict
        Debugging hints (`powerflow_success`, `faulted_line`, DER bound, grace budget).
        """

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

        # Cascading (overload) state
        self._active_faults.clear()
        self._overload_watch.clear()
        self._cascade_remaining = int(self._cascade_cfg()["max_additional"])

        # Grace-step controls
        self._grace_budget_max = int(self.config.get("grace_steps", 0))
        self._grace_budget = 0
        self._needs_grace = False

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

            # DER (if enabled by scenario)
            if bool(self.scenario.get("der", False)):
                self.der_cfg = self.config.get("der_unavailability", {})
                self._apply_der_unavailability(seed=None if seed is None else seed + 1337 + t)
                if self.der_cfg.get("enabled", False):
                    print("[DEBUG] Applying DER unavailability with config:", self.der_cfg)
            else:
                self._der_bound_frac_ep = 1.0

            # Rebuild K-sized switch subset AFTER we know the fault
            self.switch_ids = self._select_switch_subset(self.faulted_line)

            # First PF attempt
            ok = self._runpf()

            # Grace path if infeasible and not live-fault
            if not ok and not self._fault_live:
                self._needs_grace = (self._grace_budget_max > 0)
                self._grace_budget = self._grace_budget_max
                if self._needs_grace:
                    print(f"[DEBUG] reset(): PF infeasible -> starting grace window with {self._grace_budget} steps")
                    ok = True
                break

            if ok or self._fault_live:
                break

        return self._obs(), {
            "powerflow_success": ok,
            "faulted_line": self.faulted_line,
            "der_bound_frac": float(self._der_bound_frac_ep),
            "grace_budget": int(self._grace_budget),
            "needs_grace": bool(self._needs_grace),
        }

    # ------------------------------------------------------------------ #
    # _safe_fault_line: 
    # ------------------------------------------------------------------ #
    def _safe_fault_line(self) -> Optional[int]:
        """
        Choose a *non-bridge* line to fault if possible.

        We prefer non-bridges to avoid splitting the graph at reset time. If all lines are
        bridges (or detection fails), fall back to any in-service line. Returns `None` if
        there are no suitable lines.
        """
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
            if not safe:
                return None
            return int(np.random.default_rng().choice(safe))

        except Exception as e:
            self.log.debug(f"_safe_fault_line fallback due to: {e}")
            if len(self.net.line):
                candidates = [int(i) for i, r in self.net.line.iterrows() if bool(r.in_service)]
                return None if not candidates else int(np.random.default_rng().choice(candidates))
            return None

    # ------------------------------------------------------------------ #
    # _pick_fault_line
    # ------------------------------------------------------------------ #
    def _pick_fault_line(self, rng, seed=None) -> Optional[int]:
        """
        Randomly pick a line index to mark as *live fault*.

        If `scenario.exclude_bridges` or `scenario.exclude_bridges_prob` is set, the
        selection is biased toward non-bridge lines using :func:`_safe_fault_line`.
        """
        if not len(self.net.line):
            return None
        p = float(self.scenario.get("exclude_bridges_prob", 0.0))
        excl = bool(self.scenario.get("exclude_bridges", False) or (rng.random() < p))
        if excl:
            return self._safe_fault_line()
        return int(np.random.default_rng(seed).choice(self.net.line.index))

    # ------------------------------------------------------------------ #
    # _inject_disturbance
    # ------------------------------------------------------------------ #
    def _inject_disturbance(self, seed=None):
        """
        Apply the per-episode disturbance defined by `scenario`.

        Modes
        -----
        - "line": create a *live* line fault — the line stays energized until the agent
        opens an on-fault line switch; PF returns `False` while the fault is live.
        - "load": apply a one-off random load increase to a single load (independent of
        topology-based shedding).

        Cascading line faults
        ---------------------
        If `scenario.cascading_line_faults` is True, with probability `cascade_prob` we
        schedule an additional lines to fail after a small random delay, to mimic real 
        world failure. The cascade is queued in `_cascade_targets` and triggered 
        in :func:`step`.
        """

        rng = np.random.default_rng(seed)
        modes = self.scenario.get("modes", ["line", "load"])
        sev = float(self.scenario.get("severity", 0.8))
        mode = rng.choice(modes) if modes else None

        if mode == "line" and len(self.net.line):
            li = self._pick_fault_line(rng, seed)
            if li is not None:
                # Primary live fault
                self.faulted_line = int(li)
                self._fault_live = True
                self._fault_live_steps = 0
                self.switch_ids = self._select_switch_subset(self.faulted_line)
                self._active_faults.add(int(li))
            return

        if mode == "load" and len(self.net.load):
            li = int(rng.choice(self.net.load.index))
            f = float(np.clip(1.0 + 0.2 * sev + rng.uniform(0, 0.3 * sev), 1.0, 2.0))
            self.net.load.at[li, "p_mw"] = float(self.net.load.at[li, "p_mw"]) * f
            if "q_mvar" in self.net.load.columns:
                self.net.load.at[li, "q_mvar"] = float(self.net.load.at[li, "q_mvar"]) * f


    # ------------------------------------------------------------------ #
    # _sample_der_bound_frac: sample available DER supply fraction
    # ------------------------------------------------------------------ #
    def _sample_der_bound_frac(self, rng) -> float:
        """
        Sample the per-episode DER supply bound as a fraction of base load (0..1).

        The bound is drawn from `der_unavailability.p_deficit_frac = [lo, hi]` and stored
        in `_der_bound_frac_ep`. It controls the *post-isolation* `bound_violation` reward.
        """
        cfg = self.der_cfg or {}
        lo, hi = cfg.get("p_deficit_frac", [0.0, 0.0])
        lo = float(lo); hi = float(hi)
        hi = max(hi, lo)
        deficit = rng.uniform(lo, hi)
        return float(np.clip(1.0 - deficit, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # _apply_der_unavailability: random small DER outages + common derating (static per reset)
    # ------------------------------------------------------------------ #
    def _apply_der_unavailability(self, seed=None):
        """
        Apply DER outages & derating once per episode (when `scenario.der` is True).

        Logic
        -----
        1) Sample `_der_bound_frac_ep` from `p_deficit_frac`.
        2) With probability `random_outage_prob`, mark a *small* online DER (≤ `max_der_size_mw`)
        as out-of-service. (You can cap how disruptive outages are during early training.)
        3) Compute a *common scale* that would bring online DER toward the target bound and
        apply (optionally jittered by `scaling_range`) to each online DER's `p_mw`.
        4) Print a concise debug summary if `der_unavailability.enabled`.
        """
        # If not configured for static application, only sample the bound fraction and return
        if not self.der_cfg or not self.der_cfg.get("static_on_reset", False):
            rng = np.random.default_rng(seed)
            self._der_bound_frac_ep = self._sample_der_bound_frac(rng)
            return

        rng = np.random.default_rng(seed)
        cfg = self.der_cfg
        self._der_bound_frac_ep = self._sample_der_bound_frac(rng)

        outage_p = float(cfg.get("random_outage_prob", 0.0))
        s_lo, s_hi = cfg.get("scaling_range", [0.5, 1.0])
        s_lo = float(s_lo); s_hi = float(max(s_hi, s_lo))
        max_size = float(cfg.get("max_der_size_mw", 1e9))  # default huge → disable filter
        max_disabled = int(cfg.get("max_disabled", 1))

        # Base total load (for bound target)
        if len(self.net.load):
            if "p_base_mw" not in self.net.load.columns:
                self.net.load["p_base_mw"] = self.net.load["p_mw"].astype(float)
            P_L = float(self.net.load["p_base_mw"].sum())
        else:
            P_L = 0.0

        # Target bound (fraction of load that should be feasible)
        P_target = self._der_bound_frac_ep * P_L

        der_tables = [tb for tb in ("gen", "sgen") if hasattr(self.net, tb) and len(getattr(self.net, tb))]
        if not der_tables:
            return

        # Random small-DER outages (capped)
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

        # Common derating scale with jitter
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

        # Actual online capacity after outages/derating (debug)
        P_online = 0.0
        for tb in der_tables:
            df = getattr(self.net, tb)
            P_online += float(df.loc[df.in_service.astype(bool), "p_mw"].sum())

        if self.der_cfg.get("enabled", False):
            print("[DEBUG] Disabled DERs:")
            if disabled:
                for tb, idx, p in disabled:
                    print(f"  - {tb}[{idx}] P={p:.2f} MW")
            else:
                print("  - None")
            print(f"[DEBUG] Load={P_L:.2f} MW | Target={P_target:.2f} MW | Online={P_online:.2f} MW")

        # Note: ext_grid (slack) may still cover any mismatch; the *bound penalty* (below)
        # forces the agent to shed if it serves above the bound.

    # ------------------------------------------------------------------ #
    # _on_fault_switch_ids: switches that sit on the currently faulted line
    # ------------------------------------------------------------------ #
    def _on_fault_switch_ids(self) -> List[int]:
        """Return switch indices physically on the faulted line (pandapower switch.et == 'l')."""
        if self.faulted_line is None or not len(self.net.switch):
            return []
        try:
            return list(self.net.switch.index[
                (self.net.switch["et"] == "l") &
                (self.net.switch["element"] == int(self.faulted_line))
            ])
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    # _select_switch_subset: choose exactly K switches, biased near the fault
    # ------------------------------------------------------------------ #
    def _select_switch_subset(self, faulted_line: Optional[int]):
        """
        Choose a K-sized list of switch IDs to expose in the action space.

        Priority order
        --------------
        1. Switches on the faulted line (if known).
        2. *Open* tie switches close to the fault endpoints (graph distance within
        `near_fault_hops`).
        3. Remaining switches by proximity.
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
            import pandapower.topology as top
            g = top.create_nxgraph(self.net, include_switches=True)
        except Exception:
            pass
        if faulted_line is not None and faulted_line in self.net.line.index:
            row = self.net.line.loc[faulted_line]
            fault_end_buses = {int(row.from_bus), int(row.to_bus)}
        if g is not None and fault_end_buses:
            import networkx as nx  # only used if graph created successfully
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
    
    def _fault_cluster_buses(self) -> set:
        """
        Return the set of buses at the endpoints of currently faulted/tripped lines.
        We use this as the 'origin' of the cascade neighborhood.
        """
        buses = set()
        if not hasattr(self, "_active_faults") or not len(self._active_faults):
            return buses
        for ln in self._active_faults:
            if ln in self.net.line.index:
                row = self.net.line.loc[ln]
                buses.add(int(row.from_bus)); buses.add(int(row.to_bus))
        return buses

    def _line_neighbors_within_hops(self, hop_limit: int, sample_k: int = 0) -> List[int]:
        """
        Get line ids whose endpoints lie within 'hop_limit' graph hops of the current
        fault cluster. Only returns lines that are currently in_service (i.e., candidates).
        """
        import pandapower.topology as top
        if hop_limit <= 0 or not len(self.net.line):
            # all in-service lines that aren’t already faulted/tripped
            return [int(i) for i in self.net.line.index if bool(self.net.line.at[i, "in_service"]) and (i not in self._active_faults)]

        src_buses = self._fault_cluster_buses()
        if not src_buses:
            return [int(i) for i in self.net.line.index if bool(self.net.line.at[i, "in_service"]) and (i not in self._active_faults)]

        try:
            g = top.create_nxgraph(self.net, include_switches=True)
        except Exception:
            return [int(i) for i in self.net.line.index if bool(self.net.line.at[i, "in_service"]) and (i not in self._active_faults)]

        # single-source SSSP from each src bus up to hop_limit
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
        """Trip the given lines either as live faults or as immediate outages."""
        tripped = []
        for ln in line_ids:
            if ln not in self.net.line.index or ln in self._active_faults:
                continue
            if as_live_fault:
                # convert to a new live fault the agent must isolate
                self.faulted_line = int(ln)
                self._fault_live = True
                self._fault_live_steps = 0
                self.switch_ids = self._select_switch_subset(self.faulted_line)
            else:
                # hard-trip the line out of service
                self.net.line.at[int(ln), "in_service"] = False
            self._active_faults.add(int(ln))
            tripped.append(int(ln))
            # reset watch counter so we don't re-consider it
            self._overload_watch.pop(int(ln), None)

        if tripped:
            info.setdefault("cascade_tripped", []).extend(tripped)

    def _cascade_overload_step(self, pf_ok: bool, info: dict):
        """
        Evaluate overload-driven cascading after a *successful* PF.
        Tripping respects:
        - max_additional total this episode,
        - per_wave_max per step,
        - neighborhood hop_limit around the current fault cluster.
        """
        cfg = self._cascade_cfg()
        # Enable only if the new cascade block allows additional trips
        if cfg["strategy"] != "overload" or cfg["max_additional"] <= 0:
            return
        if not pf_ok:
            # only update/act when PF produced results; otherwise res_line.loading_percent is invalid
            return
        if not hasattr(self.net, "res_line") or not len(self.net.res_line):
            return
        if self._cascade_remaining <= 0:
            return

        hop_limit = cfg["hop_limit"]
        cand = self._line_neighbors_within_hops(hop_limit, sample_k=cfg["neighbor_sample_k"])
        if not cand:
            return

        # Read loading & update watch counters
        load = self.net.res_line.loading_percent  # series aligned with line index
        min_pct = cfg["overload_min_pct"]
        hold = cfg["min_hold_steps"]
        drop = cfg["hysteresis_drop"]
        live_faults = cfg["live_faults"]
        per_wave_max = min(cfg["per_wave_max"], self._cascade_remaining)

        eligible = []
        for ln in cand:
            try:
                lpct = float(load.at[ln])
            except Exception:
                continue
            # hysteresis maintenance
            if lpct >= (min_pct - 1e-6):
                self._overload_watch[ln] = self._overload_watch.get(ln, 0) + 1
            elif lpct <= (min_pct - drop):
                self._overload_watch.pop(ln, None)
            # eligibility check
            if self._overload_watch.get(ln, 0) >= hold and lpct >= min_pct:
                eligible.append((ln, lpct))

        if not eligible:
            return

        # Map loading% -> trip probability
        pl = sorted((float(a), float(b)) for a, b in cfg["prob_loading_map"])
        def prob_for_loading(x: float) -> float:
            p = 0.0
            for thr, val in pl:
                if x >= thr:
                    p = val
            return float(np.clip(p, 0.0, 1.0))

        # Sample up to per_wave_max to trip this step
        rng = np.random.default_rng()
        chosen = []
        for ln, lpct in sorted(eligible, key=lambda t: -t[1]):  # higher loading first
            if len(chosen) >= per_wave_max:
                break
            if rng.random() < prob_for_loading(lpct):
                chosen.append(int(ln))

        if not chosen:
            return

        # Enforce global cap
        if len(chosen) > self._cascade_remaining:
            chosen = chosen[: self._cascade_remaining]

        # Trip them
        self._trip_lines(chosen, as_live_fault=live_faults, info=info)
        self._cascade_remaining -= len(chosen)




    # ------------------------------------------------------------------ #
    # step: apply action, (optionally) cascade, run PF, compute reward, terminate
    # ------------------------------------------------------------------ #
    def step(self, action):
        """
        Advance one control step: apply action, (optionally) trigger cascade, run PF, reward.

        The function has two paths:
        1) **Grace mode**: if `_needs_grace` and budget > 0, apply the action *then* attempt PF.
        PF penalties are suppressed so the agent can recover.
        2) **Normal mode**: apply action, attempt PF, handle failure streak / termination.


        Returns follow the Gymnasium API: `(obs, reward, terminated, truncated, info)`.
        """
    
        try:
            self.prev = copy.deepcopy(self.net)
            self.current_step += 1
            info = {"step": self.current_step, "faulted_line": self.faulted_line}
            ok = False  # ensure defined

            # Parse action
            a = int(np.asarray(action).item() if isinstance(action, (np.ndarray, list)) else int(action))
            if a < 0 or a >= self.action_space.n:
                a = 0

            K, L = self.K, self.L
            local_mask_pen = 0.0

            # -----------------------------
            # Grace path
            # -----------------------------
            if self._needs_grace and self._grace_budget > 0:
                info["emergency"] = 1
                info["grace_left_before"] = int(self._grace_budget)

                # Apply action
                if a == 0:
                    info["noop"] = True
                elif 1 <= a <= K:
                    sw = self.switch_ids[a - 1]
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
                            if self.faulted_line is not None:
                                row = self.net.switch.loc[sw]
                                if (row.et == "l") and (int(row.element) == int(self.faulted_line)) and (new is False):
                                    self.net.line.at[int(self.faulted_line), "in_service"] = False
                                    self._fault_live = False
                                    self._fault_award_pending = True
                                    self._fault_isolated_at = self.current_step
                                    self.switch_ids = self._select_switch_subset(self.faulted_line)
                    else:
                        curr = bool(self.net.switch.at[sw, "closed"])
                        self.net.switch.at[sw, "closed"] = (not curr)
                        info["toggled_switch"] = int(sw)
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

                # PF after action
                ok = self._runpf()
                self._grace_budget -= 1
                info["grace_left_after"] = int(self._grace_budget)

                # Overload-driven cascade (only if PF succeeded)
                if ok:
                    self._cascade_overload_step(pf_ok=True, info=info)
                    info["cascade_remaining"] = int(self._cascade_remaining)

                # Fault protection timer
                if self._fault_live:
                    self._fault_live_steps += 1
                else:
                    self._fault_live_steps = 0

                # Reward (suppress PF fail penalty during grace)
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

                # Info enrich
                served_tot_frac, served_crit_frac, served_imp_frac = self._served_fractions()
                info.update({
                    "served_total_frac": served_tot_frac,
                    "served_crit_frac":  served_crit_frac,
                    "served_imp_frac":   served_imp_frac,
                    "der_bound_frac":    float(self._der_bound_frac_ep),
                    "fault_live":        bool(self._fault_live),
                    "isolation_happened": (self._fault_isolated_at == self.current_step),
                    "powerflow_success": bool(ok),
                })
                if len(self.net.load):
                    base_mw = float(self.net.load["p_base_mw"].sum())
                    mask = self._served_mask_by_connectivity(self.net)
                    served_mw = float(self.net.load.loc[mask, "p_mw"].sum())
                else:
                    base_mw, served_mw = 0.0, 0.0
                info.update({"der_bound_mw": base_mw * float(self._der_bound_frac_ep), "served_mw": served_mw})

                done = (
                    (self.current_step >= self.max_steps)
                    or (self._fault_live and (self._fault_live_steps >= int(self.protection_trip_delay)))
                    or ((not ok) and (self._grace_budget == 0))
                )
                info.update({"reward_components": comps})
                return self._obs(), float(sum(comps.values())), bool(done), False, info

            # -----------------------------
            # Normal path
            # -----------------------------
            if a == 0:
                info["noop"] = True
            elif 1 <= a <= K:
                sw = self.switch_ids[a - 1]
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
                        if self.faulted_line is not None:
                            row = self.net.switch.loc[sw]
                            if (row.et == "l") and (int(row.element) == int(self.faulted_line)) and (new is False):
                                self.net.line.at[int(self.faulted_line), "in_service"] = False
                                self._fault_live = False
                                self._fault_award_pending = True
                                self._fault_isolated_at = self.current_step
                                self.switch_ids = self._select_switch_subset(self.faulted_line)
                else:
                    curr = bool(self.net.switch.at[sw, "closed"])
                    self.net.switch.at[sw, "closed"] = (not curr)
                    info["toggled_switch"] = int(sw)
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
                    info["noop"] = True

            # PF
            ok = self._runpf()

            # Overload-driven cascade when PF succeeded
            if ok:
                self._cascade_overload_step(pf_ok=True, info=info)
                info["cascade_remaining"] = int(self._cascade_remaining)

            # Enter grace window next step if infeasible, not live, and budget available
            if not ok and not self._fault_live and (self._grace_budget == 0) and (self._grace_budget_max > 0):
                self._needs_grace = True
                self._grace_budget = self._grace_budget_max
                print(f"[DEBUG] step(): PF infeasible -> starting grace window with {self._grace_budget} steps (from non-grace path)")

            # Fault protection timer
            if self._fault_live:
                self._fault_live_steps += 1
            else:
                self._fault_live_steps = 0

            # Reward
            _, comps = self._reward()

            if local_mask_pen:
                comps["invalid_live_action"] = comps.get("invalid_live_action", 0.0) + float(local_mask_pen)

            if not ok:
                if not self._fault_live:
                    self.pf_fail_streak += 1
                    comps["pf_fail"] = float(self.pf_failure_penalty)
            else:
                self.pf_fail_streak = 0

            # Info enrich
            served_tot_frac, served_crit_frac, served_imp_frac = self._served_fractions()
            info.update({
                "served_total_frac": served_tot_frac,
                "served_crit_frac":  served_crit_frac,
                "served_imp_frac":   served_imp_frac,
                "der_bound_frac":    float(self._der_bound_frac_ep),
                "fault_live":        bool(self._fault_live),
                "isolation_happened": (self._fault_isolated_at == self.current_step),
            })
            if len(self.net.load):
                base_mw = float(self.net.load["p_base_mw"].sum())
                mask = self._served_mask_by_connectivity(self.net)
                served_mw = float(self.net.load.loc[mask, "p_mw"].sum())
            else:
                base_mw, served_mw = 0.0, 0.0
            info.update({"der_bound_mw": base_mw * float(self._der_bound_frac_ep), "served_mw": served_mw})

            done = (
                (self.current_step >= self.max_steps)
                or (self.pf_fail_streak >= self.failure_patience)
                or (self._fault_live and (self._fault_live_steps >= int(self.protection_trip_delay)))
            )

            info.update({"powerflow_success": bool(ok),
                        "reward_components": comps,
                        "grace_left": int(self._grace_budget),
                        "needs_grace": bool(self._needs_grace)})
            return self._obs(), float(sum(comps.values())), bool(done), False, info

        except Exception as e:
            self.log.error(f"Error in step(): {e}", exc_info=True)
            comps = {"pf_fail": float(self.pf_failure_penalty)}
            return self._obs(), float(sum(comps.values())), True, False, {"error": str(e), "reward_components": comps}


    # ------------------------------------------------------------------ #
    # Reward / Observation / PF helpers
    # ------------------------------------------------------------------ #
    def _source_buses_for(self, net) -> set[int]:
        """
        Return buses treated as energized sources (ext_grid ∪ gen ∪ sgen).

        If none exist, fall back to the first in-service bus to keep masks sane.
        """
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
        Boolean mask of loads that are connected to any source bus.

        Uses `pandapower.topology` reachability while respecting switch states. If newer
        `unsupplied_buses` is available we use it; otherwise we fall back to components.
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
        """
        Compute shaped reward and its components for the current transition.

        Components
        ----------
        - `crit`, `imp`: fraction of critical/important demand currently served.
        - `volt`: penalty for voltage deviations outside `[vmin, vmax]`, plus a shaping
        bonus (`volt_delta`) for improvement vs previous step.
        - `shed`: penalty for curtailment fraction (only *after* fault isolation).
        - `switch`: small penalty for switch state changes to discourage chattering.
        - `fault_live`: per-step penalty while a live fault persists.
        - `fault`: one-off isolation award when the on-fault switch is opened.
        - `restore_delta`: shaping term proportional to increases in total served fraction
        after isolation.
        - `bound_violation`: penalty if served fraction exceeds the DER episode bound.
        - `alive`: optional small per-step reward to prevent early termination bias.
        """
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

        # Connectivity-aware served / kept
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

        if hasattr(self, "net") and hasattr(self.net, "res_bus") and len(self.net.res_bus):
            comps["alive"] = float(w.get("alive", 0.0))

        total = float(sum(comps.values()))
        return total, comps

    def _obs(self):
        """Build observation: [bus voltages | K selected switch states | L load shed_frac]."""
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
    # _runpf: run PF without pruning; fault-live counts as failure
    # ------------------------------------------------------------------ #
    def _runpf(self) -> bool:
        """
        Attempt a power flow without altering the network.

        Returns `False` immediately while a live fault is active (training signal). Otherwise
        tries NR with Q-limits, then BFSW (flat & dc init) as fallbacks.
        """
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
