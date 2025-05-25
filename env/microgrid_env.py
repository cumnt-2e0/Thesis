import numpy as np
import pandapower as pp
import pandapower.plotting as plot
import gym
import matplotlib.pyplot as plt
from pandapower.plotting.geo import convert_geodata_to_geojson
from gym import spaces
from .builder import build_microgrid
from .constants import SWITCH_IDS

class MicrogridEnv(gym.Env):
    def __init__(self):
        super(MicrogridEnv, self).__init__()
        self.net = build_microgrid()
        self.switch_idx = SWITCH_IDS["SW1"]
        self.current_step = 0
        self.max_steps = 20

        obs_dim = len(self._get_state())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: do nothing, 1: open, 2: close

    def reset(self):
        self.net = build_microgrid()
        self.net.ext_grid.at[0, "in_service"] = False
        pp.runpp(self.net)
        self.current_step = 0
        return self._get_state().astype(np.float32)

    def step(self, action):
        if action == 1:
            self.net.switch.at[self.switch_idx, 'closed'] = False
        elif action == 2:
            self.net.switch.at[self.switch_idx, 'closed'] = True

        pp.runpp(self.net)

        state = self._get_state().astype(np.float32)
        done = self._check_done()
        self.current_step += 1
        info = {}

        return state, 0.0, done, info

    def _get_state(self):
        voltages = self.net.res_bus.vm_pu.values
        load_powers = self.net.res_load.p_mw.values
        switch_states = self.net.switch.closed.values.astype(float)
        return np.concatenate([voltages, load_powers, switch_states])

    def _check_done(self):
        return self.current_step >= self.max_steps

    def render(self, mode='human'):
        # Convert to GeoJSON format for plotting
        convert_geodata_to_geojson(self.net)

        # Get the axes from the base plot
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        plot.simple_plot(self.net, respect_switches=True, ax=ax)

        # === Annotate Elements ===
        for i, row in self.net.bus.iterrows():
            if i not in self.net.bus_geodata.index:
                continue
            x = self.net.bus_geodata.at[i, "x"]
            y = self.net.bus_geodata.at[i, "y"]

            # Storage
            if i in self.net.storage.bus.values:
                ax.scatter(x, y, color='orange', marker='s', s=100, label='Battery' if i == self.net.storage.bus.values[0] else "")
                ax.text(x, y + 0.1, "BAT", ha='center', color='orange')

            # DERs
            if i in self.net.sgen.bus.values:
                ax.scatter(x, y, color='green', marker='^', s=100, label='DER' if i == self.net.sgen.bus.values[0] else "")
                ax.text(x, y + 0.1, "DER", ha='center', color='green')

            # Critical loads
            for lb, prio in zip(self.net.load.bus.values, self.net.load.priority.values):
                if lb == i and prio == 3:
                    ax.scatter(x, y, color='red', marker='*', s=120, label='Critical' if i == lb else "")
                    ax.text(x, y - 0.15, "CRIT", ha='center', color='red')

        # Switch labels
        for _, row in self.net.switch.iterrows():
            bus = row["bus"]
            if bus in self.net.bus_geodata.index:
                x = self.net.bus_geodata.at[bus, "x"]
                y = self.net.bus_geodata.at[bus, "y"]
                ax.text(x, y - 0.25, row["name"], fontsize=8, color='blue', ha='center')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower left")

        plt.title("Microgrid Topology with Node Roles and Switch Labels")
        plt.show()
