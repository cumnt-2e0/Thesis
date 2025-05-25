# env/microgrid_env.py

import numpy as np
import pandapower as pp
import pandapower.plotting as plot
import gym
from gym import spaces
import matplotlib.pyplot as plt
import networkx as nx
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
        self.action_space = spaces.Discrete(3)

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
        coords = self.net.bus_geodata
        G = nx.Graph()

        for i in self.net.bus.index:
            G.add_node(i, pos=(coords.at[i, 'x'], coords.at[i, 'y']))

        for _, row in self.net.line.iterrows():
            G.add_edge(row['from_bus'], row['to_bus'])

        pos = nx.get_node_attributes(G, 'pos')

        plt.figure(figsize=(12, 7))
        nx.draw(G, pos, node_color='lightblue', edge_color='gray', with_labels=True, node_size=300)

        legend_elements = []

        for i in self.net.sgen.bus:
            x, y = coords.at[i, 'x'], coords.at[i, 'y']
            plt.scatter(x, y, color='green', marker='^', s=150, zorder=3, label='DER')

        for i in self.net.storage.bus:
            x, y = coords.at[i, 'x'], coords.at[i, 'y']
            plt.scatter(x, y, color='orange', marker='s', s=150, zorder=3, label='Battery')

        for bus, priority in zip(self.net.load.bus.values, self.net.load.priority.values):
            if priority == 3:
                x, y = coords.at[bus, 'x'], coords.at[bus, 'y']
                plt.scatter(x, y, color='red', marker='*', s=200, zorder=3, label='Critical Load')

        gen_buses = self.net.gen.bus.values
        for i in gen_buses:
            x, y = coords.at[i, 'x'], coords.at[i, 'y']
            plt.scatter(x, y, color='purple', marker='P', s=150, zorder=3, label='Blackstart Gen')

        for _, row in self.net.switch.iterrows():
            from_bus = row['bus']
            line_idx = row['element']
            line = self.net.line.loc[line_idx]
            fb, tb = line['from_bus'], line['to_bus']
            x1, y1 = coords.at[fb, 'x'], coords.at[fb, 'y']
            x2, y2 = coords.at[tb, 'x'], coords.at[tb, 'y']
            x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
            plt.scatter(x_mid, y_mid, color='blue', marker='X', s=100, zorder=4)
            plt.text(x_mid, y_mid - 0.15, row['name'], fontsize=8, color='blue', ha='center')

        # Deduplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower left')
        plt.title("Annotated Microgrid Topology with Symbols and Switches")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
