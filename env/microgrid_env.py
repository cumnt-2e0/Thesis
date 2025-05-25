import numpy as np
import pandapower as pp
import pandapower.plotting.plotly as plotly
import gym
from gym import spaces
from .builder import build_microgrid
from .constants import SWITCH_INDICES

class MicrogridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.net = build_microgrid()
        self.switch_idx = SWITCH_INDICES["SW1"]
        self.current_step = 0
        self.max_steps = 20

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._get_state()),), dtype=np.float32)
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
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_state().astype(np.float32), 0.0, done, {}

    def _get_state(self):
        return np.concatenate([
            self.net.res_bus.vm_pu.values,
            self.net.res_load.p_mw.values,
            self.net.switch.closed.values.astype(float)
        ])

    def render(self, mode='human'):
        fig = plotly.simple_plotly(self.net, respect_switches=True, bus_size=1.5, line_width=2)

        # Annotate switch names
        for i, row in self.net.switch.iterrows():
            bus_idx = row["bus"]
            x = self.net.bus_geodata.at[bus_idx, "x"]
            y = self.net.bus_geodata.at[bus_idx, "y"]
            fig.add_annotation(
                x=x, y=y, text=row["name"],
                showarrow=True, arrowhead=1, ax=0, ay=-20,
                font=dict(size=12, color="black")
            )

        fig.update_layout(title="Expanded Microgrid Layout with Switches")
        fig.show()

