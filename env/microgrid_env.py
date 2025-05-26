# This module defines a custom Gym environment representing a microgrid,
# enabling reinforcement learning agents to interact with and control a simulated power system.
# It includes visualisation with Plotly and dynamic updates to grid topology via switch operations.import numpy as np

import pandapower as pp
import gym
from gym import spaces
import numpy as np
import plotly.graph_objects as go
from pandapower.plotting.plotly.traces import create_line_trace
from .builder import build_microgrid
from .constants import SWITCH_IDS, LOAD_PRIORITIES

class MicrogridEnv(gym.Env):
    def __init__(self):
        super(MicrogridEnv, self).__init__()
        # Build initial grid structure from builder module
        self.net = build_microgrid()

        # Index of the switch controlled by the environment (single-switch version)
        self.switch_idx = SWITCH_IDS["SW1"]
        self.current_step = 0
        self.max_steps = 20 # Defines maximum episode length

        # Define observation space: voltages, load powers, and switch states
        obs_dim = len(self._get_state())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Action space: 0 = do nothing, 1 = open switch, 2 = close switch
        self.action_space = spaces.Discrete(3)

    def reset(self):
        # Reset grid and simulation state
        self.net = build_microgrid()
        self.net.ext_grid.at[0, "in_service"] = False # islanded by default
        pp.runpp(self.net) # Run power flow
        self.current_step = 0
        return self._get_state().astype(np.float32)

    def step(self, action):
        # Apply action to the target switch
        if action == 1:
            self.net.switch.at[self.switch_idx, 'closed'] = False
        elif action == 2:
            self.net.switch.at[self.switch_idx, 'closed'] = True

        # Recalculate power flow after topology change
        pp.runpp(self.net)

        state = self._get_state().astype(np.float32)
        done = self._check_done()
        self.current_step += 1
        
        # Placeholder reward, to be replaced by detailed function
        info = {}
        return state, 0.0, done, info

    def _get_state(self):

        # Extract voltages, load values, and switch states into a single vector
        voltages = self.net.res_bus.vm_pu.values
        load_powers = self.net.res_load.p_mw.values
        switch_states = self.net.switch.closed.values.astype(float)
        return np.concatenate([voltages, load_powers, switch_states])

    def _check_done(self):

        # Terminate after fixed number of steps
        return self.current_step >= self.max_steps

    def render(self, mode='human'):

        # Visualises the microgrid with topology and overlay symbols
        if self.net.bus_geodata.empty:
            raise ValueError("Missing bus_geodata. Define coordinates in builder.py.")

        # Draw power lines as background network lines
        line_trace = create_line_trace(self.net, width=2)

        # Basic bus info for voltage and index display
        bus_coords = self.net.bus_geodata
        bus_voltages = self.net.res_bus.vm_pu

        hover_text = []
        for idx in self.net.bus.index:
            name = self.net.bus.at[idx, 'name']
            v = bus_voltages.at[idx] if idx in bus_voltages.index else 'N/A'
            hover_text.append(f"Bus {idx} ({name})<br>Voltage: {v:.3f} pu")

        bus_trace = go.Scatter(
            x=bus_coords['x'],
            y=bus_coords['y'],
            mode='markers+text',
            marker=dict(size=10, color='blue'),
            text=[str(i) for i in self.net.bus.index],
            textposition='top center',
            hovertext=hover_text,
            hoverinfo='text',
            name='Bus Index'
        )

        fig = go.Figure(data=line_trace + [bus_trace])

        # Draw switch symbols between connected buses
        for _, sw in self.net.switch.iterrows():
            line = self.net.line.loc[sw['element']]
            fb, tb = line['from_bus'], line['to_bus']
            x1, y1 = self.net.bus_geodata.at[fb, 'x'], self.net.bus_geodata.at[fb, 'y']
            x2, y2 = self.net.bus_geodata.at[tb, 'x'], self.net.bus_geodata.at[tb, 'y']
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            fig.add_trace(go.Scatter(x=[xm], y=[ym], mode='markers+text',
                                     marker=dict(size=10, color='red', symbol='x'),
                                     text=[sw['name']],
                                     textposition="bottom center",
                                     showlegend=False))
        
        # Track which legend labels have been added to avoid duplicates
        shown_labels = set()

        # Helper function for placing role-specific icons on nodes
        def add_marker(bus_idx, symbol, color, label):
            x = self.net.bus_geodata.at[bus_idx, 'x']
            y = self.net.bus_geodata.at[bus_idx, 'y']
            show_legend = label not in shown_labels
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=18, color=color, symbol=symbol),
                name=label,
                hovertext=[f"{label} at Bus {bus_idx}"],
                hoverinfo='text',
                showlegend=show_legend
            ))
            shown_labels.add(label)

        # Annotate all load nodes by priority level
        for i, row in self.net.load.iterrows():
            bus = row['bus']
            priority = row['priority']
            if priority == 3:
                add_marker(bus, 'triangle-up', 'red', 'Critical Load')
            elif priority == 2:
                add_marker(bus, 'diamond', 'orange', 'Important Load')
            elif priority == 1:
                add_marker(bus, 'circle', 'gray', 'Non-Critical Load')

        # Add icons for distributed generators
        for i, row in self.net.sgen.iterrows():
            add_marker(row['bus'], 'star', 'yellow', 'DER')

        # Add symbol for black-start generator
        for i, row in self.net.gen.iterrows():
            add_marker(row['bus'], 'x', 'black', 'Blackstart Gen')

        # Add battery icons at storage buses
        for i, row in self.net.storage.iterrows():
            add_marker(row['bus'], 'square', 'green', 'Battery')

        # Final layout
        fig.update_layout(title="Interactive Microgrid Layout with Hover Info",
                          xaxis=dict(visible=False),
                          yaxis=dict(visible=False),
                          plot_bgcolor='white')
        fig.show()
