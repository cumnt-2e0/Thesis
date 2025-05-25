import pandapower as pp
import pandas as pd
import numpy as np
from .constants import LOAD_PRIORITIES

def build_microgrid():
    # Create an empty pandapower network
    net = pp.create_empty_network()

    # Create 16 buses with nominal voltage of 0.4kV and store in dictionary
    buses = {}
    for i in range(16):
        buses[i] = pp.create_bus(net, vn_kv=0.4, name=f"Bus {i}")

    # Slack bus represents connection to the grid
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0, name="Slack Bus")

    # Physical line connections between buses
    # each line has a resistance, reactance, and max current rating
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (1, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (10, 12), (12, 13), (12, 14), (9, 15)
    ]
    for idx, (from_bus, to_bus) in enumerate(lines):
        pp.create_line_from_parameters(
            net, buses[from_bus], buses[to_bus], length_km=0.1,
            r_ohm_per_km=0.32, x_ohm_per_km=0.08, c_nf_per_km=0,
            max_i_ka=0.14, name=f"Line {idx}"
        )

    # Switches placed on lines for reconfiguration
    switches = [(1, 1), (3, 2), (5, 4), (6, 5), (7, 6), (8, 7), (12, 13), (9, 14)]
    for i, (bus, element) in enumerate(switches):
        pp.create_switch(net, bus=buses[bus], element=element, et="l", closed=True, name=f"SW{i+1}")

    # static generators (sgen) representing distributed energy resources
    # produce power into the system
    pp.create_sgen(net, bus=buses[1], p_mw=0.03, name="DER1 - Solar")
    pp.create_sgen(net, bus=buses[5], p_mw=0.02, name="DER2 - Wind")
    pp.create_sgen(net, bus=buses[13], p_mw=0.025, name="DER3 - CHP")

    # Blackstart genset for restoring power after outages
    # later exploration can occur on required capacity and characteristics
    pp.create_gen(net, bus=buses[15], p_mw=0.04, vm_pu=1.0, slack=True, name="Blackstart GenSet")

    # battery storage systems with max capacity and state of charge
    pp.create_storage(net, bus=buses[6], p_mw=0.01, max_e_mwh=0.05, soc_percent=50, name="Battery 1")
    pp.create_storage(net, bus=buses[11], p_mw=0.015, max_e_mwh=0.06, soc_percent=60, name="Battery 2")

    # Attaches real loads to specific buses
    # each one is tagged with a priority level (3,2,1) = (critical, important, non-critical)
    for bus, priority in LOAD_PRIORITIES.items():
        pp.create_load(net, bus=buses[bus], p_mw=0.02, name=f"Load Bus {bus}")

    # Store priorities in net.load DataFrame
    net.load["priority"] = [LOAD_PRIORITIES[bus] for bus in net.load.bus]
    
    coords = {
        0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 1), 4: (4, 1),
        5: (2, -1), 6: (3, -1), 7: (1, 1), 8: (2, 2), 9: (0, 1),
        10: (1, 2), 11: (2, 3), 12: (3, 3), 13: (4, 3), 14: (3, 4), 15: (-1, 0)
    }

    # Ensure proper DataFrame with correct dtypes and labels
    bus_geodata_df = pd.DataFrame(coords).T
    bus_geodata_df.columns = ["x", "y"]
    bus_geodata_df.index.name = "bus"

    # Assign it to the net object
    net.bus_geodata = bus_geodata_df.astype(np.float64)

    # return the constructed pandapower network
    return net
