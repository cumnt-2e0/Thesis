# env/builder.py

# This module defines the structure of a custom 18-bus microgrid using the pandapower library.
# It includes slack sources, DERs, batteries, and loads (with priority annotations), along with
# logical switch placement for load shedding and resilience testing. Coordinates and GeoJSON are
# prepared for Plotly-based visualisation.

import pandapower as pp
import pandas as pd
import numpy as np
import geojson
from .constants import LOAD_PRIORITIES

def build_microgrid():

    #Initialise empty power network
    net = pp.create_empty_network()

    # Create 18 Bus network
    # These represent physical or virtual locations on the microgrid 
    buses = {}
    for i in range(18):
        buses[i] = pp.create_bus(net, vn_kv=0.4, name=f"Bus {i}")

    # Slack and Generator Buses
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0, name="Slack Bus")
    pp.create_gen(net, bus=buses[15], p_mw=0.04, vm_pu=1.0, slack=True, name="Blackstart GenSet")

    # Define Lines Between Buses
    # Each tuple defines a line connecting from_bus -> to_bus with standard impedance values.
    lines = [
        (0, 1),   # Line 0
        (1, 2),   # Line 1
        (2, 3),   # Line 2
        (2, 4),   # Line 3
        (2, 5),   # Line 4
        (5, 6),   # Line 5
        (6, 7),   # Line 6
        (7, 8),   # Line 7
        (0, 9),   # Line 8
        (9, 10),  # Line 9
        (10, 11), # Line10
        (10, 12), # Line11
        (13, 16), # Line12 
        (11, 17), # Line13 
        (16, 14), # Line14
        (17, 14), # Line15
        (0, 15)   # Line16
    ]
    for idx, (from_bus, to_bus) in enumerate(lines):
        pp.create_line_from_parameters(
            net, buses[from_bus], buses[to_bus], length_km=0.1,
            r_ohm_per_km=0.32, x_ohm_per_km=0.08, c_nf_per_km=0,
            max_i_ka=0.14, name=f"Line {idx}"
        )

    # Switches
    # Switches can open or clines lines, dynamically during simulation.
    switches = [
        (0, 0),    # SW1
        (2, 2),    # SW2
        (2, 3),    # SW3
        (2, 4),    # SW4
        (13, 12),  # SW5: on line to Bus 16
        (11, 13),  # SW6: on line to Bus 17
        (0, 16)    # SW7
    ]
    for i, (bus, line_idx) in enumerate(switches):
        pp.create_switch(net, bus=buses[bus], element=line_idx, et="l", closed=True, type="LBS", name=f"SW{i+1}")

    # Distributed Energy Resources (DERs) or system generators (sgen)
    pp.create_sgen(net, bus=buses[1], p_mw=0.03, name="DER1 - Solar")
    pp.create_sgen(net, bus=buses[12], p_mw=0.025, name="DER2 - CHP")

    # Battery Storage Units
    pp.create_storage(net, bus=buses[5], p_mw=0.01, max_e_mwh=0.05, soc_percent=50, name="Battery 1")
    pp.create_storage(net, bus=buses[11], p_mw=0.015, max_e_mwh=0.06, soc_percent=60, name="Battery 2")

    # Loads with assigned priorities
    # These priorities are used to assess system resilience (3 = critical, 1 = lowest priority).
    for bus, priority in LOAD_PRIORITIES.items():
        pp.create_load(net, bus=buses[bus], p_mw=0.02, name=f"Load Bus {bus}")

    net.load["priority"] = [LOAD_PRIORITIES[bus] for bus in net.load.bus]

    # Define 2D Coordinates for Plotting
    coords = {
        0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 1), 4: (3, -1),
        5: (2, -1.5), 6: (3, -1.5), 7: (4, -1.5), 8: (5, -1.5),
        9: (1, 1), 10: (2, 1), 11: (3, 1.5), 12: (3, 2),
        13: (4, 2), 14: (5, 2), 15: (-1, 0),
        16: (4.5, 2), 17: (4.5, 1.5)  # Virtual switch nodes
    }
    bus_geodata_df = pd.DataFrame.from_dict(coords, orient="index", columns=["x", "y"])
    net.bus_geodata = bus_geodata_df.astype(np.float64)

    # GeoJSON formatting for Plotly
    net.bus["geo"] = None
    for idx in net.bus.index:
        if idx in net.bus_geodata.index:
            x = net.bus_geodata.at[idx, "x"]
            y = net.bus_geodata.at[idx, "y"]
            net.bus.at[idx, "geo"] = geojson.dumps(geojson.Point((x, y)))

    return net