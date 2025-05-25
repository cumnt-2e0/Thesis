import pandapower as pp
import pandas as pd

def build_microgrid():
    net = pp.create_empty_network()

    # --- Buses ---
    b0 = pp.create_bus(net, vn_kv=0.4, name="Slack (B0)")
    b1 = pp.create_bus(net, vn_kv=0.4, name="DER1 (B1)")
    b2 = pp.create_bus(net, vn_kv=0.4, name="Mid (B2)")
    b3 = pp.create_bus(net, vn_kv=0.4, name="Critical Load 1 (B3)")
    b4 = pp.create_bus(net, vn_kv=0.4, name="Non-Critical Load 1 (B4)")
    b5 = pp.create_bus(net, vn_kv=0.4, name="DER2 (B5)")
    b6 = pp.create_bus(net, vn_kv=0.4, name="Critical Load 2 (B6)")
    b7 = pp.create_bus(net, vn_kv=0.4, name="Non-Critical Load 2 (B7)")
    b8 = pp.create_bus(net, vn_kv=0.4, name="DER3 (B8)")

    # --- Slack ---
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, name="Slack")

    # --- Lines ---
    lines = [
        (b0, b1), (b1, b2), (b2, b3), (b2, b4),
        (b4, b5), (b1, b6), (b6, b7), (b7, b8)
    ]
    for i, (fb, tb) in enumerate(lines):
        pp.create_line_from_parameters(net, fb, tb, 0.1, 0.32, 0.08, 210, 0.14, name=f"Line {i}")

    # --- Switches (on lines 1, 4, 6) ---
    pp.create_switch(net, bus=b1, element=1, et="l", closed=True, name="SW1")  # B1-B2
    pp.create_switch(net, bus=b4, element=4, et="l", closed=True, name="SW2")  # B4-B5
    pp.create_switch(net, bus=b6, element=6, et="l", closed=True, name="SW3")  # B6-B7

    # --- DERs ---
    pp.create_gen(net, bus=b1, p_mw=0.05, vm_pu=1.0, slack=True, name="MG1 - Black Start")
    pp.create_sgen(net, bus=b5, p_mw=0.03, name="MG2 - Solar")
    pp.create_sgen(net, bus=b8, p_mw=0.02, name="MG3 - Battery")

    # --- Loads ---
    pp.create_load(net, bus=b3, p_mw=0.02, name="Critical Load 1")
    pp.create_load(net, bus=b6, p_mw=0.015, name="Critical Load 2")
    pp.create_load(net, bus=b4, p_mw=0.02, name="Non-Critical Load 1")
    pp.create_load(net, bus=b7, p_mw=0.01, name="Non-Critical Load 2")
    net.load["priority"] = [1, 1, 0, 0]

    # --- Geodata for Layout ---
    coords = {
        b0: (0, 0), b1: (1, 0), b2: (2, 0), b3: (3, 1), b4: (3, -1),
        b5: (4, -1), b6: (1, 1), b7: (2, 1), b8: (3, 2)
    }
    net.bus_geodata = pd.DataFrame.from_dict({k: {"x": x, "y": y} for k, (x, y) in coords.items()}, orient="index")

    return net
