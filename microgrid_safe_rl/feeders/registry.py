import pandapower.networks as pn

_FEEDER_BUILDERS = {
    "case33":   lambda: pn.case33bw(),          # 33-bus distribution
    "case145":  lambda: pn.case145(),           # 145-bus
    "case300":  lambda: pn.case300(),           # 300-bus
    "case1888": lambda: pn.case1888_rte() if hasattr(pn, "case1888_rte") else pn.case1888rte(),
}

def get_feeder(name: str):
    key = name.lower()
    try:
        return _FEEDER_BUILDERS[key]
    except KeyError:
        raise KeyError(f"Unknown feeder '{name}'. Options: {list(_FEEDER_BUILDERS.keys())}")
