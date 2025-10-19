import pandapower.networks as pn

_FEEDER_BUILDERS = {
    "case33":   lambda: pn.case33bw(),          # 33-bus distribution (RECOMMENDED)
    "case145":  lambda: pn.case145(),           # 145-bus transmission
    "case118":  lambda: pn.case118(),           # 118-bus IEEE (coming soon)
    "case300":  lambda: pn.case300(),           # 300-bus transmission
}

def get_feeder(name: str):
    key = name.lower()
    try:
        return _FEEDER_BUILDERS[key]
    except KeyError:
        raise KeyError(f"Unknown feeder '{name}'. Options: {list(_FEEDER_BUILDERS.keys())}")