# env/constants.py

# Define categories and priorities for loads
CRITICAL_LOAD_BUSES = [3, 6, 13]
IMPORTANT_LOAD_BUSES = [8, 12]
NON_CRITICAL_LOAD_BUSES = [4, 7, 11, 14]

# Load priority scale: 3 = critical, 2 = important, 1 = non-critical
LOAD_PRIORITIES = {
    3: 3,
    4: 1,
    6: 3,
    7: 1,
    8: 2,
    11: 1,
    12: 2,
    13: 3,
    14: 1
}

# Map switch names to their internal indices in pandapower
SWITCH_IDS = {
    "SW1": 0,
    "SW2": 1,
    "SW3": 2,
    "SW4": 3,
    "SW5": 4,
    "SW6": 5,
    "SW7": 6,
    "SW8": 7
}
