from network import Device
from dataclasses import dataclass, field
from graphillion import GraphSet
from typing import List, Dict

N_ENCLAVES = 5 # Number of enclaves in the network
DYN_DEVICES = 20 # Number of low value devices in the network
P_UPDATE = 1/90  # Probability of successful update
P_NETWORK_ERROR = 0.7  # Probability of network error
P_DEVICE_ERROR = 0.7  # Probability of device error
R_RECONNAISSANCE = 0.7  # Reconnaissance rate
C_APPETITE = 0.9  # Compromise appetite
I_APPETITE = 0.4  # Infection appetite
BETA = 2

DEVICE_TYPES = [
    "Printer",
    "Employee computer",
    "Printer server",
    "DNS Server",
    "DHCP Server",
    "E-mail server",
    "Web server",
    "SQL Database",
    "Syslog server",
    "Authentication server",
    "Low value device",
]


def generate_universe(n_enclaves: int, constraints: dict[str, int]) -> GraphSet:
    # Generate all possible topologies
    universe = [(i, j) for i in range(n_enclaves) for j in range(n_enclaves) if i < j]
    GraphSet.set_universe(universe)
    if "max_isps" in constraints.keys():
        degree_constraints = {0: range(1, constraints["max_isps"] + 1)}  # Maximum ISPs constraint
    else:
        degree_constraints = {0: range(1, n_enclaves)}

    if "max_degree" in constraints.keys():
        for i in range(1, n_enclaves):
            degree_constraints[i] = range(1, constraints["max_degree"] + 1)
    else:
        for i in range(1, n_enclaves):
            degree_constraints[i] = range(1, n_enclaves)

    return GraphSet.graphs(vertex_groups=[range(n_enclaves)], degree_constraints=degree_constraints)

def generate_devices(n: int) -> List[Device]:
    special_devices = [t for t in DEVICE_TYPES if t != "Employee computer" and t != "Low value device"]
    devices = [Device(f"Employee computer {i}", device_type="Employee computer") for i in range(n-len(special_devices))]
    for t in special_devices:
        devices.append(Device(t, device_type=t))
    return devices

@dataclass
class MapElitesConfig:
    name: str = "Default Configuration"
    universe: GraphSet = field(default_factory=lambda: generate_universe(N_ENCLAVES, {"max_isps": 2}))
    n_enclaves: int = 5
    init_batch: int = 200
    batch_size: int = 1
    devices: List = field(default_factory=lambda: generate_devices(10))
    sensitivities: List[float] = field(default_factory=lambda: [0.5] * 5)
    vulnerabilities: List[float] = field(default_factory=lambda: [1, 0.5, 0.6, 0.8, 0.7])
    generations: int = 20
    n_simulations: int = 30 # TODO: Can improve
    total_sim_time: int = 24
    times_in_enclaves: List[int] = field(default_factory=lambda: [0, 6, 6, 6, 6])
    descriptors: List[str] = field(default_factory=lambda: ["nb_high_deg_nodes", "std_devices"])
    p_update: float = 0.1
    p_network_error: float = 0.05
    p_device_error: float = 0.02
    r_reconnaissance: float = 0.01
    n_low_value_device: int = 5
    c_appetite: float = 1.0
    i_appetite: float = 1.0
    beta: int = 1
    eta: float = 10.0
    metric_weights: List[float] = field(default_factory=lambda: [1, 0, 0, 0])

def define_config(**overrides) -> MapElitesConfig:
    return MapElitesConfig(**overrides)


CONFIG_1 = define_config(
    name="Exp1",
    init_batch=400,
    batch_size=400,
    generations=30
)

CONFIG_2 = define_config(
    name="Exp2",
    init_batch=10000,
    batch_size=4000,
    generations=100,
    devices=generate_devices(44),
    vulnerabilities=[1, 0.5, 0.6, 0.7, 0.8],
    total_sim_time=132,
    times_in_enclaves=[0, 30, 30, 30, 30],
    descriptors=["nb_high_deg_nodes", "std_devices", "std_web_facing"],
    metric_weights=[1, 0.05, 0.5]
)

CONFIG_ADP = define_config(
    name="Adaptation",
    metric_weights=[1, 0, 0, 200],
    init_batch=400,
    batch_size=400,
    generations=30
)

MY_CONFIG = define_config(
    name="myConfig",
    init_batch=400,
    batch_size=400,
    generations=100
)

