from dataclasses import dataclass, field
from graphillion import GraphSet
from typing import List, Dict, Tuple
import os
import yaml
import json
from datetime import datetime

from network import Device, Segmentation

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
    "DNS server",
    "DHCP server",
    "E-mail server",
    "Web server",
    "SQL Database",
    "Syslog server",
    "Authentication server",
    "Low value device",
]

def generate_universe(n_enclaves: int, constraints: Dict[str, int]) -> GraphSet:
    # Generate all possible topologies
    universe = [(i, j) for i in range(n_enclaves) for j in range(n_enclaves) if i < j]
    GraphSet.set_universe(universe)
    degree_constraints = {}

    # Constraint: Maximum number of ISPs
    if "max_isps" in constraints.keys():
        degree_constraints[0] = range(1, constraints["max_isps"] + 1)  # Maximum ISPs constraint

    # Constraint: Maximum degree for other enclaves
    max_deg = constraints.get("max_degree", n_enclaves - 1)
    for i in range(1, n_enclaves):
        degree_constraints[i] = range(1, max_deg + 1)

    return GraphSet.graphs(vertex_groups=[range(1, n_enclaves)], degree_constraints=degree_constraints)

def generate_devices(n_devices: Dict[str, int]) -> List[Device]:
    """Generate a list of devices based on the provided device counts dictionary."""
    devices = []
    for device_type, count in n_devices.items():
        if count == 1:
            devices.append(Device(name=device_type, device_type=device_type))
        else:
            for i in range(count):
                name = f"{device_type} {i}"
                devices.append(Device(name, device_type=device_type))
    return devices

@dataclass
class MapElitesConfig:
    name: str = "Default Configuration"
    universe: GraphSet = field(default_factory=lambda: generate_universe(N_ENCLAVES, {}))
    n_enclaves: int = 5
    init_batch: int = 200
    batch_size: int = 100
    n_low_value_device: int = DYN_DEVICES
    devices: List = field(default_factory=lambda: generate_devices({"Employee computer": 5}))
    sensitivities: List[float] = field(default_factory=lambda: [0.5] * 5)
    vulnerabilities: List[float] = field(default_factory=lambda: [1, 0.5, 0.6, 0.8, 0.7])
    generations: int = 20
    n_simulations: int = 30
    total_sim_time: int = 24
    times_in_enclaves: List[int] = field(default_factory=lambda: [0, 6, 6, 6, 6])
    descriptors: List[str] = field(default_factory=lambda: ["nb_high_deg_nodes", "std_devices"])
    p_update: float = P_UPDATE
    p_network_error: float = P_NETWORK_ERROR
    p_device_error: float = P_DEVICE_ERROR
    r_reconnaissance: float = R_RECONNAISSANCE
    c_appetite: float = C_APPETITE
    i_appetite: float = I_APPETITE
    k_to_infect: int = BETA
    eta: float = 10.0
    metric_weights: List[float] = field(default_factory=lambda: [1, 0, 0, 0])

def define_config(**overrides) -> MapElitesConfig:
    return MapElitesConfig(**overrides)

def load_config_from_yaml(yaml_path: str) -> MapElitesConfig:
    """Load a configuration from a YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if "constraints" in data.keys():
        # Generate the universe if it is specified in the YAML
        data["universe"] = generate_universe(data.get("n_enclaves", 5), data.get("constraints", {}))
        print(data["universe"])
        data.pop("constraints")

    if "n_devices" in data.keys():
        # Generate devices if specified in the YAML
        data["devices"] = generate_devices(data.get("n_devices", {}))
        data.pop("n_devices")

    if "time_in_enclave" in data.keys():
        # Generate list of times in enclaves
        data["times_in_enclaves"] = [0] + [data["time_in_enclave"]] * (data.get("n_enclaves", 5) - 1)
        data.pop("time_in_enclave")

    return MapElitesConfig(**data)

def update_config_for_adptation(config: MapElitesConfig) -> MapElitesConfig:
    """Update the configuration for adaptation."""
    config.name += "_adaptation"
    assert len(config.metric_weights) == 3, "Metric weights should have 3 elements for adaptation."
    config.metric_weights.append(200)  # Add weight for "dissimilarity"
    return config
    
    
def generate_filename(config: MapElitesConfig) -> str:
    """Generate a filename based on the configuration and current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.name}_batch{config.batch_size}_gen{config.generations}_{timestamp}"

def save_results(seg: Segmentation, grid: Dict[Tuple, Tuple[Segmentation, float]], config: str, filename: str):
    """Save the segmentation and grid results to JSON files."""
    dir_path = f"segmentations/{config}"
    os.makedirs(dir_path, exist_ok=True) # Ensure directory exists

    # Save segmentation
    seg_path = f"{dir_path}/seg_{filename}.json"
    with open(seg_path, "w") as f:
        json.dump(seg.to_dict(), f, indent=2)
    print(f"Saved segmentation to {seg_path}")

    # Save grid
    serializable_grid = {
        str(key): {
            "fitness": fitness,
            "segmentation": segmentation.to_dict()
        }
        for key, (segmentation, fitness) in grid.items()
    }
    grid_path = f"{dir_path}/grid_{filename}.json"
    with open(grid_path, "w") as f:
        json.dump(serializable_grid, f, indent=2)
    print(f"Saved grid to {grid_path}")

def load_segmentation(filename: str) -> Segmentation:
    """Load a segmentation from a JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)
    return Segmentation.from_dict(data)

def load_grid(filename: str) -> Dict[Tuple, Tuple[Segmentation, float]]:
    """Load a grid from a JSON file."""
    from_dict = Segmentation.from_dict
    with open(filename, "r") as f:
        data = json.load(f)
    grid = {
        tuple(map(float, key.strip("()").split(","))): (from_dict(value["segmentation"]), value["fitness"])
        for key, value in data.items()
    }
    return grid

