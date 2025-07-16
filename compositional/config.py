from dataclasses import dataclass, field
import networkx as nx
from typing import List, Dict, Tuple, Optional
import os
import yaml
import json
from datetime import datetime
from itertools import combinations

from network import Device, Segmentation, SegmentationNode

N_ENCLAVES = 5 # Number of enclaves in the network
P_UPDATE = 1/90  # Probability of successful update
R_RECONNAISSANCE = 0.7  # Reconnaissance rate
C_APPETITE = 0.9  # Compromise appetite
I_APPETITE = 0.4  # Infection appetite
BETA = 2

def generate_universe(n_enclaves: int, constraints: Dict[str, int] = {}) -> List[List[Tuple[int, int]]]:
    """
    Generate all valid undirected graph topologies satisfying:
    - Nodes 1 to n-1 form a connected graph (not using node 0)
    - Node 0 is connected to at least one internal node (as the Internet)
    - Degree constraints on node 0 and other nodes
    """
    nodes = list(range(n_enclaves))  # node 0 = Internet
    internal_nodes = list(range(1, n_enclaves))
    internal_edges = list(combinations(internal_nodes, 2))
    internet_edges = [(0, i) for i in internal_nodes]

    max_isps = constraints.get("max_isps", n_enclaves - 1)
    max_degree = constraints.get("max_degree", n_enclaves - 1)

    valid_graphs = []

    # Enumerate all internal connected graphs
    for i in range(2 ** len(internal_edges)):
        bits = bin(i)[2:].zfill(len(internal_edges))
        internal_graph_edges = [e for b, e in zip(bits, internal_edges) if b == '1']

        G_internal = nx.Graph()
        G_internal.add_nodes_from(internal_nodes)
        G_internal.add_edges_from(internal_graph_edges)

        # Skip if not connected
        if not nx.is_connected(G_internal):
            continue

        # Now try adding internet connections
        for j in range(1, 2 ** len(internet_edges)):
            bits2 = bin(j)[2:].zfill(len(internet_edges))
            internet_graph_edges = [e for b, e in zip(bits2, internet_edges) if b == '1']

            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(internal_graph_edges)
            G.add_edges_from(internet_graph_edges)

            # Check degree constraints
            node_0_degree = len(list(G.neighbors(0)))
            if node_0_degree > max_isps:
                continue
            if any(len(list(G.neighbors(i))) > max_degree for i in internal_nodes):
                continue

            # Skip if not fully connected
            if not nx.is_connected(G):
                continue

            valid_graphs.append(list(G.edges()))

    return valid_graphs

def generate_devices(n_devices: Dict[str, int], device_profiles: Dict[str, Dict]) -> List[Device]:
    """Generate a list of devices based on the provided device counts dictionary."""
    devices = []
    for device_type, count in n_devices.items():
        if device_type not in device_profiles:
            raise ValueError(f"Device type '{device_type}' not found in device profiles.\nAvailable types: {list(device_profiles.keys())}")
        else:
            profile = device_profiles[device_type]
        if count == 1:
            devices.append(Device(name=device_type, device_type=device_type, profile=profile))
        else:
            for i in range(count):
                name = f"{device_type} {i+1}"
                devices.append(Device(name, device_type=device_type, profile=profile))
    return devices

@dataclass
class MapElitesConfig:
    name: str = "map_elites"
    level: int = 0
    devices: List[Device] = field(default_factory=list)
    universe: List[List[Tuple[int, int]]] = field(default_factory=lambda: generate_universe(N_ENCLAVES, {}))
    n_enclaves: int = N_ENCLAVES
    init_batch: int = 200
    batch_size: int = 100
    generations: int = 20
    sensitivities: List[float] = field(default_factory=lambda: [0.5] * N_ENCLAVES)
    n_simulations: int = 30
    total_sim_time: int = 24
    time_in_enclaves: int = 6
    descriptors: List[str] = field(default_factory=lambda: ["nb_high_deg_nodes", "std_devices"])
    evaluation_metrics: Dict[str, float] = field(default_factory=lambda: {"security_loss": 1.0})
    p_update: float = P_UPDATE
    r_reconnaissance: float = R_RECONNAISSANCE
    c_appetite: float = C_APPETITE
    i_appetite: float = I_APPETITE

@dataclass
class CompositionalConfig:
    name: str
    network_name: str
    configs: Dict[str, List[MapElitesConfig]]
    devices: Dict[str, List[Device]]

class ConfigManager:
    """Manages configuration loading and handling for both MapElites and Compositional configs."""
    
    def __init__(self, device_profile_path, network_path: str, config_path: str):
        self.devices = {}
        self._load_device_profiles(device_profile_path)
        self._load_network(network_path)
        self._load_config(config_path)
        self.filename = self.generate_filename()
    
    def _load_device_profiles(self, device_profile_path):
        """Load device profiles from YAML file."""
        with open(device_profile_path, "r") as f:
            data = yaml.safe_load(f)
        self.device_profiles = data["device_profiles"]
    
    def _load_network(self, network_path: str):
        """Load the network from a YAML file."""
        with open(network_path, "r") as f:
            data = yaml.safe_load(f)
        self.network_name = data["name"]
        self.network = generate_devices(data["devices"], self.device_profiles)

    def _load_config(self, yaml_path: str):
        """Load a compositional configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        name = data.get("name", "Default")
        data.pop("name")

        configs_dict = {}
        for trust_level, configs in data.items():
            self.devices[trust_level] = [device for device in self.network if device.trust_level == trust_level]
            configs_dict[trust_level] = [self.parse_config(config, self.devices[trust_level]) for config in configs]

        self.config = CompositionalConfig(name, self.network_name, configs_dict, self.devices)

    def parse_config(self, config: Dict, devices: Optional[List[Device]] = None) -> MapElitesConfig:
        """Parse a configuration dictionary into a MapElitesConfig object."""
        config = dict(config)  # Copy so we can modify

        # Generate universe if constraints provided
        n_enclaves = config.get("n_enclaves", 5)
        if "constraints" in config:
            constraints = config.pop("constraints")
            config["universe"] = generate_universe(n_enclaves, constraints)
        else:
            config["universe"] = generate_universe(n_enclaves, {})

        config["devices"] = devices

        return MapElitesConfig(**config)
    
    def generate_filename(self) -> str:
        """Generate a filename based on the configuration and current timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.name}_{self.config.network_name}_{timestamp}"

    def update_config_for_adaptation(self, config: MapElitesConfig) -> MapElitesConfig:
        # TODO: for adaptation
        """Update the configuration for adaptation."""
        config.name += "_adaptation"
        return config
    
    def save_results(self, seg_node: SegmentationNode, grid: Optional[Dict] = None) -> str:
        """Save the segmentation and grid results to JSON files."""
        dir_path = f"compositional/segmentations/{self.config.name}"
        os.makedirs(dir_path, exist_ok=True) # Ensure directory exists

        # Save segmentation
        seg_path = f"{dir_path}/seg_{self.filename}.json"
        with open(seg_path, "w") as f:
            json.dump(seg_node.to_dict(), f, indent=2)
        print(f"Saved segmentation to {seg_path}")

        # Save grid
        if grid:
            serializable_grid = {
                str(tuple(round(k, 2) for k in key)): {
                    "fitness": fitness,
                    "segmentation": seg_node.to_dict()
                }
                for key, (seg_node, fitness) in grid.items()
            }
            grid_path = f"{dir_path}/grid_{self.filename}.json"
            with open(grid_path, "w") as f:
                json.dump(serializable_grid, f, indent=2)
            print(f"Saved grid to {grid_path}")

        return self.filename
    
    def load_segmentation_node(self, filename: str) -> SegmentationNode:
        """Load a segmentation from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)
        return SegmentationNode.from_dict(data)

