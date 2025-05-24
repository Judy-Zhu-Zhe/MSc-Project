import random

from network import Device, Enclave, Segmentation
from simulation import Simulation
from optimization import MapElitesConfig, map_elites, topology_neighbours
from visualization import plot_behavior_map, draw_segmentation_topology

N_ENCLAVES = 5 # Number of enclaves in the network
DYN_DEVICES = 10 # Number of low value devices in the network
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

def generate_devices(n):
    devices = [Device(f"Employee computer {i}", device_type="Employee computer") for i in range(n)]
    for t in DEVICE_TYPES:
        if t != "Employee computer" and t != "Low value device":
            devices.append(Device(f"{t}", device_type=t))
    return devices

CONFIG_1 = MapElitesConfig(
    init_batch = 400,
    batch_size = 1,
    devices = generate_devices(10),
    sensitivities = [random.uniform(0.2, 0.8) for _ in range(N_ENCLAVES)],
    vulnerabilities = [1, 0.5, 0.6, 0.8, 0.7],
    generations = 30,
    bins = [1, 2, 1, 1, 1, 10],
    n_simulations = 20,
    total_sim_time = 24,
    times_in_enclaves = [0, 6, 6, 6, 6],
    descriptors = ["nb_high_deg_nodes", "std_devices"],
    p_update = P_UPDATE,
    p_network_error = P_NETWORK_ERROR,
    p_device_error = P_DEVICE_ERROR,
    r_reconnaissance = R_RECONNAISSANCE,
    n_low_value_device = DYN_DEVICES,
    c_appetite = C_APPETITE,
    i_appetite = I_APPETITE,
    beta = BETA,
    eta = 10,
    metric_weights = [1, 0, 0]
)

CONFIG = MapElitesConfig(
    init_batch = 100,
    batch_size = 1,
    devices = generate_devices(20),
    sensitivities = [random.uniform(0.2, 0.8) for _ in range(N_ENCLAVES)],
    vulnerabilities = [1, 0.5, 0.6, 0.8, 0.7],
    generations = 10000,
    bins = [1, 2, 1, 1, 1, 10],
    n_simulations = 20,
    total_sim_time = 24,
    times_in_enclaves = [0, 6, 6, 6, 6],
    descriptors = ["nb_high_deg_nodes", "std_devices"],
    p_update = P_UPDATE,
    p_network_error = P_NETWORK_ERROR,
    p_device_error = P_DEVICE_ERROR,
    r_reconnaissance = R_RECONNAISSANCE,
    n_low_value_device = DYN_DEVICES,
    c_appetite = C_APPETITE,
    i_appetite = I_APPETITE,
    beta = BETA,
    eta = 10,
    metric_weights = [1, 0, 0]
)