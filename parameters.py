import random

from network import Device, Enclave, Segmentation
from simulation import Simulation
from optimization import MapElitesConfig, map_elites, topology_neighbours
from visualization import plot_behavior_map, draw_segmentation_topology

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

def generate_devices(n):
    special_devices = [t for t in DEVICE_TYPES if t != "Employee computer" and t != "Low value device"]
    devices = [Device(f"Employee computer {i}", device_type="Employee computer") for i in range(n-len(special_devices))]
    for t in special_devices:
        devices.append(Device(f"{t}", device_type=t))
    return devices

def modify_config(config: MapElitesConfig, init_batch: int, batch_size: int, generations: int):
    config.init_batch = init_batch
    config.batch_size = batch_size
    config.generations = generations
    return config

EXP_CONFIG = MapElitesConfig(
    n_enclaves = N_ENCLAVES,
    init_batch = 10,
    batch_size = 1,
    devices = generate_devices(10),
    sensitivities = [random.uniform(0, 1) for _ in range(N_ENCLAVES)],
    vulnerabilities = [1, 0.5, 0.6, 0.8, 0.7],
    generations = 10,
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

CONFIG_1 = modify_config(EXP_CONFIG, init_batch=400, batch_size=400, generations=30)
CONFIG_2 = modify_config(EXP_CONFIG, init_batch=10000, batch_size=4000, generations=100)

MY_CONFIG = modify_config(EXP_CONFIG, init_batch=10, batch_size=1, generations=100)

# CONFIG = MapElitesConfig(
#     init_batch = 100,
#     batch_size = 1,
#     devices = generate_devices(20),
#     sensitivities = [random.uniform(0.2, 0.8) for _ in range(N_ENCLAVES)],
#     vulnerabilities = [1, 0.5, 0.6, 0.8, 0.7],
#     generations = 10000,
#     n_simulations = 20,
#     total_sim_time = 24,
#     times_in_enclaves = [0, 6, 6, 6, 6],
#     descriptors = ["nb_high_deg_nodes", "std_devices"],
#     p_update = P_UPDATE,
#     p_network_error = P_NETWORK_ERROR,
#     p_device_error = P_DEVICE_ERROR,
#     r_reconnaissance = R_RECONNAISSANCE,
#     n_low_value_device = DYN_DEVICES,
#     c_appetite = C_APPETITE,
#     i_appetite = I_APPETITE,
#     beta = BETA,
#     eta = 10,
#     metric_weights = [1, 0, 0]
# )