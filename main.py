from graphillion import GraphSet
import random
import time

from network import Device, Enclave, Segmentation
from simulation import Simulation
from optimization import MapElitesConfig, map_elites, topology_neighbours
from visualization import plot_behavior_map, draw_segmentation_topology

N_ENCLAVES = 5 # Number of enclaves in the network
DYN_DEVICES = 3 # Number of low value devices in the network
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


# CONFIG1 = MapElitesConfig(
#     devices=[Device(f"Device {i}", "high_value") for i in range(10)],
#     sensitivities = None,
#     vulnerabilities = [1, 0.5, 0.6, 0.8, 0.7],
#     generations = 20,
#     bins = [1, 1, 1, 1, 12],
#     n_simulations = 10,
#     total_sim_time = 24,
#     times_in_enclaves = [1, 1, 1, 1, 1], #TODO: check
#     descriptors = ["nb_high_deg_nodes", "std_devices"],
#     p_update = P_UPDATE,
#     p_network_error = P_NETWORK_ERROR,
#     p_device_error = P_DEVICE_ERROR,
#     r_reconnaissance = R_RECONNAISSANCE,
#     n_low_value_device = DYN_DEVICES,
#     c_appetite = C_APPETITE,
#     i_appetite = I_APPETITE,
#     beta = BETA,
#     metric_weights = [1, 0, 0]
# )


CONFIG1 = MapElitesConfig(
    devices = generate_devices(10),
    sensitivities = [random.uniform(0.2, 0.8) for _ in range(N_ENCLAVES)],
    vulnerabilities = [1, 0.5, 0.6, 0.8, 0.7],
    generations = 50,
    bins = [1, 2, 1, 1, 1, 10],
    n_simulations = 10,
    total_sim_time = 12,
    times_in_enclaves = [0, 5, 5, 5, 5],
    descriptors = ["nb_high_deg_nodes", "std_devices"],
    p_update = P_UPDATE,
    p_network_error = P_NETWORK_ERROR,
    p_device_error = P_DEVICE_ERROR,
    r_reconnaissance = R_RECONNAISSANCE,
    n_low_value_device = DYN_DEVICES,
    c_appetite = C_APPETITE,
    i_appetite = I_APPETITE,
    beta = BETA,
    metric_weights = [1, 0, 0]
)

def main():
    # Generate all possible topologies
    universe = [(i, j) for i in range(N_ENCLAVES) for j in range(N_ENCLAVES) if i < j]
    GraphSet.set_universe(universe)
    degree_constraints = {i: [1, 2] if i == 0 else range(1, N_ENCLAVES) for i in range(N_ENCLAVES)} # Maximum 2 ISPs
    graphs = GraphSet.graphs(vertex_groups=[range(N_ENCLAVES)], degree_constraints=degree_constraints)

    # Run MAP-Elites for optimization
    start_time = time.time()
    topology_list, neighbours_table, distances_table = topology_neighbours(graphs, N_ENCLAVES, K=5)
    seg, fitness = map_elites(topology_list, neighbours_table, distances_table, CONFIG1)
    end_time = time.time()
    elapsed = end_time - start_time

    # Visualize the results
    # print("Archive size:", len(archive))
    # print("Archive:", archive)
    # plot_behavior_map(archive)
    # for k in archive.values():
    #     draw_segmentation_topology(k[0])
    #     print(k[0].topology.adj_matrix)
    #     print("Fitness: ", k[1])

    print("\nNumber of graphs:", len(graphs))
    print("Graphs:", graphs)
    print("Results:")
    print(f"Execution time for MAP-Elite: {elapsed:.2f} seconds")
    print(f"Fitness: {fitness}")
    for e in seg.enclaves:
        if e.id == 0:
            print(e)
            continue
        print(f"Enclave {e.id}")
        print(f"  Sensitivity: {e.sensitivity:.2f}")
        print(f"  Vulnerability: {e.vulnerability:.2f}")
        print(f"  Devices: {[d.name for d in e.devices]}")
    draw_segmentation_topology(seg)

if __name__ == "__main__":
    main()
