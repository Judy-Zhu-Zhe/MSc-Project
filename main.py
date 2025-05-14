import random
from typing import List
from graphillion import GraphSet
import random

from network import Device, Enclave, Segmentation
from simulation import Simulation
from optimization import map_elites, initialization

N_ENCLAVES = 5 # Number of enclaves in the network
N_DEVICES = 55 # Number of devices in the network (1 infected employee computer)
VULNERABILITIES = [0.2, 0.3, 0.5, 0.8]
SIMULATION_TIME = 81 # Simulation time in seconds
MAX_SPREADING_TIME = 30 # Enclave maximum spreading time

DYN_DEVICES = 5 # Number of dynamic low value devices in the network
P_UPDATE = 1/90  # Probability of successful update
P_NETWORK_ERROR = 0.7  # Probability of network error
P_DEVICE_ERROR = 0.7  # Probability of device error

R_RECONNAISSANCE = 0.5  # Reconnaissance rate

def main():
    universe = [(i, j) for i in range(N_ENCLAVES) for j in range(N_ENCLAVES) if i < j]
    GraphSet.set_universe(universe)

    # TODO: Generate all connected graphs
    degree_constraints = {i: range(1, N_ENCLAVES) for i in range(N_ENCLAVES)}
    graphs = GraphSet.graphs(vertex_groups=[range(N_ENCLAVES)], degree_constraints=degree_constraints, num_edges=4)
    print("Number of connected graphs with 5 edges:", len(graphs))
    print("Graphs:", graphs)

    devices = [Device(f"Device {i}", "high_value") for i in range(N_DEVICES)]
    segmentations, neighbours, distances = initialization(graphs, devices, num_enclaves=N_ENCLAVES, K=5)
    archive = map_elites(segmentations, neighbours, distances, generations=5)
    print("Archive size:", len(archive))


if __name__ == "__main__":
    main()
