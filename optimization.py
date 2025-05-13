import statistics
import random
import math
from typing import List, Tuple, Dict
from graphillion import GraphSet

from network import Device, Enclave, Segmentation
from simulation import Simulation
from metrics import security_loss, performance_loss, resilience_loss

# ========================================================================================================================
#                                                 BEHAVIOR DESCRIPTORS
# ========================================================================================================================

def num_high_degree_nodes(segmentation: Segmentation) -> int:
    """Returns the number of enclaves with more than 2 neighbours."""
    return sum(1 for e in segmentation.enclaves if len(e.neighbours) > 2)

def std_enclave_size(segmentation: Segmentation) -> float:
    """Returns the standard deviation of the number of devices in enclaves."""
    if len(segmentation.enclaves) > 1:
        return statistics.stdev([len(e.devices) for e in segmentation.enclaves])
    return 0.0

def std_device_group(segmentation: Segmentation, group: str) -> float:
    """Returns the standard deviation of the number of devices in a specific group across enclaves."""
    if len(segmentation.enclaves) > 1:
        return statistics.stdev([sum(1 for d in e.devices if group in d.device_group) for e in segmentation.enclaves])
    return 0.0

def distance_high_val(segmentation: Segmentation) -> float:
    """Returns the sum of distances to the internet for high-value devices."""
    return sum(e.distance_to_internet for e in segmentation.enclaves for d in e.devices if "high_value" in d.device_group and e.distance_to_internet is not None)

def behavior_descriptors(seg: Segmentation) -> Tuple[int, float, float, float, float, int]:
    """Calculates behavior descriptors for a given segmentation."""
    nb_high_deg = num_high_degree_nodes(seg)
    std_dev = std_enclave_size(seg)
    std_high_val = std_device_group(seg, "high_value")
    std_web_facing = std_device_group(seg, "performance_affecting")
    std_res = std_device_group(seg, "resilience_affected")
    dist_high_val = distance_high_val(seg)
    return (nb_high_deg, std_dev, std_high_val, std_web_facing, std_res, dist_high_val)

def discretize(descriptor: Tuple, bins: Tuple[int, ...]) -> Tuple:
    """Discretizes the behavior descriptor into bins."""
    return tuple(int(d // b) for d, b in zip(descriptor, bins))


# ========================================================================================================================
#                                                     MUTATOR
# ========================================================================================================================

def mutate(
    parent_index: int,
    segmentations: List[Segmentation],
    neighbours_table: List[List[int]],
    distances_table: List[List[float]],
    eta: float = 20.0,
    mutate_prob: float = 0.3
) -> Segmentation:
    """Mutates a segmentation by modifying its topology, device distribution, and sensitivities."""
    parent = segmentations[parent_index]

    # Mutate topology (deep copy for isolation)
    new_topology = mutate_topology(parent_index, neighbours_table, distances_table, segmentations)
    # Mutate device partition
    new_partition = mutate_device_distribution(parent.partition())
    # Mutate sensitivities
    new_sensitivities = mutate_enclave_sensitivity(parent.sensitivities(), mutate_prob=mutate_prob, eta=eta)

    return Segmentation(
        topology=new_topology,
        partition=new_partition,
        sensitivities=new_sensitivities
    )

def mutate_topology(seg_index: int, segmentations: List[Segmentation], neighbours_table: List[List[int]], distances_table: List[List[float]]) -> Segmentation:
    """
    ### Equation 7 ###
    Mutate the topology of the segmentation using KNN-based neighbor sampling.
    
    :param seg_index: Index of the current segmentation
    :param segmentations: The list of all segmentations
    :param neighbours_table: List of neighbor indices for each segmentation
    :param distances_table: Corresponding distances for each neighbor
    :return: Mutated topology as a list of neighbor indices per enclave
    """
    neighbours = neighbours_table[seg_index]
    distances = distances_table[seg_index]

    # Convert distances to inverse weights (closer = higher weight)
    weights = [1 / d if d > 0 else 1e9 for d in distances]
    probabilities = [w / sum(weights) for w in weights]

    # Choose a neighbor based on weighted probability and return the mutation
    selected_index = random.choices(neighbours, weights=probabilities, k=1)[0]
    return segmentations[selected_index]

def mutate_device_distribution(partition: List[List[Device]]) -> List[List[Device]]:
    """
    Mutates device distribution by randomly moving one device to a different enclave.

    :param partition: Current list of devices per enclave
    :return: New mutated partition
    """
    # Deep copy the partition
    new_partition = [list(devices) for devices in partition]

    # Select a non-empty enclave
    non_empty_indices = [i for i, p in enumerate(new_partition) if p]
    if len(non_empty_indices) < 1:
        return new_partition  # No mutation possible

    from_idx = random.choice(non_empty_indices)
    device = random.choice(new_partition[from_idx])
    new_partition[from_idx].remove(device)

    # Select a new enclave different from the current one
    enclave_indices = list(range(len(new_partition)))
    enclave_indices.remove(from_idx)
    to_idx = random.choice(enclave_indices)
    new_partition[to_idx].append(device)

    return new_partition

def mutate_enclave_sensitivity(s_parent: float, eta: float = 20.0, s_upper: float = 1.0, s_lower: float = 0.0) -> float:
    """
    ### Equation 8, 9 ###
    Generates a child sensitivity value based on the parent sensitivity.
    
    :param s_parent: Parent sensitivity value.
    :param eta: Sensitivity hyperparameter.
    :param s_upper: Sensitivity upper bound.
    :param s_lower: Sensitivity lower bound.
    
    :return: Child sensitivity value.
    """
    r = random.random()
    if r < 0.5:
        delta = (2 * r) ** (1 / (eta + 1)) - 1
    else:
        delta = 1 - (2 * (1 - r)) ** (1 / (eta + 1))
    return s_parent + (s_upper - s_lower) * delta 

def topology_distance(matrix1: List[List[int]], matrix2: List[List[int]]) -> float:
    """
    Computes the Euclidean distance between two adjacency matrices.

    :param matrix1: First adjacency matrix (list of lists of 0s and 1s)
    :param matrix2: Second adjacency matrix
    :return: Euclidean distance
    """
    assert len(matrix1) == len(matrix2), "Matrices must be the same size"
    assert all(len(row1) == len(row2) for row1, row2 in zip(matrix1, matrix2)), "Matrices must be square and aligned"

    flat1 = [val for row in matrix1 for val in row]
    flat2 = [val for row in matrix2 for val in row]
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(flat1, flat2)))


# ========================================================================================================================
#                                                   OPERATORS
# ========================================================================================================================

SECURITY_WEIGHT = 1
PERFORMANCE_WEIGHT = 0
RESILIENCE_WEIGHT = 0

def initialization(graphs, universe, devices: List[Device], num_enclaves: int, K: int = 5) -> Tuple[List[Segmentation], List[List[int]], List[List[float]]]:
    """
    Initialize Segmentation objects using Graphillion-generated valid topologies.

    :param graphs: List of Graphillion graphs (each a list of (int, int) edges)
    :param universe: The universe of edges used to define GraphSet
    :param devices: List of Device objects
    :param num_enclaves: Number of enclaves in each segmentation
    :param K: K = Number of neighbors for KNN
    :return: List of Segmentation objects
    """
    segmentations: List[Segmentation] = []
    assert len(graphs) > 0, "No graphs provided for initialization"

    for g in graphs:
        # Build neighbor relations from the Graphillion graph (edge list)
        topology = [[] for _ in range(num_enclaves)]
        for u, v in g:
            topology[u].append(v)
            topology[v].append(u)

        # Randomly assign devices to enclaves
        partition = [[] for _ in range(num_enclaves)]
        for device in devices:
            i = random.randint(0, num_enclaves - 1)
            partition[i].append(device)

        # TODO: Generate sensitivities and vulnerabilities
        sensitivities = [random.uniform(0, 1) for _ in range(num_enclaves)]
        vulnerabilities = [random.uniform(0, 1) for _ in range(num_enclaves)]

        # Create segmentation
        segmentation = Segmentation(topology, partition, sensitivities, vulnerabilities)
        segmentations.append(segmentation)

        # Compute distance and neighbor tables
        distances_table = []
        neighbours_table = []
        adjacency_matrices = [seg.topology_matrix for seg in segmentations]
        for i, mat_i in enumerate(adjacency_matrices):
            distances = []
            for j, mat_j in enumerate(adjacency_matrices):
                dist = topology_distance(mat_i, mat_j)
                distances.append((j, dist))
            distances.sort(key=lambda x: x[1])
            top_n = distances[1 : K + 1]  # skip self (distance 0)
            neighbours_table.append([idx for idx, _ in top_n])
            distances_table.append([d for _, d in top_n])

    return segmentations, neighbours_table, distances_table

def fitness(seg: Segmentation, generations: int = 50) -> float:
    """Calculates the fitness of a segmentation based on its behavior descriptors."""
    simulations = [Simulation(seg, ) for _ in range(generations)]
    loss = security_loss(simulations) * SECURITY_WEIGHT + \
           performance_loss(seg) * PERFORMANCE_WEIGHT + \
           resilience_loss(seg) * RESILIENCE_WEIGHT
    return - loss

# MAP-Elites main loop
def map_elites(
    initial_segmentations: List[Segmentation],
    neighbours_table: List[List[int]],
    distances_table: List[List[float]],
    generations: int = 100,
    bins: Tuple[int, ...] = (1, 1, 1, 1, 1, 10),
) -> Dict[Tuple, Tuple[Segmentation, float]]:
    """
    MAP-Elites evolutionary algorithm for exploring segmentation space.

    :param initial_segmentations: List of initial Segmentations
    :param neighbours_table: KNN neighbor indices for each topology
    :param distances_table: Corresponding KNN distances
    :param generations: Number of generations to evolve
    :param bins: Discretization bin sizes for behavior descriptors
    :return: Archive grid mapping behavior bins to (segmentation, fitness)
    """
    grid: Dict[Tuple, Tuple[Segmentation, float]] = {}

    # === INITIAL POPULATION EVALUATION ===
    for i, seg in enumerate(initial_segmentations):
        desc = behavior_descriptors(seg)
        key = discretize(desc, bins)
        f = fitness(seg)
        if key not in grid or f > grid[key][1]:
            grid[key] = (seg, f)

    # === EVOLUTIONARY LOOP ===
    for _ in range(generations):
        for _ in range(len(initial_segmentations)):
            parent_index = random.randint(0, len(initial_segmentations) - 1)
            child = mutate(parent_index, initial_segmentations, neighbours_table, distances_table)

            desc = behavior_descriptors(child)
            key = discretize(desc, bins)
            f = fitness(child)

            if key not in grid or f > grid[key][1]:
                grid[key] = (child, f)

    return grid

N_encalves = 5

universe = [(i, j) for i in range(N_encalves) for j in range(N_encalves) if i < j]
GraphSet.set_universe(universe)

# Generate all connected graphs with exactly 3 edges
graphs = GraphSet.graphs(num_edges=3)
print("Number of connected graphs with 3 edges:", len(graphs))
print(graphs)

devices = [Device(f"Device {i}", "high_value") for i in range(55)]
segmentations, neighbours, distances = initialization(graphs, universe, devices, num_enclaves=5, K=5)
archive = map_elites(segmentations, neighbours, distances, generations=10)
print("Archive size:", len(archive))