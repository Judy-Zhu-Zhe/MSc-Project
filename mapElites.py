import statistics
import random
from typing import List, Tuple, Dict
from graphillion import GraphSet

from network import Device, Enclave, Topology, Segmentation
from simulation import Simulation
from metrics import security_loss, performance_loss, resilience_loss, topology_distance
from config import MapElitesConfig

# ========================================================================================================================
#                                                 BEHAVIOR DESCRIPTORS
# ========================================================================================================================

def num_high_degree_nodes(segmentation: Segmentation) -> int:
    """Returns the number of enclaves with more than 2 neighbours."""
    return sum(1 for n in segmentation.topology.adj_matrix if sum(n) > 3)

def std_devices(segmentation: Segmentation) -> float:
    """Returns the standard deviation of the number of devices in enclaves."""
    if len(segmentation.enclaves) > 1:
        return statistics.stdev([len(e.devices) for e in segmentation.enclaves])
    return 0.0

def std_high_value(segmentation: Segmentation) -> float:
    """Returns the standard deviation of the number of high-value devices in enclaves."""
    if len(segmentation.enclaves) > 1:
        return statistics.stdev([sum(1 for d in e.devices if "high_value" in d.device_group) for e in segmentation.enclaves])
    return 0.0

def std_web_facing(segmentation: Segmentation) -> float:
    """Returns the standard deviation of the number of web-facing devices in enclaves."""
    if len(segmentation.enclaves) > 1:
        return statistics.stdev([sum(1 for d in e.devices if "performance_affecting" in d.device_group) for e in segmentation.enclaves])
    return 0.0

def std_resilience(segmentation: Segmentation) -> float:
    """Returns the standard deviation of the number of resilience-affected devices in enclaves."""
    if len(segmentation.enclaves) > 1:
        return statistics.stdev([sum(1 for d in e.devices if "resilience_affected" in d.device_group) for e in segmentation.enclaves])
    return 0.0

def distance_high_val(segmentation: Segmentation) -> int:
    """Returns the sum of distances to the internet for high-value devices."""
    return sum(e.dist_to_internet for e in segmentation.enclaves for d in e.devices if "high_value" in d.device_group and e.dist_to_internet)

DESCRIPTOR_FUNCTIONS = {
    "nb_high_deg_nodes": num_high_degree_nodes,
    "std_devices": std_devices,
    "std_high_value": std_high_value,
    "std_web_facing": std_web_facing,
    "std_resilience": std_resilience,
    "distance_high_val": distance_high_val,
}

def behavior_descriptors(seg: Segmentation, descriptors: List[str] = list(DESCRIPTOR_FUNCTIONS.keys())) -> Tuple:
    """Computes behavior descriptors dynamically based on selected keys."""
    try:
        values = [DESCRIPTOR_FUNCTIONS[name](seg) for name in descriptors]
    except KeyError as e:
        raise ValueError(f"Invalid descriptor name: {e}. Available descriptors are: {list(DESCRIPTOR_FUNCTIONS.keys())}")
    
    return tuple(values)

def bins(config: MapElitesConfig) -> List[float]:
    """Discretizes the behavior descriptor into bins."""
    # TODO: Better binning strategy
    n_enclaves = config.n_enclaves
    num_std = sum(1 for d in config.descriptors if "std" in d)
    std_bin = max(2, int(round((config.batch_size * 3) ** (1 / num_std))))  # at least 2 per dim
    bin_widths = []
    for descriptor in config.descriptors:
        if "std" in descriptor:
            bin_widths.append(round((n_enclaves + 1) / std_bin, 2)) # Standard deviation descriptors
        elif "nb" in descriptor or "distance" in descriptor:
            bin_widths.append(1.0) # integer descriptors
        else:
            bin_widths.append(1.0)
    return bin_widths

def discretize(values: List[float], bin_widths: List[float]) -> List[float]:
    """Discretizes a list of values into bins based on bin widths."""
    assert len(values) == len(bin_widths), "Values and bin widths must have the same length"
    return tuple(round(d // b * b, 2) for d, b in zip(values, bin_widths))


# ========================================================================================================================
#                                                     MUTATOR
# ========================================================================================================================

def mutate(
        parent: Segmentation,
        topology_list: List[Topology],
        neighbours_table: List[List[int]],
        distances_table: List[List[float]],
        mutation_list: List[int] = ["topology", "partition", "sensitivity"],
        eta: float = 20.0,
        n_low_value_device: int = 0
) -> Segmentation:
    """Mutates a segmentation by modifying its topology, device distribution, and sensitivities."""
    # Mutate topology
    if "topology" in mutation_list:
        parent_index = parent.topology.id
        new_topology = mutate_topology(parent_index, topology_list, neighbours_table, distances_table)
    else:
        new_topology = parent.topology
    # Mutate device partition
    if "partition" in mutation_list:
        new_partition = mutate_device_distribution(parent.partition())
    else:
        new_partition = parent.partition()
    # Mutate sensitivities
    if "sensitivity" in mutation_list:
        new_sensitivities = [mutate_enclave_sensitivity(s, eta=eta) for s in parent.sensitivities()]
    else:
        new_sensitivities = parent.sensitivities()

    new_segmentation = Segmentation(
        topology=new_topology,
        partition=new_partition,
        sensitivities=new_sensitivities,
        vulnerabilities=parent.vulnerabilities(),
        n_low_value_device=n_low_value_device
    )
    return new_segmentation

def mutate_topology(seg_idx: int, topology_list: List[Topology], neighbours_table: List[List[int]], distances_table: List[List[float]]) -> Topology:
    """
    ### Equation 7 ###
    Mutate the topology of the segmentation using KNN-based neighbor sampling.
    """
    r = random.random()
    if r < 0.5:
        return topology_list[seg_idx]  # No mutation, return the same topology
    
    # Else, mutate the topology by selecting a neighbour based on KNN
    neighbours = neighbours_table[seg_idx]
    distances = distances_table[seg_idx]

    # Convert distances to inverse weights (closer = higher weight)
    weights = [1 / d if d > 0 else 1e9 for d in distances]
    probabilities = [w / sum(weights) for w in weights]

    # Choose a neighbour based on weighted probability and return the mutation
    selected_index = random.choices(neighbours, weights=probabilities)[0]
    return topology_list[selected_index]

def mutate_device_distribution(partition: List[List[Device]]) -> List[List[Device]]:
    """Mutates device distribution by randomly moving one device to a different enclave."""
    # Deep copy of the partition to avoid modifying the original
    new_partition = [list(devices) for devices in partition]
    assert new_partition, "Partition should not be empty"

    enclave_indices = list(range(1, len(partition))) # Non-Internet enclaves
    if len(enclave_indices) <= 1:
        return new_partition  # No mutation possible
    
    # Remove a random device from a randomly-selected non-empty enclave
    non_empty_indices = [i for i, devices in enumerate(new_partition) if devices]
    from_idx = random.choice(non_empty_indices)
    device = random.choice(new_partition[from_idx])
    new_partition[from_idx].remove(device)

    # Select a new enclave different from the current one
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
    s_child = s_parent + (s_upper - s_lower) * delta 
    return max(s_lower, min(s_child, s_upper))


# ========================================================================================================================
#                                                   OPERATORS
# ========================================================================================================================

def topology_neighbours(graphs: GraphSet, num_enclaves: int, K: int = 5) -> Tuple[List[Topology], List[List[int]], List[List[float]]]:
    """
    Calculate KNN neighbor and distance tables for a list of graphs.

    :param graphs: List of Graphillion graphs, where each graph is a list of (int, int) edges.
    :param num_enclaves: Number of enclaves (nodes) in each topology.
    :param devices: List of Device objects to assign to enclaves.
    :param K: Number of nearest neighbors to compute for each topology.

    :return: A tuple containing:
        - topology_list: List of Topology objects initialized from the input graphs.
        - neighbours_table: Top K nearest neighbors for each topology, 
        - distance table: Their corresponding distances)
    """
    assert len(graphs) > 0, "No graphs provided for initialization"
    topology_list: List[Topology] = [Topology(id=i, n_enclaves=num_enclaves, topology=g) for i, g in enumerate(graphs)]
    adj_matrices = [topology.adj_matrix for topology in topology_list]
    distances_table = []
    neighbours_table = []

    for i, mat_i in enumerate(adj_matrices):
        distances = []
        for j, mat_j in enumerate(adj_matrices):
            dist = topology_distance(mat_i, mat_j)
            distances.append((j, dist))

        distances.sort(key=lambda x: x[1]) # Sort by distance
        top_k = distances[1 : K + 1]  # skip self (distance 0)

        neighbours_table.append([idx for idx, _ in top_k])
        distances_table.append([dist for _, dist in top_k])

    print(f"Initialization done!")
    return topology_list, neighbours_table, distances_table

def random_segmentation(topology: Topology, config: MapElitesConfig) -> Segmentation:
    """Generates a random segmentation given a topology."""
    partition = [[] for _ in range(topology.n_enclaves)]
    for device in config.devices:
        enclave_index = random.randint(1, topology.n_enclaves - 1) # Randomly assign to a non-Internet enclave
        partition[enclave_index].append(device)
    return Segmentation(
        topology=topology,
        partition=partition,
        sensitivities=config.sensitivities,
        vulnerabilities=config.vulnerabilities,
        n_low_value_device=config.n_low_value_device
    )

def fitness(config: MapElitesConfig, seg: Segmentation, infected_seg: Segmentation = None) -> float:
    """Calculates the negative total loss of a segmentation based on its behavior descriptors."""
    is_adaptation = True if infected_seg else False
    simulations = [Simulation(seg, config, is_adaptation) for _ in range(config.n_simulations)]
    loss = security_loss(simulations) * config.metric_weights[0] + \
           performance_loss(seg) * config.metric_weights[1] + \
           resilience_loss(seg) * config.metric_weights[2]
    if is_adaptation and len(config.metric_weights) > 3:
        # Add topology distance for adapted segmentations
        loss += topology_distance(seg.topology.adj_matrix, infected_seg.topology.adj_matrix) * config.metric_weights[3]
    return - loss


# MAP-Elites main loop
def map_elites(topology_list: List[Topology], 
               neighbours_table: List[List[int]], 
               distances_table: List[List[float]], 
               config: MapElitesConfig,
               infected_segmentation: Segmentation = None
               ) -> Tuple[Tuple[Segmentation, float], Dict[Tuple, Tuple[Segmentation, float]]]:
    """
    MAP-Elites evolutionary algorithm for exploring segmentation space.

    :return: Tuple(Best result, Last archive grid (segmentation, fitness))
    """
    grid: Dict[Tuple, Tuple[Segmentation, float]] = {}
    bin_widths = bins(config)

    # Initialization phase
    print(f"\n++++++++++++++++++ Initializing with {config.init_batch} segmentations ++++++++++++++++++")
    for _ in range(config.init_batch):
        seg = random_segmentation(random.choice(topology_list), config)
        desc = behavior_descriptors(seg, config.descriptors)
        key = discretize(desc, bin_widths)
        f = fitness(config, seg, infected_segmentation)
        if key not in grid or f > grid[key][1]:
            grid[key] = (seg, f)

    # Evolutionary loop
    for g in range(config.generations):
        print(f"\n++++++++++++++++++ Generation [{g + 1}/{config.generations}] ++++++++++++++++++")
        parents = random.sample(list(grid.values()), min(len(grid), config.batch_size))
        
        for parent_seg, _ in parents:
            # mutation_type = random.choice(["topology", "partition", "sensitivity"])
            child = mutate(parent_seg, topology_list, neighbours_table, distances_table, eta=config.eta, n_low_value_device=config.n_low_value_device)
            desc = behavior_descriptors(child, config.descriptors)
            key = discretize(desc, bin_widths)
            f = fitness(config, child, infected_segmentation)

            if key not in grid or f > grid[key][1]:
                print(f"Updated bin {key} with fitness {f:.3f}")
                grid[key] = (child, f)

    best_seg = max(grid.values(), key=lambda x: x[1])

    return best_seg, grid
