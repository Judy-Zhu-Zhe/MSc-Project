import random
import copy
from typing import List, Tuple, Dict, Optional

from network import Device, Topology, Segmentation, SegmentationNode
from simulation import Simulation
from metrics import *
from descriptors import *
from config import MapElitesConfig, CompositionalConfig


# ========================================================================================================================
#                                                     MUTATOR
# ========================================================================================================================

def mutate(
        parent: Segmentation,
        topology_list: List[Topology],
        neighbours_table: List[List[int]],
        distances_table: List[List[float]],
        mutation_list: List[str] = ["topology", "partition", "sensitivity"], 
        fine_tuning: bool = False
) -> Segmentation:
    """Mutates a segmentation by modifying its topology, device distribution, and sensitivities."""
    if fine_tuning:   # Conservative fine-tuning
        eta = 5.0
        n_mutations = 1
    else:            # More aggressive
        eta = 10.0 
        n_mutations = 5

    # Mutate topology
    if "topology" in mutation_list:
        parent_index = parent.topology.id
        new_topology = mutate_topology(parent_index, topology_list, neighbours_table, distances_table)
    else:
        new_topology = parent.topology
    # Mutate device partition
    if "partition" in mutation_list:
        new_partition = mutate_device_distribution(parent.partition(), n_mutations=n_mutations)
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
        sensitivities=new_sensitivities
    )
    return new_segmentation

def mutate_topology(
        seg_idx: int, 
        topology_list: List[Topology], 
        neighbours_table: List[List[int]], 
        distances_table: List[List[float]]
    ) -> Topology:
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

def mutate_device_distribution(partition: List[List[Device]], n_mutations: int = 1) -> List[List[Device]]:
    """
    Mutates device distribution by randomly moving one or more devices to different enclaves.

    :param partition: Current device-to-enclave assignment.
    :param level: Compositional level (0 = macro, higher = micro).
    :return: New partition.
    """
    new_partition = [list(devices) for devices in partition]  # Deep copy

    enclave_indices = list(range(1, len(partition)))  # Exclude Internet enclave (index 0)
    if len(enclave_indices) <= 1:
        return new_partition  # No mutation possible

    # Choose source enclave randomly (must not be empty)
    non_empty = [i for i in enclave_indices if new_partition[i]]
    if not non_empty:
        return new_partition
    source_idx = random.choice(non_empty)
    source_devices = new_partition[source_idx]
    # Select devices to move (at most n_mutations)
    moved_devices = random.sample(source_devices, min(n_mutations, len(source_devices)))
    for d in moved_devices:
        new_partition[source_idx].remove(d)

    # Choose a target enclave that is different from the source
    target_candidates = [i for i in enclave_indices if i != source_idx]
    target_idx = random.choice(target_candidates)
    # Move devices to target enclave
    new_partition[target_idx].extend(moved_devices)

    return new_partition

def mutate_enclave_sensitivity(s_parent: float, eta: float = 5.0, s_upper: float = 1.0, s_lower: float = 0.0) -> float:
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

def topology_neighbours(graphs: List[List[Tuple[int, int]]], num_enclaves: int, K: int = 5, verbose: bool = False) -> Tuple[List[Topology], List[List[int]], List[List[float]]]:
    """
    ### Equation 6 ###
    Generate topology list and KNN-based neighbour tables for topology mutation.
    """
    topology_list = [Topology(id=i, n_enclaves=num_enclaves, topology=graph) for i, graph in enumerate(graphs)]
    neighbours_table = []
    distances_table = []

    for i, topology in enumerate(topology_list):
        distances = []
        for j, other_topology in enumerate(topology_list):
            # Calculate distance between topologies using adjacency matrix
            dist = topology_distance(topology.adj_matrix, other_topology.adj_matrix)
            distances.append((j, dist))
        
        distances.sort(key=lambda x: x[1]) # Sort by distance
        top_k = distances[1 : K + 1]  # skip self (distance 0)

        neighbours_table.append([idx for idx, _ in top_k])
        distances_table.append([dist for _, dist in top_k])

    if verbose:
        print(f"Topology initialization done!")
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
        sensitivities=config.sensitivities
    )

def fitness(config: MapElitesConfig, seg: Segmentation, infected_seg: Optional[Segmentation] = None) -> float:
    """
    Calculates the negative weighted loss of a segmentation using evaluation metric functions.
    Each key in `config.evaluation_metrics` is a metric function name, each value is its weight.
    """
    is_adaptation = infected_seg is not None

    total_loss = 0.0
    for metric_name, weight in config.evaluation_metrics.items():
        # Get cached metric if it exists
        if seg.has_cached_metric(metric_name):
            loss = seg.metrics_cache[metric_name]
        # Else, calculate the metric
        else:
            try:
                # Lookup the function by name
                metric_func = globals()[metric_name]
            except KeyError:
                raise ValueError(f"Unknown evaluation metric: '{metric_name}'")

            # Check if it needs both seg and infected_seg
            if metric_name == "topology_distance" and is_adaptation:
                loss = metric_func(seg.topology.adj_matrix, infected_seg.topology.adj_matrix)
            elif metric_name == "security_loss":
                simulations = [Simulation(seg, config, is_adaptation, verbose=config.verbose) for _ in range(config.n_simulations)]
                loss = metric_func(simulations)
            else:
                loss = metric_func(seg)
            # Add the metric to the cache
            seg.add_metric_cache(metric_name, loss)

        if config.verbose:
            print(f"Metric {metric_name} loss: {loss}")
        print(f"Metric {metric_name} loss: {round(loss, 2)}, weight: {round(weight, 2)}, total_loss: {round(weight * loss, 2)}")
        total_loss += weight * loss

    return - total_loss


# MAP-Elites main loop
def map_elites(topology_list: List[Topology], 
               neighbours_table: List[List[int]], 
               distances_table: List[List[float]], 
               config: MapElitesConfig,
               infected_segmentation: Optional[Segmentation] = None,
               verbose: bool = False
               ) -> Tuple[Tuple[Segmentation, float], Dict[Tuple[float, ...], Tuple[Segmentation, float]]]:
    """
    MAP-Elites evolutionary algorithm for exploring segmentation space.

    :return: Tuple(Best result, Last archive grid (segmentation, fitness))
    """
    grid: Dict[Tuple[float, ...], Tuple[Segmentation, float]] = {}
    bin_widths = bins(config)

    # Initialization phase
    if verbose:
        print(f"\n++++++++++++++++++ Starting MAP-Elites with {config.generations} generations ++++++++++++++++++")
    for _ in range(config.init_batch):
        if infected_segmentation:
            seg = mutate(infected_segmentation, topology_list, neighbours_table, distances_table, fine_tuning=False)
        else:
            seg = random_segmentation(random.choice(topology_list), config)
        desc = behavior_descriptors(seg, config.descriptors)
        key = tuple(discretize(desc, bin_widths))
        f = fitness(config, seg, infected_segmentation)
        if key not in grid or f > grid[key][1]:
            grid[key] = (seg, f)

    # Evolutionary loop
    for g in range(config.generations):
        if verbose:
            print(f"\n++++++++++++++++++ Generation [{g + 1}/{config.generations}] ++++++++++++++++++")
        parents = random.sample(list(grid.values()), min(len(grid), config.batch_size))
        fine_tuning = True if g > config.generations * 0.8 else False # Fine-tuning only after 80% of the generations
        
        for parent_seg, _ in parents:
            child = mutate(parent_seg, topology_list, neighbours_table, distances_table, fine_tuning=fine_tuning)
            desc = behavior_descriptors(child, config.descriptors)
            key = tuple(discretize(desc, bin_widths))
            f = fitness(config, child, infected_segmentation)

            if key not in grid or f > grid[key][1]:
                if verbose:
                    print(f"Updated bin {key} with fitness {f:.3f}")
                grid[key] = (child, f)

    best_seg = max(grid.values(), key=lambda x: x[1])

    return best_seg, grid

def compositional_map_elites(
    comp_config: CompositionalConfig,
    infected_node: Optional[SegmentationNode] = None,
    loss_threshold: float = 0.0,
    verbose: bool = False
) -> SegmentationNode:
    """
    Runs a compositional MAP-Elites algorithm across multiple refinement levels.
    If infected_node is provided, performs adaptation starting from the deepest level containing the compromised part.

    :param comp_config: CompositionalConfig object containing hierarchical configurations.
    :param infected_node: Existing (possibly compromised) SegmentationNode to adapt from.
    :param loss_threshold: Loss threshold to determine when adaptation is sufficient.
    :return: Final refined SegmentationNode from the last level.
    """
    # If no adaptation, run as before
    if not infected_node:
        # Create a simple hierarchical structure without trust levels
        best_node = recursive_compositional_map_elites_node(comp_config.configs, comp_config.devices, verbose)
        if best_node:
            return best_node
        else:
            # Fallback: create a simple root node
            root_seg = Segmentation(
                topology=Topology(id=0, n_enclaves=2, topology=[(0, 1)]),
                partition=[[], comp_config.devices],
                sensitivities=[0.0, 0.5]
            )
            return SegmentationNode(root_seg, level=0)

    # ADAPTATION MODE
    # TODO: Check config
    def get_deepest_compromised_path(seg_node: SegmentationNode, path: List[str | int] = []):
        """
        Recursively find the path to the smallest segmentation node that contains all compromised devices.
        Returns (path, node) where node is the smallest such SegmentationNode.
        """
        if not seg_node.children:
            return path, seg_node

        num_compromised_here = len(seg_node.seg.all_compromised_devices())
        num_compromised_subtree = {idx: len(child.seg.all_compromised_devices()) for idx, child in seg_node.children.items()}

        # Check if any child contains all compromised devices in the subtree
        for idx, child in num_compromised_subtree.items():
            if child == num_compromised_here:
                return get_deepest_compromised_path(seg_node.children[idx], path + [idx])

        return path, seg_node
    
    path, current_node = get_deepest_compromised_path(infected_node)
    # For testing
    assert path, "No path found"
    
    # Get configs for the path (simplified for non-trust-level structure)
    configs = comp_config.configs[:len(path)]
    while current_node and current_node.parent is not None:
        # Run MAP-Elites adaptation at this level
        config = configs[-1]
        devices = current_node.seg.all_devices()
        if verbose:
            print(f"\nAdaptation: Running MAP-Elites at level {config.level} ({path})")
        adp_node = recursive_compositional_map_elites_node(configs, devices, verbose)
        if adp_node is not None:
            # Calculate the loss of the entire segmentation after adaptation
            new_node = copy.deepcopy(infected_node)
            new_node.update_by_path(path, adp_node)
            loss = fitness(config, new_node.seg, infected_node.seg)
            if verbose:
                print(f"Adaptation: Loss at path {path} is {loss}")
            if loss <= loss_threshold:
                if verbose:
                    print(f"Adaptation: Acceptable loss {loss} <= {loss_threshold} at path {path}")
                return new_node
        else:
            if verbose:
                print(f"Adaptation failed at path {path}: adp_node is None")
            return infected_node

        # Update current node and path
        current_node = new_node
        path = path[:-1]
            
    if verbose:
        print(f"No adaptation found within loss threshold {loss_threshold} (current loss: {loss}). Need urgent fix for compromise!!")
    return current_node


def recursive_compositional_map_elites_node(
    configs: List[MapElitesConfig],
    devices: List[Device],
    verbose: bool = False
) -> Optional[SegmentationNode]:
    """
    Recursively runs MAP-Elites for each level, partitioning devices at each step.
    Returns a SegmentationNode tree.
    """
    if not configs:
        return None

    config = configs[0]
    config.devices = devices

    topology_list, neighbours_table, distances_table = topology_neighbours(config.universe, config.n_enclaves, verbose=config.verbose)
    (best_seg, loss), grid = map_elites(
        topology_list,
        neighbours_table,
        distances_table,
        config,
        verbose=config.verbose
    )
    node = SegmentationNode(best_seg, level=config.level, config=config)

    if len(configs) > 1:
        next_configs = configs[1:]
        for i, enclave in enumerate(best_seg.enclaves):
            if i == 0:
                continue  # skip Internet
            child_node = recursive_compositional_map_elites_node(next_configs, enclave.devices, verbose)
            if child_node:
                node.add_child(i, child_node)

    return node

