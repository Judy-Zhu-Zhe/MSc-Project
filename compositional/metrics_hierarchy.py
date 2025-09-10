from typing import List, Dict, Optional
import math
from collections import deque
from network import SegmentationNode
from metrics import topology_distance as base_topology_distance, security_loss as base_security_loss, performance_loss as base_performance_loss, resilience_loss as base_resilience_loss
from simulation import Simulation


def topology_distance(node1: SegmentationNode, node2: SegmentationNode, 
                     level_weights: Optional[Dict[int, float]] = None) -> float:
    """
    Computes the hierarchical topology distance between two SegmentationNode objects.
    This function sums all the topology distances in their child segmentations with weights,
    where larger weights are assigned to higher levels so that structural changes at 
    coarse granularity contribute more to the total score.
    
    The child topology distance is only re-calculated if its cache does not contain the value.
    
    :param node1: First SegmentationNode
    :param node2: Second SegmentationNode  
    :param level_weights: Optional dictionary mapping level to weight. If None, uses default weights.
    :return: Weighted sum of topology distances across all levels
    """
    # Default level weights: higher levels get larger weights
    if level_weights is None:
        level_weights = _get_default_level_weights(node1, node2)
    
    total_distance = 0.0
    
    # Calculate topology distance at current level
    current_level = node1.level
    weight = level_weights.get(current_level, 1.0)
    
    # Check cache first for current level topology distance
    cache_key = f"topology_distance_level_{current_level}"
    
    if (node1.seg.has_cached_metric(cache_key) and 
        node2.seg.has_cached_metric(cache_key)):
        # Use cached values
        distance = abs(node1.seg.metrics_cache[cache_key] - node2.seg.metrics_cache[cache_key])
    else:
        # Calculate topology distance between current level segmentations
        distance = base_topology_distance(
            node1.seg.topology.adj_matrix, 
            node2.seg.topology.adj_matrix
        )
        # Cache the calculated distance
        node1.seg.add_metric_cache(cache_key, distance)
        node2.seg.add_metric_cache(cache_key, distance)
    
    total_distance += weight * distance
    
    # Recursively calculate distances for children
    total_distance += _calculate_children_distances(node1, node2, level_weights)
    
    return total_distance


def _get_default_level_weights(node1: SegmentationNode, node2: SegmentationNode) -> Dict[int, float]:
    """
    Generate default level weights where higher levels (coarser granularity) get larger weights.
    
    :param node1: First SegmentationNode
    :param node2: Second SegmentationNode
    :return: Dictionary mapping level to weight
    """
    # Find the maximum level in the hierarchy
    max_level = max(_get_max_level(node1), _get_max_level(node2))
    
    # Generate weights: level 0 (root) gets highest weight, each level down gets half the weight
    # Lower level numbers = higher hierarchy = higher weights
    # Example: if max_level=3, then level 0 gets 8.0, level 1 gets 4.0, level 2 gets 2.0, level 3 gets 1.0
    weights = {}
    for level in range(max_level + 1):
        weights[level] = 2.0 ** (max_level - level)
    
    return weights


def _get_max_level(node: SegmentationNode) -> int:
    """
    Recursively find the maximum level in the hierarchy starting from the given node.
    
    :param node: Starting SegmentationNode
    :return: Maximum level found
    """
    max_level = node.level
    
    for child in node.children.values():
        child_max = _get_max_level(child)
        max_level = max(max_level, child_max)
    
    return max_level


def _calculate_children_distances(node1: SegmentationNode, node2: SegmentationNode, 
                                level_weights: Dict[int, float]) -> float:
    """
    Recursively calculate topology distances for all child nodes.
    
    :param node1: First SegmentationNode
    :param node2: Second SegmentationNode
    :param level_weights: Dictionary mapping level to weight
    :return: Sum of weighted distances for all children
    """
    total_distance = 0.0
    
    # Get all unique child indices from both nodes
    all_child_indices = set(node1.children.keys()) | set(node2.children.keys())
    
    for child_idx in all_child_indices:
        child1 = node1.children.get(child_idx)
        child2 = node2.children.get(child_idx)
        
        if child1 is not None and child2 is not None:
            # Both nodes have this child, calculate distance recursively
            child_distance = topology_distance(child1, child2, level_weights)
            total_distance += child_distance
        elif child1 is not None or child2 is not None:
            # Only one node has this child, this represents a structural difference
            # Calculate the "distance" as the topology distance of the existing child
            # compared to an empty topology (all zeros)
            existing_child = child1 if child1 is not None else child2
            child_level = existing_child.level
            weight = level_weights.get(child_level, 1.0)
            
            # Create empty adjacency matrix for comparison
            n_enclaves = existing_child.seg.topology.n_enclaves
            empty_matrix = [[0 for _ in range(n_enclaves)] for _ in range(n_enclaves)]
            
            # Check cache for this comparison
            cache_key = f"topology_distance_empty_level_{child_level}"
            
            if existing_child.seg.has_cached_metric(cache_key):
                distance = existing_child.seg.metrics_cache[cache_key]
            else:
                distance = base_topology_distance(
                    existing_child.seg.topology.adj_matrix, 
                    empty_matrix
                )
                existing_child.seg.add_metric_cache(cache_key, distance)
            
            total_distance += weight * distance
    
    return total_distance


def clear_topology_distance_cache(node: SegmentationNode):
    """
    Clear all topology distance related cache entries from a SegmentationNode and its children.
    
    :param node: SegmentationNode to clear cache for
    """
    # Clear current level cache
    current_level = node.level
    cache_key = f"topology_distance_level_{current_level}"
    node.seg.invalidate_metric(cache_key)
    
    # Clear empty comparison cache
    empty_cache_key = f"topology_distance_empty_level_{current_level}"
    node.seg.invalidate_metric(empty_cache_key)
    
    # Recursively clear children cache
    for child in node.children.values():
        clear_topology_distance_cache(child)


def hierarchical_security_loss(node: SegmentationNode, simulations: List[Simulation]) -> float:
    """
    Hierarchical version of security loss that computes recursively.
    
    For a node u with children C(u):
    L_s(u) = sum(L_s(c) for c in C(u)) + L_s^cross(u)
    
    where L_s(c) is the aggregated loss from child c, and L_s^cross(u) captures 
    only inter-enclave propagation at level u, preventing double counting.
    
    :param node: The SegmentationNode to evaluate
    :param simulations: List of simulations for the current level
    :return: Hierarchical security loss
    """
    # Base case: if no children, use original device-level simulation
    if not node.children:
        return base_security_loss(simulations)
    
    # Recursive case: sum losses from children
    total_loss = 0.0
    
    # Aggregate losses from all children
    for child_idx, child_node in node.children.items():
        # Get simulations for this child (filtered by enclave)
        child_simulations = _filter_simulations_by_enclave(simulations, child_idx)
        if child_simulations:
            child_loss = hierarchical_security_loss(child_node, child_simulations)
            total_loss += child_loss
    
    # Add cross-level propagation loss (inter-enclave at current level)
    cross_loss = _calculate_cross_level_security_loss(node, simulations)
    total_loss += cross_loss
    
    return total_loss


def hierarchical_performance_loss(node: SegmentationNode) -> float:
    """
    Hierarchical version of performance loss.
    
    In the hierarchical case, L_p is evaluated primarily at macro levels, 
    as micro-segmentation rarely affects Internet hop distance.
    
    For each enclave E:
    L_p = sum(|P(D) ∩ devices(E)| * d(Internet, E))
    
    where |P(D) ∩ devices(E)| is the number of performance-sensitive devices 
    in enclave E, and d(Internet, E) is its shortest-path distance to the Internet.
    
    :param node: The SegmentationNode to evaluate
    :return: Hierarchical performance loss
    """
    # Base case: if no children, use original performance loss calculation
    if not node.children:
        return base_performance_loss(node.seg)
    
    # For hierarchical case, evaluate at current level
    loss = 0.0
    
    for enclave in node.seg.enclaves:
        if enclave.id == 0:  # Skip Internet enclave
            continue
            
        # Count performance-sensitive devices in this enclave
        num_performance_sensitive = 0
        for device in enclave.devices:
            if device.internet_sensitive:
                num_performance_sensitive += 1
        
        # Calculate loss: number of performance-sensitive devices * distance to Internet
        if enclave.dist_to_internet is not None:
            loss += num_performance_sensitive * enclave.dist_to_internet
        elif num_performance_sensitive > 0:
            # If performance-sensitive devices are not connected to Internet, return infinity
            return float("inf")
    
    return loss


def hierarchical_resilience_loss(node: SegmentationNode) -> float:
    """
    Hierarchical version of resilience loss.
    
    In the hierarchical version, resilience is evaluated independently at each level 
    of the hierarchy, as resilience depends on the enclave graph at that level. 
    Each enclave removal considers both direct effects (its own devices) and 
    indirect effects (subtrees disconnected from the Internet).
    
    :param node: The SegmentationNode to evaluate
    :return: Hierarchical resilience loss
    """
    # Base case: if no children, use original resilience loss calculation
    if not node.children:
        return base_resilience_loss(node.seg)
    
    # For hierarchical case, evaluate at current level
    loss = 0.0
    n_enclaves = node.seg.topology.n_enclaves
    
    for enclave_id in range(1, n_enclaves):  # Skip Internet (id=0)
        # Calculate reachable enclaves when this enclave is blocked
        reachable = _bfs_connected_hierarchical(
            start=0, 
            blocked=enclave_id, 
            adj_matrix=node.seg.topology.adj_matrix
        )
        
        # Direct loss: compromise value of devices in the failed enclave
        enclave_loss = sum(d.compromise_value for d in node.seg.enclaves[enclave_id].devices)
        
        # Indirect loss: enclaves that become disconnected from Internet
        for i in range(1, n_enclaves):
            if i != enclave_id and i not in reachable:
                disconnected_enclave = node.seg.enclaves[i]
                # Add compromise value of all devices in disconnected enclave
                enclave_loss += sum(d.compromise_value for d in disconnected_enclave.devices)
        
        loss += enclave_loss
    
    return loss / (n_enclaves - 1) if n_enclaves > 1 else 0


def _filter_simulations_by_enclave(simulations: List[Simulation], enclave_id: int) -> List[Simulation]:
    """
    Filter simulations to only include those relevant to a specific enclave.
    This is a simplified implementation - in practice, you might need more sophisticated filtering.
    
    :param simulations: List of all simulations
    :param enclave_id: ID of the enclave to filter for
    :return: Filtered list of simulations
    """
    # For now, return all simulations as they should contain information about all enclaves
    # In a more sophisticated implementation, you might filter based on which devices
    # were involved in each simulation
    return simulations


def _calculate_cross_level_security_loss(node: SegmentationNode, simulations: List[Simulation]) -> float:
    """
    Calculate the cross-level security loss component.
    This captures inter-enclave propagation at the current level, preventing double counting.
    
    :param node: The SegmentationNode to evaluate
    :param simulations: List of simulations for the current level
    :return: Cross-level security loss
    """
    # This is a simplified implementation
    # In practice, you would need to:
    # 1. Identify devices that were compromised through inter-enclave propagation
    # 2. Calculate the loss from these devices
    # 3. Ensure no double counting with child-level losses
    
    cross_loss = 0.0
    
    # For now, we'll use a simplified approach based on enclave connectivity
    # and vulnerability distributions
    for i, enclave_i in enumerate(node.seg.enclaves):
        if i == 0:  # Skip Internet
            continue
            
        for j, enclave_j in enumerate(node.seg.enclaves):
            if j <= i:  # Skip Internet and avoid double counting
                continue
                
            # If enclaves are connected, there's potential for cross-enclave propagation
            if node.seg.topology.adj_matrix[i][j]:
                # Calculate potential loss from cross-enclave propagation
                # This is a simplified heuristic based on enclave vulnerabilities
                avg_vuln_i = sum(d.vulnerability for d in enclave_i.devices) / len(enclave_i.devices) if enclave_i.devices else 0
                avg_vuln_j = sum(d.vulnerability for d in enclave_j.devices) / len(enclave_j.devices) if enclave_j.devices else 0
                
                # Cross-propagation risk is proportional to average vulnerabilities
                cross_risk = (avg_vuln_i + avg_vuln_j) / 2.0
                cross_loss += cross_risk * 0.1  # Scale factor
    
    return cross_loss


def _bfs_connected_hierarchical(start: int, blocked: int, adj_matrix: List[List[int]]) -> set:
    """
    Return the set of nodes reachable from `start` without passing through `blocked`.
    This is the hierarchical version of the BFS function used in resilience calculation.
    
    :param start: Starting node index
    :param blocked: Blocked node index
    :param adj_matrix: Adjacency matrix
    :return: Set of reachable node indices
    """
    visited = set()
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        if current == blocked or current in visited:
            continue
        visited.add(current)
        
        for neighbor, connected in enumerate(adj_matrix[current]):
            if connected and neighbor not in visited:
                queue.append(neighbor)
    
    return visited
