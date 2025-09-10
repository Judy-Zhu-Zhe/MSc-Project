from typing import List, Dict, Optional
import math
from network import SegmentationNode
from metrics import topology_distance as base_topology_distance


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
