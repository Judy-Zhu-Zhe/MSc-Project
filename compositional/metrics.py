from typing import List
import math
from collections import deque
from itertools import combinations

from network import Segmentation
from simulation import Simulation

def security_loss(simulations: List[Simulation]) -> float:
    """Calculate the average security loss of a network across a number of simulations.
    
    :param networks: List of networks across simulations.
    :return: Average security loss across all networks.
    """
    assert simulations, "Simulation list cannot be empty"
    loss = 0
    for s in simulations:
        seg = s.segmentation
        # Sum of information value of all infected devices
        information_loss = sum([d.prior_information_value for d in seg.all_compromised_devices()]) 
        # Sum of compromise value of all turned down devices
        compromise_loss = sum([d.compromise_value for d in seg.all_turned_down_devices()])
        # The cleansing loss if any
        cleansing_loss = s.cleansing_loss
        # Sum of all losses
        loss += information_loss + compromise_loss + cleansing_loss
    return loss / len(simulations)

def performance_loss(segmentation: Segmentation) -> float:
    """Calculate the average performance loss of a network segmentation.
    
    :param network: Network segmentation.
    :return: Increased latency of all performance affecting device.
    """
    loss = 0
    for enclave in segmentation.enclaves:
        num = 0
        # Count the number of performance affecting devices in the enclave
        for device in enclave.devices:
            if device.internet_sensitive:
                num += 1
        # Increased latency = the number of performance affecting devices * the distance to the internet
        if enclave.dist_to_internet:
            loss += num * enclave.dist_to_internet
        elif num > 0:
            # If an enclave with performing affecting devices is not connected to the Internet, return infinity
            return float("inf")
    return loss

def resilience_loss(segmentation: Segmentation) -> float:
    """Calculate the resilience loss of segmentation as the average loss incurred by removing one enclave.
    
    :param network: Network segmentation.
    :return: Average resilience loss across all enclaves.
    """
    def bfs_connected(start: int, blocked: int, adj_matrix: List[List[int]]) -> set:
        """Return the set of nodes reachable from `start` without passing through `blocked`."""
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

    loss = 0
    n_enclaves = segmentation.topology.n_enclaves

    for id in range(1, n_enclaves):  # Skip Internet
        e = segmentation.enclaves[id]
        reachable = bfs_connected(start=0, blocked=id, adj_matrix=segmentation.topology.adj_matrix)

        # Loss incurred inside the enclave
        enclave_loss = sum(d.compromise_value for d in e.devices)

        # Loss from enclaves that become disconnected
        for i in range(1, n_enclaves):
            if i != id and i not in reachable:
                disconnected = segmentation.enclaves[i]
                enclave_loss += sum(d.compromise_value for d in disconnected.devices)

        loss += enclave_loss

    return loss / (n_enclaves - 1) if n_enclaves > 1 else 0


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


def attack_surface_exposure(seg: Segmentation, vuln_threshold: float = 7.0, info_threshold: float = 7.0) -> float:
    """
    Penalty score for high vulnerability devices based on their distance from Internet and information value.
    
    :param seg: The segmentation object
    :param vuln_threshold: Vulnerability score threshold to consider "high"
    :param info_threshold: Information value threshold to consider "high"
    :return: Float penalty score for attack surface exposure
    """
    total_exposure_score = 0.0
    
    # Calculate exposure score for each enclave
    for i, enclave in enumerate(seg.enclaves):
        if i == 0:  # Skip Internet enclave
            continue
        
        # Calculate exposure for each device in this enclave
        for device in enclave.all_devices():
            # Only consider high vulnerability devices
            if device.vulnerability >= vuln_threshold:
                # Base penalty: vulnerability score
                base_penalty = device.vulnerability
                
                # Distance penalty: Use exponential decay
                distance_penalty = base_penalty * (0.5 ** enclave.dist_to_internet)
                
                # Information value scaling: higher information value = higher penalty
                info_scaling = device.information_value / 10.0  # Normalize to 0-1 range
                
                # Final exposure score for this device
                device_exposure = distance_penalty * (1.0 + info_scaling)
                
                total_exposure_score += device_exposure
    
    return total_exposure_score

def trust_separation_score(seg: Segmentation) -> float:
    """
    Penalty for compromised values within and between enclaves.
    Optimized version with O(D + E²) complexity instead of O(D² + E² * D).
    
    :param seg: The segmentation object
    :return: Float penalty score based on compromised values
    """
    penalty = 0.0

    # Pre-calculate average compromise values for each enclave to avoid repeated calculations
    enclave_avg_compromise = {}
    for i, enclave in enumerate(seg.enclaves):
        devices = enclave.all_devices()
        if devices:
            avg_comp = sum(d.compromise_value for d in devices) / len(devices)
            enclave_avg_compromise[i] = avg_comp

    # Intra-Enclave Mismatch - Optimized approach
    for enclave in seg.enclaves:
        devices = enclave.all_devices()
        if len(devices) < 2:
            continue
            
        # Instead of checking all combinations, use statistical approach
        # Calculate variance of compromise values as a proxy for mismatch
        comp_values = [d.compromise_value for d in devices]
        mean_comp = sum(comp_values) / len(comp_values)
        variance = sum((comp - mean_comp) ** 2 for comp in comp_values) / len(comp_values)
        
        # Convert variance to penalty (higher variance = higher penalty)
        if variance > 4.0:  # Threshold for significant variance
            penalty += (variance / 4.0) * len(devices) * 0.5  # Scale by device count

    # Inter-Enclave Mismatch - Use pre-calculated averages
    for i in range(len(seg.enclaves)):
        for j in range(i + 1, len(seg.enclaves)):
            if seg.topology.adj_matrix[i][j]:
                # Use pre-calculated averages
                if i in enclave_avg_compromise and j in enclave_avg_compromise:
                    avg_comp_i = enclave_avg_compromise[i]
                    avg_comp_j = enclave_avg_compromise[j]
                    
                    # Penalize connecting enclaves with very different compromise profiles
                    comp_diff = abs(avg_comp_i - avg_comp_j)
                    if comp_diff > 6:  # Threshold for significant mismatch
                        penalty += comp_diff

    return penalty

def trust_separation_score_nested(seg: Segmentation) -> float:
    """
    Penalty for compromised values within and between enclaves.
    
    :param seg: The segmentation object
    :return: Float penalty score based on compromised values
    """
    penalty = 0.0

    # Intra-Enclave Mismatch - Penalize mixing devices with very different compromise values
    for enclave in seg.enclaves:
        devices = enclave.all_devices()
        for d1, d2 in combinations(devices, 2):
            # Penalize large differences in compromise values within same enclave
            comp_diff = abs(d1.compromise_value - d2.compromise_value)
            if comp_diff > 3:  # Threshold for significant difference
                penalty += (comp_diff / 5) ** 2

    # Inter-Enclave Mismatch - Penalize connecting enclaves with very different compromise profiles
    for i in range(len(seg.enclaves)):
        for j in range(i + 1, len(seg.enclaves)):
            if seg.topology.adj_matrix[i][j]:
                # Calculate average compromise values for each enclave
                enclave_i_devices = seg.enclaves[i].all_devices()
                enclave_j_devices = seg.enclaves[j].all_devices()
                
                if enclave_i_devices and enclave_j_devices:
                    # Average compromise value for each enclave
                    avg_comp_i = sum(d.compromise_value for d in enclave_i_devices) / len(enclave_i_devices)
                    avg_comp_j = sum(d.compromise_value for d in enclave_j_devices) / len(enclave_j_devices)
                    
                    # Penalize connecting enclaves with very different compromise profiles
                    comp_diff = abs(avg_comp_i - avg_comp_j)
                    if comp_diff > 6:  # Threshold for significant mismatch
                        penalty += comp_diff

    return penalty


def sensitivity_penalty(segmentation: Segmentation, 
                                info_threshold: float = 7.5, 
                                comp_threshold: float = 5.0,
                                sensitivity_threshold: float = 0.5) -> float:
    """
    This function identifies devices that have both high information value and 
    high compromise value, and penalizes their placement in enclaves with 
    low sensitivity (which would be slow to respond to compromises).
    
    :param segmentation: The network segmentation object
    :param info_threshold: Information value threshold to consider "high"
    :param comp_threshold: Compromise value threshold to consider "high"
    :param sensitivity_threshold: Sensitivity threshold below which is considered "low"
    :return: Float penalty score for sensitivity mismatches
    """
    total_penalty = 0.0
    
    for enclave in segmentation.enclaves:
        # Skip Internet enclave (id 0)
        if enclave.id == 0:
            continue
            
        # Check if this enclave has low sensitivity
        if enclave.sensitivity < sensitivity_threshold:
            # Look for high-value devices in this low-sensitivity enclave
            for device in enclave.all_devices():
                # Check if device has both high information value and high compromise value
                if (device.information_value >= info_threshold and 
                    device.compromise_value >= comp_threshold):
                    
                    # Calculate penalty based on the severity of the mismatch
                    # Higher penalty for:
                    # 1. Higher information value
                    # 2. Higher compromise value  
                    # 3. Lower enclave sensitivity
                    
                    # Normalize values to 0-1 range for penalty calculation
                    info_factor = min(device.information_value / 10.0, 1.0)
                    comp_factor = min(device.compromise_value / 10.0, 1.0)
                    sensitivity_factor = max(0.0, 1.0 - (enclave.sensitivity / sensitivity_threshold))
                    
                    # Calculate device penalty: product of all factors
                    device_penalty = info_factor * comp_factor * sensitivity_factor * 10.0
                    
                    # Additional penalty multiplier for very high-value devices in very low-sensitivity enclaves
                    if device.information_value >= 9.0 and device.compromise_value >= 9.0 and enclave.sensitivity < 0.7:
                        device_penalty *= 2.0
                    
                    total_penalty += device_penalty
    
    return total_penalty


