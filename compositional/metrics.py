from typing import List
import math
from collections import deque

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
        cleansing_loss = s.total_loss
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
            if "performance_affecting" in device.device_group:
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

