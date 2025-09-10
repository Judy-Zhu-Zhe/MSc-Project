import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random

from network import Device, Topology, Segmentation, SegmentationNode
from config import ConfigManager


class KWayMinimumCut:
    """
    Implements k-way minimum cut algorithm for network segmentation.
    Uses spectral clustering and minimum cut optimization to partition devices into enclaves.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.devices = config_manager.devices
        self.device_profiles = config_manager.device_profiles
        
    def create_device_graph(self) -> nx.Graph:
        """
        Create a weighted graph where devices are nodes and edges represent 
        security relationships based on device characteristics.
        
        Returns:
            nx.Graph: Weighted graph representing device relationships
        """
        G = nx.Graph()
        
        # Add all devices as nodes
        for device in self.devices:
            G.add_node(device.name, device=device)
        
        # Create edges based on security relationships
        for i, device1 in enumerate(self.devices):
            for j, device2 in enumerate(self.devices):
                if i >= j:  # Avoid duplicate edges and self-loops
                    continue
                    
                # Calculate edge weight based on security relationship
                weight = self._calculate_edge_weight(device1, device2)
                if weight > 0:
                    G.add_edge(device1.name, device2.name, weight=weight)
        
        return G
    
    def _calculate_edge_weight(self, device1: Device, device2: Device) -> float:
        """
        Calculate edge weight between two devices based on security characteristics.
        Higher weight = stronger connection = should be in same enclave.
        
        Args:
            device1: First device
            device2: Second device
            
        Returns:
            float: Edge weight (0.0 to 1.0)
        """
        # Base weight from device type similarity
        type_similarity = 1.0 if device1.device_type == device2.device_type else 0.3
        
        # Security value similarity (devices with similar compromise/information values)
        comp_diff = abs(device1.compromise_value - device2.compromise_value)
        info_diff = abs(device1.information_value - device2.information_value)
        
        # Normalize differences to 0-1 range
        comp_similarity = max(0, 1 - (comp_diff / 10.0))
        info_similarity = max(0, 1 - (info_diff / 10.0))
        
        # Internet sensitivity compatibility
        internet_compatibility = 1.0 if device1.internet_sensitive == device2.internet_sensitive else 0.5
        
        # Vulnerability similarity (devices with similar vulnerability levels)
        vuln_diff = abs(device1.vulnerability - device2.vulnerability)
        vuln_similarity = max(0, 1 - (vuln_diff / 10.0))
        
        # Combine all factors with weights
        weight = (
            0.3 * type_similarity +
            0.25 * comp_similarity +
            0.25 * info_similarity +
            0.1 * internet_compatibility +
            0.1 * vuln_similarity
        )
        
        return weight
    
    def spectral_clustering_partition(self, graph: nx.Graph, n_enclaves: int) -> List[List[str]]:
        """
        Use spectral clustering to partition devices into enclaves.
        
        Args:
            graph: Device relationship graph
            n_enclaves: Number of enclaves to create
            
        Returns:
            List[List[str]]: List of enclaves, each containing device names
        """
        if n_enclaves <= 1:
            return [list(graph.nodes())]
        
        # Convert to adjacency matrix
        adj_matrix = nx.adjacency_matrix(graph, weight='weight').toarray()
        
        # Create Laplacian matrix
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree_matrix - adj_matrix
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(laplacian)
        
        # Use k smallest non-zero eigenvectors for clustering
        # Skip the first eigenvalue (should be 0)
        k_eigenvecs = eigenvecs[:, 1:n_enclaves]
        
        # Normalize rows for better clustering
        row_norms = np.linalg.norm(k_eigenvecs, axis=1)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        normalized_eigenvecs = k_eigenvecs / row_norms[:, np.newaxis]
        
        # Use k-means clustering on the normalized eigenvectors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_enclaves, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_eigenvecs)
        
        # Group devices by cluster
        device_names = list(graph.nodes())
        enclaves = [[] for _ in range(n_enclaves)]
        for i, label in enumerate(cluster_labels):
            enclaves[label].append(device_names[i])
        
        return enclaves
    
    def minimum_cut_refinement(self, graph: nx.Graph, initial_partition: List[List[str]], 
                              n_enclaves: int) -> List[List[str]]:
        """
        Refine the initial partition using minimum cut optimization.
        
        Args:
            graph: Device relationship graph
            initial_partition: Initial partition from spectral clustering
            n_enclaves: Number of enclaves
            
        Returns:
            List[List[str]]: Refined partition
        """
        refined_partition = [list(enclave) for enclave in initial_partition]
        
        # Iteratively refine each enclave
        for iteration in range(3):  # Limit iterations to avoid infinite loops
            improved = False
            
            for i in range(n_enclaves):
                for j in range(i + 1, n_enclaves):
                    # Try to improve partition by swapping devices between enclaves
                    if self._try_swap_devices(graph, refined_partition, i, j):
                        improved = True
            
            if not improved:
                break
        
        return refined_partition
    
    def _try_swap_devices(self, graph: nx.Graph, partition: List[List[str]], 
                          enclave1_idx: int, enclave2_idx: int) -> bool:
        """
        Try to improve partition by swapping devices between two enclaves.
        
        Args:
            graph: Device relationship graph
            partition: Current partition
            enclave1_idx: Index of first enclave
            enclave2_idx: Index of second enclave
            
        Returns:
            bool: True if swap improved the partition, False otherwise
        """
        enclave1 = partition[enclave1_idx]
        enclave2 = partition[enclave2_idx]
        
        if len(enclave1) == 0 or len(enclave2) == 0:
            return False
        
        # Calculate current cut weight
        current_cut_weight = self._calculate_partition_cut_weight(graph, partition)
        
        # Try swapping each device from enclave1 to enclave2
        best_swap = None
        best_improvement = 0
        
        for device_name in enclave1:
            # Create temporary partition for testing
            temp_partition = [list(enclave) for enclave in partition]
            temp_partition[enclave1_idx].remove(device_name)
            temp_partition[enclave2_idx].append(device_name)
            
            # Calculate new cut weight
            new_cut_weight = self._calculate_partition_cut_weight(graph, temp_partition)
            improvement = current_cut_weight - new_cut_weight
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = device_name
        
        # Apply the best swap if it improves the partition
        if best_swap and best_improvement > 0:
            partition[enclave1_idx].remove(best_swap)
            partition[enclave2_idx].append(best_swap)
            return True
        
        return False
    
    def _calculate_partition_cut_weight(self, graph: nx.Graph, partition: List[List[str]]) -> float:
        """
        Calculate the total weight of edges that cross between enclaves.
        Lower weight = better partition.
        
        Args:
            graph: Device relationship graph
            partition: Current partition
            
        Returns:
            float: Total cut weight
        """
        cut_weight = 0.0
        
        # Check all edges in the graph
        for edge in graph.edges(data=True):
            device1, device2, edge_data = edge
            weight = edge_data.get('weight', 0)
            
            # Find which enclaves contain these devices
            enclave1_idx = None
            enclave2_idx = None
            
            for i, enclave in enumerate(partition):
                if device1 in enclave:
                    enclave1_idx = i
                if device2 in enclave:
                    enclave2_idx = i
                if enclave1_idx is not None and enclave2_idx is not None:
                    break
            
            # If devices are in different enclaves, add to cut weight
            if enclave1_idx != enclave2_idx:
                cut_weight += weight
        
        return cut_weight
    
    def generate_sensitivities(self, partition: List[List[str]], base_sensitivity: float = 0.5) -> List[float]:
        """
        Generate sensitivity values for enclaves based on their device characteristics.
        
        Args:
            partition: Device partition
            base_sensitivity: Base sensitivity value
            
        Returns:
            List[float]: Sensitivity values for each enclave
        """
        sensitivities = [0.0]  # Internet enclave always has 0 sensitivity
        
        for enclave_devices in partition:
            if not enclave_devices:
                sensitivities.append(base_sensitivity)
                continue
            
            # Calculate average compromise and information values for the enclave
            total_compromise = 0
            total_information = 0
            count = 0
            
            for device_name in enclave_devices:
                device = next(d for d in self.devices if d.name == device_name)
                total_compromise += device.compromise_value
                total_information += device.information_value
                count += 1
            
            if count > 0:
                avg_compromise = total_compromise / count
                avg_information = total_information / count
                
                # Higher sensitivity for enclaves with high-value devices
                sensitivity = base_sensitivity + (avg_compromise + avg_information) / 20.0
                sensitivity = min(1.0, max(0.1, sensitivity))  # Clamp between 0.1 and 1.0
            else:
                sensitivity = base_sensitivity
            
            sensitivities.append(sensitivity)
        
        return sensitivities
    
    def create_topology(self, n_enclaves: int, max_degree: int = 3) -> List[Tuple[int, int]]:
        """
        Create a simple topology connecting enclaves.
        
        Args:
            n_enclaves: Number of enclaves
            max_degree: Maximum degree for any enclave
            
        Returns:
            List[Tuple[int, int]]: List of edges between enclaves
        """
        if n_enclaves <= 1:
            return []
        
        # Start with a tree structure (minimum spanning tree)
        edges = []
        for i in range(1, n_enclaves):
            # Connect to a random previous enclave
            parent = random.randint(0, i-1)
            edges.append((parent, i))
        
        # Add some additional edges to improve connectivity while respecting max_degree
        for i in range(n_enclaves):
            current_degree = sum(1 for edge in edges if i in edge)
            if current_degree < max_degree:
                # Try to add edges to other enclaves
                for j in range(i+1, n_enclaves):
                    if j not in [neighbor for edge in edges if i in edge for neighbor in edge if neighbor != i]:
                        j_degree = sum(1 for edge in edges if j in edge)
                        if j_degree < max_degree:
                            edges.append((i, j))
                            break
        
        return edges
    
    def run_kway_segmentation(self, n_enclaves: int, max_degree: int = 3) -> Segmentation:
        """
        Run the complete k-way minimum cut segmentation algorithm.
        
        Args:
            n_enclaves: Number of enclaves to create
            max_degree: Maximum degree for topology
            
        Returns:
            Segmentation: Complete network segmentation
        """
        print(f"Running k-way minimum cut segmentation for {n_enclaves} enclaves...")
        
        # Step 1: Create device relationship graph
        print("Creating device relationship graph...")
        graph = self.create_device_graph()
        
        # Step 2: Spectral clustering for initial partition
        print("Performing spectral clustering...")
        initial_partition = self.spectral_clustering_partition(graph, n_enclaves)
        
        # Step 3: Minimum cut refinement
        print("Refining partition with minimum cut optimization...")
        refined_partition = self.minimum_cut_refinement(graph, initial_partition, n_enclaves)
        
        # Step 4: Generate sensitivities
        print("Generating enclave sensitivities...")
        sensitivities = self.generate_sensitivities(refined_partition)
        
        # Step 5: Create topology
        print("Creating network topology...")
        topology_edges = self.create_topology(n_enclaves + 1, max_degree)
        topology = Topology(id=0, n_enclaves=n_enclaves + 1, topology=topology_edges)
        
        # Step 6: Convert device names back to device objects
        device_partition = []
        for enclave_devices in refined_partition:
            enclave_device_objects = []
            for device_name in enclave_devices:
                device = next(d for d in self.devices if d.name == device_name)
                enclave_device_objects.append(device)
            device_partition.append(enclave_device_objects)
        
        # Add empty Internet enclave at the beginning
        device_partition.insert(0, [])
        
        # Step 7: Create and return segmentation
        segmentation = Segmentation(
            topology=topology,
            partition=device_partition,
            sensitivities=sensitivities
        )
        
        print(f"Segmentation complete! Created {n_enclaves} enclaves with {segmentation.num_devices()} devices.")
        return segmentation
    
    def run_hierarchical_segmentation(self, configs: List[Dict]) -> SegmentationNode:
        """
        Run hierarchical k-way segmentation across multiple levels.
        
        Args:
            configs: List of configuration dictionaries for each level
            
        Returns:
            SegmentationNode: Root of hierarchical segmentation tree
        """
        print("Running hierarchical k-way segmentation...")
        
        if not configs:
            raise ValueError("No configurations provided for hierarchical segmentation")
        
        # Start with root level
        root_config = configs[0]
        n_enclaves = root_config.get('n_enclaves', 3)
        max_degree = root_config.get('max_degree', 3)
        
        # Create root segmentation
        root_seg = self.run_kway_segmentation(n_enclaves, max_degree)
        root_node = SegmentationNode(root_seg, level=0)
        
        # Recursively create child segmentations
        if len(configs) > 1:
            self._create_child_segmentations(root_node, configs[1:])
        
        return root_node
    
    def _create_child_segmentations(self, parent_node: SegmentationNode, configs: List[Dict]):
        """
        Recursively create child segmentations for a hierarchical structure.
        
        Args:
            parent_node: Parent segmentation node
            configs: Remaining configurations for child levels
        """
        if not configs:
            return
        
        config = configs[0]
        n_enclaves = config.get('n_enclaves', 3)
        max_degree = config.get('max_degree', 3)
        
        # For each enclave in the parent, create a child segmentation
        for enclave_idx, enclave in enumerate(parent_node.seg.enclaves):
            if enclave_idx == 0:  # Skip Internet enclave
                continue
            
            # Get devices for this enclave
            enclave_devices = enclave.all_devices()
            if not enclave_devices:
                continue
            
            # Create a temporary config manager for this enclave
            temp_cm = self._create_temp_config_manager(enclave_devices)
            
            # Create child segmentation
            child_kway = KWayMinimumCut(temp_cm)
            child_seg = child_kway.run_kway_segmentation(n_enclaves, max_degree)
            
            # Create child node
            child_node = SegmentationNode(child_seg, level=config.get('level', 1))
            parent_node.add_child(enclave_idx, child_node)
            
            # Recursively create grandchildren
            if len(configs) > 1:
                self._create_child_segmentations(child_node, configs[1:])
    
    def _create_temp_config_manager(self, devices: List[Device]) -> ConfigManager:
        """
        Create a temporary config manager for a subset of devices.
        
        Args:
            devices: List of devices for the subset
            
        Returns:
            ConfigManager: Temporary configuration manager
        """
        # This is a simplified version - in practice, you might want to
        # create a more sophisticated temporary config manager
        class TempConfigManager:
            def __init__(self, devices, device_profiles):
                self.devices = devices
                self.device_profiles = device_profiles
        
        return TempConfigManager(devices, self.device_profiles)


def run_kway_experiment(config_manager: ConfigManager, n_enclaves: int = 5, 
                       max_degree: int = 3, hierarchical: bool = False) -> SegmentationNode:
    """
    Convenience function to run k-way segmentation experiment.
    
    Args:
        config_manager: Configuration manager
        n_enclaves: Number of enclaves to create
        max_degree: Maximum degree for topology
        hierarchical: Whether to run hierarchical segmentation
        
    Returns:
        SegmentationNode: Root of segmentation tree
    """
    kway = KWayMinimumCut(config_manager)
    
    if hierarchical:
        # Use the config from config manager for hierarchical structure
        configs = []
        for config in config_manager.config.configs:
            configs.append({
                'n_enclaves': config.n_enclaves + 1,
                'max_degree': config.n_enclaves - 1,  # Reasonable max degree
                'level': config.level
            })
        return kway.run_hierarchical_segmentation(configs)
    else:
        # Single-level segmentation
        seg = kway.run_kway_segmentation(n_enclaves, max_degree)
        return SegmentationNode(seg, level=0)

