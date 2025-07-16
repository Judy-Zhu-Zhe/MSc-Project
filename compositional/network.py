import random
from collections import deque

from typing import List, Tuple, Dict, Optional

# ========================================================================================================================
#                                                      DEVICE
# ========================================================================================================================

class Device:
    def __init__(self, name: str, device_type: str, profile: dict):
        try:
            self.name = name
            self.device_type = device_type
            self.vulnerability = profile["vulnerability"]
            self.compromise_value = profile["compromise_value"]
            self.information_value = profile["information_value"]
            self.prior_information_value = 0.0
            self.internet_required = profile["internet_required"]
            self.internet_sensitive = profile["internet_sensitive"]
            self.trust_level = profile["trust_level"]
        except KeyError as e:
            raise ValueError(f"Missing required profile key: {e}")

        self.infected = False
        self.turned_down = False
        self.has_been_infected = False

    def infect(self):
        self.infected = True
        self.has_been_infected = True

    def reset(self):
        self.infected = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "device_type": self.device_type,
            "vulnerability": self.vulnerability,
            "compromise_value": self.compromise_value,
            "information_value": self.information_value,
            "prior_information_value": self.prior_information_value,
            "internet_required": self.internet_required,
            "internet_sensitive": self.internet_sensitive,
            "infected": self.infected,
            "turned_down": self.turned_down,
            "has_been_infected": self.has_been_infected,
            "trust_level": self.trust_level
        }

    @staticmethod
    def from_dict(data: dict) -> 'Device':
        device = Device(
            name=data["name"],
            device_type=data["device_type"],
            profile={
                "vulnerability": data["vulnerability"],
                "compromise_value": data["compromise_value"],
                "information_value": data["information_value"],
                "internet_required": data["internet_required"],
                "internet_sensitive": data["internet_sensitive"],
                "trust_level": data["trust_level"]
            }
        )
        device.prior_information_value = data.get("prior_information_value", 0.0)
        device.infected = data.get("infected", False)
        device.turned_down = data.get("turned_down", False)
        device.has_been_infected = data.get("has_been_infected", False)
        return device

    def __repr__(self):
        return f"<Device: {self.name}, V={self.vulnerability}, C={self.compromise_value}, I={self.information_value} (Infected={self.infected})>"


# ========================================================================================================================
#                                                      ENCLAVE
# ========================================================================================================================

class Enclave:
    def __init__(self, 
            id: int, 
            sensitivity: float = 0.0, 
            devices: List[Device] = [], 
            dist_to_internet: Optional[int] = None
            ):
        """
        :param id: Identifier for the enclave
        :param sensitivity: How quickly enclave cleansing is triggered in the event of a compromise
        :param devices: List of devices within the enclave
        :param dist_to_internet: Number of hops to the internet (used for BFS)
        """
        self.id = id
        self.sensitivity = sensitivity
        self.devices: List[Device] = devices
        self.compromised = False
        self.dist_to_internet: Optional[int] = dist_to_internet  # Number of hops to the internet
        # self.child_enclaves: List[Enclave] = []
        
    def num_devices(self) -> int:
        """Returns the number of devices in the enclave."""
        return len(self.devices)
    
    def all_devices(self) -> List[Device]:
        """Returns a list of all devices in the enclave."""
        # if self.child_enclaves:
        #     return [device for child in self.child_enclaves for device in child.all_devices()]
        return self.devices
        
    def infected_devices(self):
        """Returns a list of devices that are infected."""
        return [d for d in self.all_devices() if d.infected]
    
    def turned_down_devices(self):
        """Returns a list of devices that are turned down."""
        return [d for d in self.all_devices() if d.turned_down]
    
    def vulnerability(self) -> float:
        """Returns the vulnerability of the enclave based on its devices."""
        if not self.devices:
            return 0.5 # Enclave with no device acts purely as a firewall
        return sum(d.vulnerability for d in self.devices) / len(self.devices)
    
    def to_dict(self):
        return {
            "id": self.id,
            "sensitivity": self.sensitivity,
            "devices": [d.to_dict() for d in self.devices],
            "dist_to_internet": self.dist_to_internet
        }

    @staticmethod
    def from_dict(data):
        enclave = Enclave(
            id=data["id"],
            sensitivity=data["sensitivity"],
            devices=[Device.from_dict(d) for d in data["devices"]],
            dist_to_internet=data["dist_to_internet"],
        )
        return enclave

    def __repr__(self):
        return f"<Enclave {self.id}: {len(self.devices)} devices, s={self.sensitivity:.2f}, v={self.vulnerability():.2f}>"
    
class Internet(Enclave):
    def __init__(self):
        super().__init__(id=0)
        self.id = 0
        self.distance_to_internet = 0
        self.compromised = True
        
    def __repr__(self):
        return "<Enclave Internet>"

# ========================================================================================================================
#                                           NETWORK TOPOLOGY / SEGMENTATION
# ========================================================================================================================

class Topology:
    def __init__(self, id: int, n_enclaves: int = 1, topology: List[Tuple[int, int]] = []):
        """
        :param id: Identifier for the topology
        :param num_enclaves: Number of enclaves in the network
        :param topology: The topology of the network (list of enclaves and their neighbours)
        """
        self.id = id
        self.n_enclaves = n_enclaves
        self.adj_matrix = [[0] * n_enclaves for _ in range(n_enclaves)]
        for i, j in topology:
            try:
                self.adj_matrix[i][j] = 1
                self.adj_matrix[j][i] = 1
            except IndexError:
                raise ValueError(f"Invalid topology: {i}, {j} out of range for {n_enclaves} enclaves.")
        self.dist_to_internet = self.distances_to_target(0)  # Compute distances to the internet (index 0)
            
    def distances_to_target(self, target_index: int) -> List[Optional[int]]:
        """
        Computes the shortest distance from every node to the target using BFS.

        :param target_index: Index of the target enclave (e.g., Internet)
        :return: List of distances to the target for each node (None if unreachable)
        """
        n = len(self.adj_matrix)
        distances: List[Optional[int]] = [None] * n
        visited = [False] * n
        queue = deque([(target_index, 0)])

        while queue:
            i, dist = queue.popleft()
            if visited[i]:
                continue
            visited[i] = True
            distances[i] = dist

            for neighbor, connected in enumerate(self.adj_matrix[i]):
                if connected and not visited[neighbor]:
                    queue.append((neighbor, dist + 1))

        return distances
    
    def enclave_neighbours(self, enclave_index: int) -> List[int]:
        """Returns a list of indices of neighbours for a given enclave."""
        assert enclave_index >= 0 and enclave_index < self.n_enclaves, "Invalid enclave index."
        return [i for i, connected in enumerate(self.adj_matrix[enclave_index]) if connected == 1 and i != enclave_index]
    
    def edges(self) -> List[Tuple[int, int]]:
        """Returns a list of edges in the topology."""
        return [(i, j) for i in range(self.n_enclaves) for j in range(i, self.n_enclaves) if self.adj_matrix[i][j] == 1]
    
    def to_dict(self):
        return {
            "id": self.id,
            "num_enclaves": self.n_enclaves,
            "topology": self.edges()
        }
    
    @staticmethod
    def from_dict(data):
        topology = Topology(
            id=data["id"],
            n_enclaves=data["num_enclaves"],
            topology=data["topology"]
        )
        return topology

    def __repr__(self):
        return f"<Network topology with {self.n_enclaves} enclaves>"


class Segmentation:
    def __init__(self, 
                 topology: Topology, 
                 partition: List[List[Device]] = [], 
                 sensitivities: List[float] = [],
                 level: int = 0):
        """
        :param topology: The topology of the network
        :param partition: The partition of the network (list of devices in each enclave)
        :param sensitivities: The sensitivities of the enclaves (how quickly they are
        """
        self.topology = topology
        self.internet = Internet()
        self.enclaves: List[Enclave] = [self.internet]
        if topology:
            assert topology.n_enclaves == len(partition) == len(sensitivities), f"Solutions must have the same length. {topology.n_enclaves} != {len(partition)} != {len(sensitivities)}"
            self.create_enclaves(partition, sensitivities)

    def create_enclaves(self, partition: List[List[Device]], sensitivities: List[float]):
        """
        ### Algorithm 6 ###
        Initialisation algorithm - Simulation starting point

        :param partition: The partition of the network (list of devices in each enclave)
        :param sensitivities: The sensitivities of the enclaves
        :param vulnerabilities: The vulnerabilities of the enclaves
        """
        # Create enclaves based on the partition
        for i in range(1, len(partition)):
            e = Enclave(
                id=i, 
                sensitivity=sensitivities[i], 
                devices=partition[i],
                dist_to_internet=self.topology.dist_to_internet[i]
                )
            self.enclaves.append(e)

    def randomly_infect(self, n: int = 1, devices: Optional[Dict[str, int]] = None):
        """
        Randomly infects n devices in the network (for adaptation).
        
        :param n: Number of devices to infect
        :param devices: Optional dictionary of device types and counts to infect specific devices
        """
        all_devices = [d for e in self.enclaves for d in e.all_devices()]
        if not all_devices:
            raise ValueError("No devices available to infect in the segmentation.")
        
        if devices:
            # If specific devices are provided, infect those instead
            infected_devices: List[Device] = []
            for device_type, count in devices.items():
                matching_devices = [d for d in all_devices if d.device_type == device_type]
                if len(matching_devices) < count:
                    raise ValueError(f"Not enough devices named {device_type} to infect {count}.")
                infected_devices.extend(random.sample(matching_devices, count))
        else:
            # Randomly select n devices from all devices
            if n > len(all_devices):
                raise ValueError(f"Cannot infect {n} devices, only {len(all_devices)} available.")
            infected_devices = random.sample(all_devices, n)

        for device in infected_devices:
            device.infect()

    def num_enclaves(self) -> int:
        """Returns the number of enclaves in the segmentation."""
        return len(self.enclaves) - 1 # -1 for the internet
        
    def compromised_enclaves(self) -> List[Enclave]:
        """Returns a list of enclaves that are compromised."""
        return [e for e in self.enclaves if e.compromised]

    def num_devices(self) -> int:
        """Returns the number of devices in the segmentation."""
        return sum(len(e.devices) for e in self.enclaves)

    def all_devices(self) -> List[Device]:
        """Returns a list of all devices in the segmentation."""
        return [d for e in self.enclaves for d in e.all_devices()]

    def num_device_partition(self) -> List[int]:
        """Returns the number of devices in each partition."""
        return [len(e.devices) for e in self.enclaves if e.id != 0] # -1 for the internet
    
    def all_compromised_devices(self) -> List[Device]:
        """Returns a list of devices that are compromised."""
        return [d for e in self.enclaves for d in e.all_devices() if d.infected]
    
    def all_turned_down_devices(self) -> List[Device]:
        """Returns a list of devices that are turned down."""
        return [d for e in self.enclaves for d in e.all_devices() if d.turned_down]
    
    def partition(self) -> List[List[Device]]:
        """Returns the partition of the network into enclaves."""
        return [e.devices for e in self.enclaves]

    def sensitivities(self) -> List[float]:
        """Returns the sensitivities of the enclaves."""
        return [e.sensitivity for e in self.enclaves]
    
    def to_dict(self):
        return {
            "topology": self.topology.to_dict(),
            "partition": [[d.to_dict() for d in enclave.devices] for enclave in self.enclaves],
            "sensitivities": [enclave.sensitivity for enclave in self.enclaves]
        }
    
    @staticmethod
    def from_dict(data):
        segmentation = Segmentation(
            topology=Topology.from_dict(data["topology"]),
            partition=[[Device.from_dict(d) for d in enclave] for enclave in data["partition"]],
            sensitivities=data["sensitivities"],
        )
        return segmentation

    def __repr__(self):
        return f"<Network Segmentation with {self.num_enclaves()} enclaves and device partition : {[len(p) for p in self.partition()]}>"

class SegmentationNode:
    def __init__(self, seg: Segmentation, level: int = 0, parent=None, trust_level: Optional[str] = None):
        self.seg = seg
        self.level = level
        self.parent: Optional[SegmentationNode] = parent
        self.children: Dict[int, SegmentationNode] = {}  # Enclave index -> child node
        self.trust_level: Optional[str] = trust_level

    def add_child(self, enclave_id: int, child_seg: 'SegmentationNode'):
        self.children[enclave_id] = child_seg
        child_seg.parent = self
    
    def print_details(self, indent: int = 0):
        prefix = '  ' * indent
        trust_info = f"({self.trust_level.upper()} Trust)" if self.trust_level else ""
        enclave_info = f"{self.seg.num_enclaves()} Enclaves"
        partition_info = f"{self.seg.num_device_partition()}"
        topology_info = f"Topology: {self.seg.topology.edges()}"
        print(f"{prefix}{f"Segmentation Level {self.level}"}{trust_info}, {enclave_info}: {partition_info}, {topology_info}")
        if not self.children:
            for i, enclave in enumerate(self.seg.enclaves):
                if i == 0:
                    print(f"{prefix}  Internet Enclave 0")
                    continue
                print(f"{prefix}  Enclave: {enclave}")
                for device in enclave.devices:
                    print(f"{prefix}    - Device: {device}")
            print()
        else:
            for idx, child in self.children.items():
                child.print_details(indent=indent+1)

    def to_dict(self):
        return {
            "seg": self.seg.to_dict(),
            "level": self.level,
            "children": {idx: child.to_dict() for idx, child in self.children.items()},
            "trust_level": self.trust_level
        }
    
    @staticmethod
    def from_dict(data):
        seg = Segmentation.from_dict(data["seg"])
        node = SegmentationNode(seg, data["level"], data["trust_level"])
        for idx, child_data in data["children"].items():
            child_node = SegmentationNode.from_dict(child_data)
            node.add_child(int(idx), child_node)
        return node

