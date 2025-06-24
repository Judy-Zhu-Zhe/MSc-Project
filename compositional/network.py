import random
import yaml
from collections import deque

from typing import List, Tuple, Dict, Optional

# ========================================================================================================================
#                                                      DEVICE
# ========================================================================================================================

device_profiles_path = "configs\device_profiles.yaml"

with open(device_profiles_path, "r") as f:
    profile_data = yaml.safe_load(f)

DEVICE_GROUPS = profile_data["device_groups"]
DEVICE_VALUES = profile_data["device_values"]

class Device:
    def __init__(self, name: str, device_type: str = None):
        """
        :param name: Identifier for the device
        :param device_type: Type of device
        """
        self.name = name
        self.device_type = device_type
        self.device_group = DEVICE_GROUPS.get(device_type, [])
        self.compromise_value = DEVICE_VALUES.get(device_type, (0, 0))[0]
        self.prior_information_value = 0
        self.information_value = DEVICE_VALUES.get(device_type, (0, 0))[1]
        self.has_been_infected = False
        self.infected = False  # TODO: returns to an operational state after a certain time depending on the device type
        self.turned_down = False

    def infect(self):
        self.infected = True
        self.has_been_infected = True

    def reset(self):
        self.infected = False

    def to_dict(self):
        return {
            "name": self.name,
            "device_type": self.device_type,
            "device_group": self.device_group,
            "compromise_value": self.compromise_value,
            "information_value": self.information_value
        }

    @staticmethod
    def from_dict(data):
        device = Device(name=data["name"], device_type=data["device_type"])
        device.device_group = data["device_group"]
        device.compromise_value = data["compromise_value"]
        device.information_value = data["information_value"]
        return device

    def __repr__(self):
        return f"<Device {self.name} ({self.compromise_value}, {self.information_value}) (Infected={self.infected})>"


# ========================================================================================================================
#                                                      ENCLAVE
# ========================================================================================================================

class Enclave:
    def __init__(self, 
            id: int, 
            vulnerability: float = 0, 
            sensitivity: float = 0, 
            devices: List[Device] = [], 
            low_value_devices: List[Device] = [], 
            dist_to_internet: int = None
            ):
        """
        :param id: Identifier for the enclave
        :param vulnerability: Vulnerability of the enclave to compromise
        :param sensitivity: How quickly enclave cleansing is triggered in the event of a compromise
        :param devices: List of devices within the enclave
        :param low_value_devices: List of low-value devices within the enclave
        :param dist_to_internet: Number of hops to the internet (used for BFS)
        """
        self.id = id
        self.vulnerability = vulnerability
        self.sensitivity = sensitivity
        self.devices: List[Device] = devices
        self.low_value_devices: List[Device] = low_value_devices
        self.compromised = False
        self.cleansing_loss = 0
        self.dist_to_internet: Optional[int] = dist_to_internet  # Number of hops to the internet
        
    def num_devices(self) -> int:
        """Returns the number of non-low-value devices in the enclave."""
        return len(self.devices)
    
    def all_devices(self) -> List[Device]:
        """Returns a list of all devices in the enclave, including low-value devices."""
        return self.devices + self.low_value_devices
        
    def infected_devices(self):
        """Returns a list of devices that are infected."""
        return [d for d in self.all_devices() if d.infected]
    
    def turned_down_devices(self):
        """Returns a list of devices that are turned down."""
        return [d for d in self.all_devices() if d.turned_down]
    
    def to_dict(self):
        return {
            "id": self.id,
            "vulnerability": self.vulnerability,
            "sensitivity": self.sensitivity,
            "devices": [d.to_dict() for d in self.devices],
            "low_value_devices": [d.to_dict() for d in self.low_value_devices],
            "dist_to_internet": self.dist_to_internet
        }

    @staticmethod
    def from_dict(data):
        enclave = Enclave(
            id=data["id"],
            vulnerability=data["vulnerability"],
            sensitivity=data["sensitivity"],
            devices=[Device.from_dict(d) for d in data["devices"]],
            low_value_devices=[Device.from_dict(d) for d in data["low_value_devices"]],
            dist_to_internet=data["dist_to_internet"],
        )
        return enclave

    def __repr__(self):
        return f"<Enclave {self.id}: {len(self.devices)} devices, Compromised={self.compromised}>"
    
class Internet(Enclave):
    def __init__(self):
        super().__init__("Internet")
        self.id = 0
        self.distance_to_internet = 0
        self.compromised = True
        
    def __repr__(self):
        return "<Enclave Internet>"

# ========================================================================================================================
#                                           NETWORK TOPOLOGY / SEGMENTATION
# ========================================================================================================================

class Topology:
    def __init__(self, id: int, n_enclaves: int = 1, topology: List[Tuple[int]] = []):
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
        distances = [None] * n
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
                 topology: Topology = None, 
                 partition: List[List[Device]] = [], 
                 sensitivities: List[float] = [], 
                 vulnerabilities: List[float] = [],
                 n_low_value_device: int = 0):
        """
        :param topology: The topology of the network
        :param partition: The partition of the network (list of devices in each enclave)
        :param sensitivities: The sensitivities of the enclaves (how quickly they are
        """
        self.topology = topology
        self.internet = Internet()
        self.enclaves: List[Enclave] = [self.internet]
        if topology:
            assert topology.n_enclaves == len(partition) == len(sensitivities) == len(vulnerabilities), "Solutions must have the same length."
            self.create_enclaves(partition, sensitivities, vulnerabilities, n_low_value_device)

    def create_enclaves(self, partition: List[List[Device]], sensitivities: List[float], vulnerabilities: List[float], n_low_value_device: int = 0):
        """
        ### Algorithm 6 ###
        Initialisation algorithm - Simulation starting point

        :param partition: The partition of the network (list of devices in each enclave)
        :param sensitivities: The sensitivities of the enclaves
        :param vulnerabilities: The vulnerabilities of the enclaves
        """
        # Create enclaves based on the partition
        for i in range(1, len(partition)):
            num_low = random.randint(1, n_low_value_device) if n_low_value_device > 0 else 0
            low_value_devices = [Device(f"Low value device {i}-{j+1}", device_type="Low value device") for j in range(num_low)]
            e = Enclave(
                id=i, 
                vulnerability=vulnerabilities[i], 
                sensitivity=sensitivities[i], 
                devices=partition[i],
                low_value_devices=low_value_devices,
                dist_to_internet=self.topology.dist_to_internet[i]
                )
            self.enclaves.append(e)

    def randomly_infect(self, n: int = 1, devices: Dict[str, int] = None):
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
        
    def compromised_enclaves(self) -> List[Enclave]:
        """Returns a list of enclaves that are compromised."""
        return [e for e in self.enclaves if e.compromised]
    
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
    
    def vulnerabilities(self) -> List[float]:
        """Returns the vulnerabilities of the enclaves."""
        return [e.vulnerability for e in self.enclaves]
    
    def to_dict(self):
        return {
            "topology": self.topology.to_dict(),
            "partition": [[d.to_dict() for d in enclave.devices] for enclave in self.enclaves],
            "sensitivities": [enclave.sensitivity for enclave in self.enclaves],
            "vulnerabilities": [enclave.vulnerability for enclave in self.enclaves],
            # "enclaves": [enclave.to_dict() for enclave in self.enclaves]
        }
    
    @staticmethod
    def from_dict(data):
        segmentation = Segmentation(
            topology=Topology.from_dict(data["topology"]),
            partition=[[Device.from_dict(d) for d in enclave] for enclave in data["partition"]],
            sensitivities=data["sensitivities"],
            vulnerabilities=data["vulnerabilities"],
        )
        return segmentation

    def __repr__(self):
        return f"<Network Segmentation with {len(self.enclaves)} enclaves and device partition : {[len(p) for p in self.partition()]}>"

