import random
from collections import deque

from typing import List

DYN_DEVICES = 5  # Number of dynamic low value devices in the network

DEVICE_GROUPS = {
    "Authentication server": ["high_value"],
    "Syslog server": ["high_value"],
    "Web server": ["performance_affecting", "resilience_affected"],
    "E-mail server": ["performance_affecting", "resilience_affected"],
    "DNS server": ["performance_affecting", "resilience_affected"],
    "Database": ["performance_affecting"],
    "Employee computer": ["resilience_affected"]
}

# Device values are tuples of (C, I) where 
#   C is the compromise value representing the impact of this device ceasing operation
#   and I is the information value representing the impact of this device being infiltrated and the data from it being stolen
DEVICE_VALUES = {
    "Printer": (0.05, 0.02),
    "Employee computer": (0.05, 0.1),
    "Printer server": (0.1, 0.2),
    "DNS Server": (0.2, 0.5),
    "DHCP Server": (0.1, 0.1),
    "E-mail server": (7, 9),
    "Web server": (10, 6),
    "SQL Database": (70, 100),
    "Syslog server": (100, 100),
    "Authentication server": (100, 100),
    "Low value device": (0.001, 0.001),
}

# ========================================================================================================================
#                                                      DEVICE
# ========================================================================================================================

class Device:
    def __init__(self, name: str, device_type: str = None):
        """
        :param name: Identifier for the device
        :param device_type: Type of device
        """
        self.name = name
        self.device_type = device_type
        self.device_group = DEVICE_GROUPS.get(device_type, [])
        # TODO: Figure prior and true value
        self.compromise_value = DEVICE_VALUES.get(device_type, (0, 0))[0]
        self.prior_information_value = DEVICE_VALUES.get(device_type, (0, 0))[1]
        self.true_information_value = self.prior_information_value
        self.group = DEVICE_GROUPS.get(device_type, [])
        self.has_been_infected = False
        self.infected = False  # TODO: returns to an operational state after a certain time depending on the device type
        self.turned_down = False

    def reset(self):
        self.infected = False

    def __repr__(self):
        return f"<Device {self.name} ({self.compromise_value}, {self.true_information_value}) (Infected={self.infected})>"


# ========================================================================================================================
#                                                      ENCLAVE
# ========================================================================================================================

class Enclave:
    def __init__(self, id: int, vulnerability: float = 0, sensitivity: float = 0, devices: List[Device] = [], neighbours: List[int] = []):
        """
        :param id: Identifier for the enclave
        :param vulnerability: Vulnerability of the enclave to compromise
        :param sensitivity: How quickly enclave cleansing is triggered in the event of a compromise
        :param devices: List of devices within the enclave
        :param neighbours: List of neighbouring enclave indices
        """
        self.id = id
        self.vulnerability = vulnerability
        self.sensitivity = sensitivity
        self.devices: List[Device] = devices
        self.neighbours: List[int] = neighbours
        self.compromised = False
        self.cleansing_loss = 0
        self.distance_to_internet: int | None = None  # Number of hops to the internet

    def add_device(self, device: Device):
        self.devices.append(device)

    def remove_device(self, device: Device):
        if device in self.devices:
            self.devices.remove(device)
        else:
            raise ValueError(f"Device {device.name} not found in enclave {self.id}.")
        
    def num_devices(self) -> int:
        """Returns the number of devices in the enclave."""
        return len(self.devices)
        
    def infected_devices(self):
        """Returns a list of devices that are infected."""
        return [d for d in self.devices if d.infected]
    
    def turned_down_devices(self):
        """Returns a list of devices that are turned down."""
        return [d for d in self.devices if d.turned_down]
    
    def distance_to(self, target: "Enclave") -> int:
        """Returns the distance to the target enclave using BFS."""
        visited = set()
        queue = deque([(self, 0)])
        while queue:
            current, depth = queue.popleft()
            if current == target:
                return depth
            visited.add(current)
            for neighbor in current.neighbours:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        return None # No path found

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
#                                                NETWORK SEGMENTATION
# ========================================================================================================================

# class Network:
#     def __init__(self, devices: List[Device] = []):
#         """
#         :param devices: List of devices in the network
#         """
#         self.devices: List[Device] = devices

#     def add_device(self, device: Device):
#         self.devices.append(device)

#     def remove_device(self, device: Device):
#         if device in self.devices:
#             self.devices.remove(device)
#         else:
#             raise ValueError(f"Device {device.name} not found in network.")
    
#     def infected_devices(self):
#         """Returns a list of devices that are infected."""
#         return [d for d in self.devices if d.infected]
    
#     def turned_down_devices(self):
#         """Returns a list of devices that are turned down."""
#         return [d for d in self.devices if d.turned_down]
    
#     def __repr__(self):
#         return f"<Network with {len(self.devices)} device(s)>"


class Segmentation:
    def __init__(self, 
                 topology: List[List[Enclave]] = [], 
                 partition: List[List[Device]] = [], 
                 sensitivities: List[float] = [], 
                 vulnerability: List[float] = []):
        """
        :param topology: The topology of the network (list of enclaves and their neighbours)
        :param partition: The partition of the network (list of devices in each enclave)
        :param sensitivities: The sensitivities of the enclaves (how quickly they are
        """
        self.internet = Internet()
        self.enclaves: List[Enclave] = [self.internet]
        self.topology_matrix = []
        if topology:
            assert len(topology) == len(partition) == len(sensitivities) == len(vulnerability), "Solutions must have the same length."
            self.initialize(topology, partition, sensitivities, vulnerability)

    def initialize(self, topology: List[List[int]], partition: List[List[Device]], sensitivities: List[float], vulnerability: List[float]):
        """
        ### Algorithm 6 ###
        Initialisation algorithm - Simulation starting point

        :param topology: The topology of the network (list neighbours indices for each enclave)
        :param partition: The partition of the network (list of devices in each enclave)
        :param sensitivities: The sensitivities of the enclaves
        :param vulnerability: The vulnerabilities of the enclaves
        """
        num_enclaves = len(topology)
        self.topology_matrix = [[0] * num_enclaves for _ in range(num_enclaves)]

        # Create enclaves based on the partition
        for i in range(1, len(topology)):
            e = Enclave(
                id=i, 
                vulnerability=vulnerability[i], 
                sensitivity=sensitivities[i], 
                devices=partition[i]
                )
            # Add dynamic low value devices
            for j in range(random.randint(0, DYN_DEVICES)):
                e.add_device(Device(f"Low value device {j+1}", device_type="Low value device"))
            self.add_enclave(e)

        # Connect enclaves based on the topology
        for i, neighbours in enumerate(topology):
            self.enclaves[i].neighbours = neighbours
            for j in neighbours:
                self.topology_matrix[i][j] = 1
                self.topology_matrix[j][i] = 1

        self.update_distances()  # Update distances to the internet for all enclaves

        # print("Topology matrix:")
        # for row in self.topology_matrix:
        #     print(row)

        # print("Enclaves:")
        # for enclave in self.enclaves:
        #     print(enclave)

    def add_enclave(self, enclave: Enclave):
        self.enclaves.append(enclave)
    
    def connect_enclaves(self, e1: Enclave, e2: Enclave):
        e1.neighbours.append(e2.id)
        e2.neighbours.append(e1.id)
        self.update_distances()

    def remove_enclave(self, enclave: Enclave):
        if enclave in self.enclaves:
            self.enclaves.remove(enclave)
            for e in self.enclaves:
                if enclave.id in e.neighbours:
                    e.neighbours.remove(enclave)
            self.update_distances()
        else:
            raise ValueError(f"Enclave {enclave.id} not found in network.")
        
    def update_distances(self):
        """Updates the distances to the internet for all enclaves."""
        for e in self.enclaves:
            if e != self.internet:
                e.distance_to_internet = None
        # BFS to calculate distances from the internet
        visited = set()
        queue = deque([(self.internet, 0)])
        while queue:
            current, dist = queue.popleft()
            current.distance_to_internet = dist
            visited.add(current)
            for n in current.neighbours:
                neighbor = self.enclaves[n]
                if neighbor not in visited and neighbor.distance_to_internet is None:
                    queue.append((neighbor, dist + 1))

    def update_partition(self, partition: List[List[Device]]):
        """
        Updates the partition of the network.

        :param partition: The new partition of the network (list of devices in each enclave)
        """
        assert len(partition) == len(self.enclaves), "Partition must have the same length as enclaves."
        self.enclaves = [self.internet]  # Reset enclaves to only the internet
        for i in range(1, len(partition)):
            e = Enclave(
                id=i, 
                # TODO: list index out of range
                vulnerability=self.enclaves[i].vulnerability, 
                sensitivity=self.enclaves[i].sensitivity,
                devices=partition[i]
                )
            self.add_enclave(e)
        self.update_distances()

    def update_sensitivities(self, sensitivities: List[float]):
        """
        Updates the sensitivities of the enclaves.

        :param sensitivities: The new sensitivities of the enclaves
        """
        assert len(sensitivities) == len(self.enclaves), "Sensitivities must have the same length as enclaves."
        for i in range(len(sensitivities)):
            self.enclaves[i].sensitivity = sensitivities[i]

    def n_enclaves(self) -> int:
        """Returns the number of enclaves in the network."""
        return len(self.enclaves)
        
    def compromised_enclaves(self) -> List[Enclave]:
        """Returns a list of enclaves that are compromised."""
        return [e for e in self.enclaves if e.compromised]
    
    def compromised_devices(self) -> List[Device]:
        """Returns a list of devices that are compromised."""
        return [d for e in self.enclaves for d in e.devices if d.infected]
    
    def turned_down_devices(self) -> List[Device]:
        """Returns a list of devices that are turned down."""
        return [d for e in self.enclaves for d in e.devices if d.turned_down]
    
    def neighbours(self) -> List[List[int]]:
        """Returns the neighbours of the network."""
        return [e.neighbours for e in self.enclaves]
    
    def partition(self) -> List[List[Device]]:
        """Returns the partition of the network into enclaves."""
        return [e.devices for e in self.enclaves]

    def sensitivities(self) -> List[float]:
        """Returns the sensitivities of the enclaves."""
        return [e.sensitivity for e in self.enclaves]

    def __repr__(self):
        return f"<Network Segmentation with {len(self.enclaves)} enclave(s) and {len(self.network.devices)} device(s)>"

