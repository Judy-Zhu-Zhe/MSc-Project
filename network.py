from typing import List

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
    def __init__(self, name: str):
        """
        :param name: Identifier for the enclave
        :param vulnerability: Vulnerability of the enclave to compromise
        :param sensitivity: How quickly enclave cleansing is triggered in the event of a compromise
        :param devices: List of devices within the enclave
        """
        self.name = name
        self.vulnerability = 0
        self.sensitibvity = 0
        self.devices: List[Device] = []
        self.infected_devices: List[Device] = []
        self.neighbours: List[Enclave] = []
        self.compromised = False

    def add_device(self, device: Device):
        self.devices.append(device)

    def remove_device(self, device: Device):
        if device in self.devices:
            self.devices.remove(device)
        else:
            raise ValueError(f"Device {device.name} not found in enclave {self.name}.")

    def __repr__(self):
        return f"<Enclave {self.name}: {len(self.devices)} devices, Compromised={self.compromised}>"


# ========================================================================================================================
#                                                      NETWORK
# ========================================================================================================================

class Network:
    def __init__(self):
        """
        :param enclaves: List of enclaves in the network
        """
        self.enclaves: List[Enclave] = []

    def add_enclave(self, enclave: Enclave):
        self.enclaves.append(enclave)

    def connect_enclaves(self, e1: Enclave, e2: Enclave):
        e1.neighbours.append(e2)
        e2.neighbours.append(e1)

    def __repr__(self):
        return f"<Network with {len(self.enclaves)} enclave(s)>"

