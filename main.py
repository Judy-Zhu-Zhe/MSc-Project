import random
from typing import List

from network import Device, Enclave, Network

N_ENCLAVES = 5 # Number of enclaves in the network
N_DEVICES = 55 # Number of devices in the network (1 infected employee computer)
VULNERABILITIES = [0.2, 0.3, 0.5, 0.8]
SIMULATION_TIME = 81 # Simulation time in seconds
MAX_SPREADING_TIME = 30 # Enclave maximum spreading time

DYN_DEVICES = 5 # Number of dynamic low value devices in the network
P_UPDATE = 1/90  # Probability of successful update
P_NETWORK_ERROR = 0.7  # Probability of network error
P_DEVICE_ERROR = 0.7  # Probability of device error

R_RECONNAISSANCE = 0.5  # Reconnaissance rate

class Solution:
    def __init__(self, sensitivity: List[float], partition: List[Enclave], topology: List[List[Enclave]]):
        """
        :param sensitivity: The sensitivity of the enclaves in the network
        :param partition: The partition of the network into enclaves
        :param topology: The topology of the network (neighbours of enclaves)
        """
        self.sensitivity = sensitivity
        self.partition = partition
        self.topology = topology


def initializtion(sol: Solution = None) -> None:
    """
    ### Algorithm 6 ###
    Initialize the network with a given number of enclaves and a solution object.

    :param sol: Solution object containing the network partition and topology
    """
    network = Network() # Initialize empty enclave list
    for i in range(N_ENCLAVES):
        e = Enclave(f"Enclave {i+1}")
        e.vulnerability = VULNERABILITIES[i]
        if sol:
            e.sensitibvity = sol.sensitivity[i]
            e.devices = sol.partition[i]
            for j in range(random.randint(0, DYN_DEVICES)):
                e.add_device(Device(f"Low value device {j+1}", device_type="Low value device"))
            e.neighbours = sol.topology[i]
        network.enclaves.append(e)


def main():
    # Initialize the network with enclaves and devices
    network = Network()
    initializtion(N_ENCLAVES)

    # Create a solution object with sensitivity, partition, and topology
    solution = Solution(sensitivity=[0.1] * N_ENCLAVES, partition=[], topology=[])

    # Initialize the network with the solution object
    initializtion(N_ENCLAVES, solution)

    # Print the initialized network and solution
    print(network)
    print(solution)


if __name__ == "__main__":
    main()
