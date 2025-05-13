from typing import List

from network import Segmentation
from simulation import Simulation

def security_loss(simulations: List[Simulation]) -> float:
    """Calculate the average security loss of a network across a number of simulations.
    
    :param networks: List of networks across simulations.
    :return: Average security loss across all networks.
    """
    loss = 0
    for s in simulations:
        seg = s.segmentation
        # Sum of information value of all infected devices
        information_loss = sum([d.prior_information_value for d in seg.compromised_devices()]) 
        # Sum of compromise value of all turned down devices
        compromise_loss = sum([d.compromise_value for d in seg.turned_down_devices()])
        # The cleansing loss if any
        cleansing_loss = s.cleansing_loss()
        # Sum of all losses
        loss += information_loss + compromise_loss + cleansing_loss
    return loss / len(simulations) if simulations else 0

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
        loss += num * enclave.distance_to_internet
    return loss

def resilience_loss(segmentation: Segmentation) -> float:
    """Calculate the resilience loss of segmentation as the average loss incurred by removing one enclave.
    
    :param network: Network segmentation.
    :return: Average resilience loss across all enclaves.
    """
    loss = 0
    for enclave in segmentation.enclaves:
        # Total compromise loss in an enclave that becomes disconnected from the Internet
        offline_loss = sum([d.compromise_value for d in enclave.devices if "resilience_affected" in d.device_group])
        # Resilience loss incurred by removing the enclave from the segmentation
        resilience_loss = offline_loss + sum([d.compromise_value for d in enclave.devices])
        loss += resilience_loss
    return loss / (len(segmentation.enclaves) - 1) if len(segmentation.enclaves) > 1 else 0

