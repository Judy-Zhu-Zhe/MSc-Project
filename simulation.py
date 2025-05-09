from typing import List
import random

from network import Device, Enclave, Network

class Simulation:
    def __init__(self, C_appetite: float, I_appetite: float, beta: int, r: float, p_device_error: float = 0.7, p_network_error: float = 0.7):
        """
        :param C_appetite: Attacker compromise appetite
        :param I_appetite: Attacker information appetite
        :param beta: Beta value for device selection
        :param r: Reconnaissance rate
        """
        self.C_appetite = C_appetite
        self.I_appetite = I_appetite
        self.beta = beta
        self.r = r
        self.p_device_error = p_device_error
        self.p_network_error = p_network_error

    # ========================================================================================================================
    #                                                      ATTACK
    # ========================================================================================================================

    def device_compromise(self, device: Device) -> bool:
        """
        ### Algorithm 4 ###
        Simulate the compromise of a device.

        :param device: The device to compromise
        :return: Tuple (turned_down, loss) where turned_down is True if the device was compromised and loss is the loss incurred.
        """
        loss = 0
        device.infected = True
        if not device.has_been_infected:
            loss += device.prior_information_value
            device.has_been_infected = True
        compromise = random.random()
        device.turned_down = False
        if compromise <= self.C_appetite:
            device.turned_down = True
            loss += device.compromise_value
        return device.turned_down, loss
    
    def enclave_spread(self, enclave: Enclave, time: int) -> float:
        """
        ### Algorithm 2 ###
        Simulate the spread of an attack in the enclave over time.

        :param time: The time to spend in the enclave.
        """
        loss = 0
        alert = 0
        enclave.compromised = True
        if not enclave.devices:
            return 0
        
        # Select a random device to infect
        target = random.choice(enclave.devices)
        is_turned_down, compromise_loss = self.device_compromise(target, self.C_appetite)
        loss += compromise_loss
        if is_turned_down:
            alert += 1
            if self.enclave_detection(enclave, alert):
                loss += self.cleansing_loss_with_investigation(enclave)
                self.enclave_cleansing(enclave)
                return loss
        
        recon_time = self.reconnaissance(enclave, time)
        for _ in range(recon_time, time):
            # TODO: Select k best devices to infect
            to_infect = self.select_k_best(5, enclave.devices)
            for device in to_infect:
                # Add alert if the infection is detected (and blocked) or if the device is turned down
                test_alert = False
                if self.device_IDS_detect():
                    test_alert = True
                else:
                    is_turned_down, compromise_loss = self.device_compromise(target, self.C_appetite)
                    loss += compromise_loss
                    if is_turned_down:
                        test_alert = True
                if test_alert:
                    alert += 1
                    if self.enclave_detection(enclave, alert):
                        loss += self.cleansing_loss_with_investigation(enclave)
                        self.enclave_cleansing(enclave)
                        return loss
        return loss
    
    def reconnaissance(self, enclave: Enclave, spreading_time: int) -> int:
        """
        ### Algorithm 3 ###
        Simulate reconnaissance in the enclave over a fraction of time.

        :param enclave: The enclave to perform reconnaissance in
        :param spreading_time: Enclave maximum spreading time
        :return: Time taken for reconnaissance
        """
        recon_time = self.r * spreading_time
        disvovered = enclave.infected_devices.copy()
        for _ in range(recon_time):
            not_discovered = [d for d in enclave.devices if d not in disvovered]
            if not_discovered:
                new_device = random.choice(not_discovered)
                disvovered.append(new_device)
                # TODO: Update attacker prior value of the new discovered device with the true device value
                new_device.prior_information_value = new_device.true_information_value
        return recon_time

    def select_k_best(self, k: int, devices: List[Device]) -> List[Device]:
        """
        Select the k best devices to infect based on their true information value and compromise value.
        
        :param k: Number of devices to select
        :param alpha: Weight for true information value
        :param beta: Weight for compromise value
        """
        return sorted(
            [d for d in devices if not d.infected],
            key=lambda d: (self.I_appetite * d.true_information_value +
                            self.C_appetite * d.compromise_value),
            reverse=True
        )[:k]
    
    def network_spread(self, network: Network, T: int, times: List[int]) -> float:
        """
        ### Algorithm 1 ###
        Simulate the spread of an attack in the network over time.

        :param T: The simulation time.
        :param times: List of times spend in each enclave
        """
        compromised_enclaves = [Enclave("Internet")]
        loss = 0
        spent_time = 0
        while spent_time <= T:
            for enclave in compromised_enclaves:
                for next in enclave.neighbours:
                    if not next.compromised:
                        infect = random.random()
                        detected = self.network_detection()
                        if infect <= next.vulnerability and not detected:
                            loss += self.enclave_spread(enclave, times[next])
                            if next.compromised:
                                compromised_enclaves.append(next)
                            spent_time += times[next]
                            if spent_time >= T:
                                return loss
            spent_time += 1
        return loss
    

    # ========================================================================================================================
    #                                                      DEFENCE
    # ========================================================================================================================
    
    def device_IDS_detect(self) -> bool:
        """
        ### Algorithm 8 ###
        Simulate the detection of a device by the IDS/IPS system.

        :return: True if intrusion is detected (and blocked), False otherwise.
        """
        return random.random() <= 1 - self.p_device_error
    
    def enclave_cleansing(enclave: Enclave) -> float:
        """Simulate the cleansing of the enclave.
        
        :return: Cleansing loss incurred by the enclave."""
        enclave.compromised = False
        enclave.infected_devices = []
        for device in enclave.devices:
            device.reset()
        print(f"[{enclave.name}] System update triggered. Threats removed.")
    
    def cleansing_loss_with_investigation(enclave: Enclave) -> int:
        """Trigger cleansing with investigation 
        (usually when a sudden device mission becomes unresponsive due to compromise).
        
        :return: Cleansing loss incurred by the enclave."""
        return sum(d.compromise_value for d in enclave.devices if d.infected)

    # def cleansing_loss(enclave: Enclave) -> float:
    #     """Trigger cleansing without investigation 
    #     (enclave cleansing normally triggered).
        
    #     :return: Cleansing loss incurred by the enclave."""
    #     return cleansing_loss_with_investigation(enclave) / 2

    def enclave_detection(self, enclave: Enclave, alert: int) -> bool:
        """
        ### Algorithm 9 ###
        Simulate the detection of too many suspicious events triggering cleaning in the enclave.

        :param alert: Number of alerts detected in the enclave
        :return: True if the enclave detects an intrusion, False otherwise.
        """
        return random.random() <= enclave.sensitibvity * alert / len(enclave.devices)
    
    def regular_update(self, network: Network, p_update: float) -> bool:
        """
        Simulate a regular update of the network.

        :param network: The network to update
        :return: True if the update is successful, False otherwise.
        """
        if random.random() <= 1 - p_update:
            for enclave in network.enclaves:
                self.enclave_cleansing(enclave) # Model scheduled updates so no loss
            return True
        return False

    def network_detection(self) -> bool:
        """
        Simulate the network-level IDS/IPS detection.

        :return: True if the network detects an intrusion, False otherwise.
        """
        return random.random() <= 1 - self.p_network_error
    

    
    