from typing import List, Optional
import random
import copy

from network import Device, Enclave, Segmentation
from config import MapElitesConfig

class Simulation:
    def __init__(self, seg: Segmentation, config: MapElitesConfig, is_adaptation: bool, verbose: bool = False) -> None:
        """
        :param seg: The initialized network segmentation
        :param config: The configuration for the simulation
        :param verbose: Whether to print verbose output
        """
        self.segmentation = copy.deepcopy(seg) # Copy to avoid modifying the original segmentation
        self.T = config.total_sim_time
        self.time_in_enclaves = config.time_in_enclaves
        self.spent_time = 0
        self.C_appetite = config.c_appetite
        self.I_appetite = config.i_appetite
        self.r_reconnaissance = config.r_reconnaissance
        self.p_update = config.p_update
        self.cleansing_loss = 0.0
        self.verbose = verbose
        if is_adaptation:
            self.simulate_adaption()
        else:
            self.simulate_network()

    def simulate_network(self):
        """Simulate the attack spread in the network (from the Internet)."""
        if self.verbose:
            print("\n==================================================================")
            print("================== Simulating network attack... ==================")
            print("==================================================================")
        return self.network_spread()
    
    def simulate_adaption(self):
        """Simulate the attack spread from internally infected devices (adaption)."""
        if self.verbose:
            print("\n==================================================================")
            print("================= Simulating adaption attack... ==================")
            print("==================================================================")
        return self.internet_spread()

    # ========================================================================================================================
    #                                                      ATTACK
    # ========================================================================================================================

    def device_compromise(self, device: Device) -> bool:
        """
        ### Algorithm 4 ###
        Simulate the compromise of a device.

        :param device: The device to compromise
        :return: Ture if the device was compromised.
        """
        if self.verbose:
            print(f"        [{device.name}] Device compromised.")
        device.infect()
        # If the attacker decides to turn down the device
        compromise = random.random()
        if compromise <= self.C_appetite:
            device.turned_down = True
        return device.turned_down
    
    def enclave_spread(self, enclave: Enclave, time: int, infected_device: Optional[Device] = None):
        """
        ### Algorithm 2 ###
        Simulate the spread of an attack in the enclave over time.

        :param time: The time to spend in the enclave.
        :param infected_device: The device that is already infected
        """
        if self.verbose:
            print(f"        [Enclave {enclave.id}] compromise spreading from {infected_device.name if infected_device else "Internet"}...")
        alert = 0
        enclave.compromised = True
        
        # Select the highest vulnerability device to infect
        if not infected_device:
            if not enclave.all_devices():
                return
            infected_device = sorted(enclave.all_devices(), key=lambda d: d.vulnerability, reverse=True)[0]
        is_turned_down = self.device_compromise(infected_device)
        if is_turned_down:
            alert += 1
            if self.cleansing_detection(enclave, alert):
                if self.verbose:
                    print("     -- Enclave cleansing triggered.")
                self.cleansing_loss += self.cleansing_loss_with_investigation(enclave)
                self.enclave_cleansing(enclave)
        
        recon_time = self.reconnaissance(enclave, time)
        for _ in range(recon_time, time):
            to_infect = self.select_best(enclave.all_devices())
            for device in to_infect:
                # Add alert if the infection is detected (and blocked) or if the device is turned down
                test_alert = False
                if self.device_sucessful_infect(device):
                    is_turned_down = self.device_compromise(device)
                    if is_turned_down:
                        test_alert = True
                else:
                    if self.verbose:
                        print(f"        [{device.name}] Infection detected and blocked.")
                    test_alert = True
                if test_alert:
                    alert += 1
                    if self.cleansing_detection(enclave, alert):
                        if self.verbose:
                            print("     [!] Enclave cleansing triggered.")
                        self.cleansing_loss += self.cleansing_loss_with_investigation(enclave)
                        self.enclave_cleansing(enclave)
                        return
    
    def reconnaissance(self, enclave: Enclave, time: int) -> int:
        """
        ### Algorithm 3 ###
        Simulate reconnaissance in the enclave over a fraction of time.

        :param enclave: The enclave to perform reconnaissance in
        :param time: The time to spend in the enclave
        :return: Time taken for reconnaissance
        """
        recon_time = int(self.r_reconnaissance * time)
        discovered = enclave.infected_devices()
        for _ in range(recon_time):
            not_discovered = [d for d in enclave.all_devices() if d not in discovered]
            if not_discovered:
                new_device = random.choice(not_discovered)
                discovered.append(new_device)
                new_device.prior_information_value = new_device.information_value
        return recon_time

    def select_best(self, devices: List[Device], k: int = 5) -> List[Device]:
        """
        Select the best 5 devices to infect based on their true information value and compromise value.
        
        :param k: Number of devices to select
        :param devices: List of devices to select from
        """
        sorted_devices = sorted(
            [d for d in devices if not d.infected],
            key=lambda d: (self.I_appetite * d.prior_information_value + # Gathered from reconnaissance
                            self.C_appetite * d.compromise_value),
            reverse=True
        )
        return sorted_devices[:k]
    
    def network_spread(self):
        """
        ### Algorithm 1 ###
        Simulate the spread of an attack in the network over time.
        """
        compromised_enclave_idx: List[int] = [e.id for e in self.segmentation.enclaves if e.compromised]
        
        while self.spent_time <= self.T:
            if self.verbose:
                print(f"Time: [{self.spent_time}/{self.T}]")
            # For each compromised enclave, infect its neighbours
            for e in compromised_enclave_idx[:]:  # Use a copy to avoid modification during iteration
                for n in self.segmentation.topology.enclave_neighbours(e):
                    next = self.segmentation.enclaves[n]
                    if not next.compromised:
                        infect = random.random()
                        if self.verbose:
                            print(f"    Infecting [Enclave {next.id}] (v={next.vulnerability():.2f}) with probability {infect:.2f}.")
                        
                        if infect <= next.vulnerability():
                            if self.verbose:
                                print(f"    -- [√] Infection successful.")
                            self.enclave_spread(next, self.time_in_enclaves)
                            if next.compromised and next.id not in compromised_enclave_idx:
                                compromised_enclave_idx.append(next.id)
                            self.spent_time += self.time_in_enclaves
                            if self.spent_time >= self.T:
                                if self.verbose:
                                    print(f"Time limit reached.")
                                return
                        else:
                            if self.verbose:
                                print(f"    -- [x] Infection detected and blocked.")
            
            # Attempt regular update at each timestep
            if self.regular_update():
                if self.verbose:
                    print("     [!] Regular update triggered and cleansing executed.")
            self.spent_time += 1
        
        if self.verbose:
            print(f"Time limit reached.")

    
    def internet_spread(self):
        """
        ### Algorithm 5 ###
        Simulate the attack spreading from internally infected devices.
        """
        for e in self.segmentation.enclaves:
            for d in e.all_devices():
                if d.infected:
                    self.enclave_spread(e, self.time_in_enclaves, d)
                    self.spent_time += self.time_in_enclaves
    

    # ========================================================================================================================
    #                                                      DEFENCE
    # ========================================================================================================================
    
    def device_sucessful_infect(self, device: Device) -> bool:
        """
        ### Algorithm 8 ###
        Simulate the detection of a device by the IDS/IPS system.

        :return: True if the device is successfully infected, False if infection is detected or blocked.
        """
        if self.verbose:
            print(f"        Attempting infection on [{device.name}] (vuln={device.vulnerability:.2f})")
        return random.random() <= device.vulnerability
    
    def enclave_cleansing(self, enclave: Enclave):
        """Simulate the cleansing of the enclave."""
        enclave.compromised = False
        for device in enclave.all_devices():
            device.reset()
        if self.verbose:
            print(f"        [Enclave {enclave.id}] regular cleansed. Threats removed.")
    
    def cleansing_loss_with_investigation(self, enclave: Enclave) -> int:
        """Trigger cleansing with investigation 
        (usually when a sudden device mission becomes unresponsive due to compromise).
        
        :return: Cleansing loss incurred by the enclave."""
        return sum(d.compromise_value for d in enclave.infected_devices())

    def cleansing_loss_without_investigation(self, enclave: Enclave) -> float:
        """Trigger cleansing without investigation 
        (enclave cleansing normally triggered).
        
        :return: Cleansing loss incurred by the enclave."""
        return sum(d.compromise_value for d in enclave.infected_devices()) / 2

    def cleansing_detection(self, enclave: Enclave, alert: int) -> bool:
        """
        ### Algorithm 9 ###
        Simulate the detection of too many suspicious events triggering cleaning in the enclave.

        :param alert: Number of alerts detected in the enclave
        :return: True if the enclave detects an intrusion, False otherwise.
        """
        num_devices = len(enclave.all_devices())
        if num_devices == 0:
            return False
        return random.random() <= enclave.sensitivity * alert / num_devices
    
    def regular_update(self) -> bool:
        """Simulate a regular update of the network."""
        if random.random() <= self.p_update:
            for enclave in self.segmentation.enclaves:
                self.cleansing_loss += self.cleansing_loss_without_investigation(enclave)
                self.enclave_cleansing(enclave) # Model scheduled updates
            return True
        return False

