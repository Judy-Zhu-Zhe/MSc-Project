from typing import List
import random
import copy

from network import Device, Enclave, Segmentation

class Simulation:
    def __init__(self, seg: Segmentation, 
                 T: int,
                 times: List[int],
                 C_appetite: float, 
                 I_appetite: float, 
                 beta: int, 
                 r_reconnaissance: float, 
                 p_update: float = 1/90,
                 p_network_error: float = 0.7,
                 p_device_error: float = 0.7
                 ) -> None:
        """
        :param seg: The initialized network segmentation
        :param T: Simulation time
        :param times: List of times spend in each enclave
        :param C_appetite: Attacker compromise appetite
        :param I_appetite: Attacker information appetite
        :param beta: Beta value for device selection
        :param r: Reconnaissance rate
        :param p_device_error: Probability of device error
        :param p_network_error: Probability of network error
        :param k: Number of top k devices to select
        """
        self.segmentation = copy.deepcopy(seg) # Copy to avoid modifying the original segmentation
        self.T = T
        self.times = times
        self.spent_time = 0
        self.C_appetite = C_appetite
        self.I_appetite = I_appetite
        self.beta = beta
        self.r_reconnaissance = r_reconnaissance
        self.p_update = p_update
        self.p_network_error = p_network_error
        self.p_device_error = p_device_error
        self.cleansing_loss = self.simulate_network()

    def simulate_network(self) -> float:
        """
        Simulate the attack spread in the network (from the Internet).
        
        :return: Total loss incurred by the attack.
        """
        print("\n==================================================================")
        print("================== Simulating network attack... ==================")
        print("==================================================================")
        return self.network_spread()
    
    def simulate_adaption(self) -> float:
        """
        Simulate the attack spread from internally infected devices (adaption).
        
        :return: Total loss incurred by the attack.
        """
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
        :return: Tuple (turned_down, loss) where turned_down is True if the device was compromised and loss is the loss incurred.
        """
        print(f"        [{device.name}] Device compromised.")
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
    
    def enclave_spread(self, enclave: Enclave, time: int, infected_device: Device = None) -> float:
        """
        ### Algorithm 2 ###
        Simulate the spread of an attack in the enclave over time.

        :param time: The time to spend in the enclave.
        :param infected_device: The device that is already infected
        """
        print(f"        [Enclave {enclave.id}] compromise spreading from {infected_device.name if infected_device else 'Internet'}...")
        loss = 0
        alert = 0
        enclave.compromised = True
        if not enclave.devices:
            return 0
        
        # Select a random device to infect
        if not infected_device:
            infected_device = random.choice(enclave.devices)
        is_turned_down, compromise_loss = self.device_compromise(infected_device)
        loss += compromise_loss
        if is_turned_down:
            alert += 1
            if self.enclave_detection(enclave, alert):
                print("     -- Enclave cleansing triggered.")
                loss += self.cleansing_loss_with_investigation(enclave)
                self.enclave_cleansing(enclave)
                return loss
        
        recon_time = self.reconnaissance(enclave, time)
        for _ in range(recon_time, time):
            to_infect = self.select_k_best(self.beta, enclave.devices)
            for device in to_infect:
                # Add alert if the infection is detected (and blocked) or if the device is turned down
                test_alert = False
                if self.device_IDS_detect():
                    print(f"        [{device.name}] Device infection detected and blocked.")
                    test_alert = True
                else:
                    is_turned_down, compromise_loss = self.device_compromise(device)
                    loss += compromise_loss
                    if is_turned_down:
                        test_alert = True
                if test_alert:
                    alert += 1
                    if self.enclave_detection(enclave, alert):
                        print("     [!] Enclave cleansing triggered.")
                        loss += self.cleansing_loss_with_investigation(enclave)
                        self.enclave_cleansing(enclave)
                        return loss
        return loss
    
    def reconnaissance(self, enclave: Enclave, time: int) -> int:
        """
        ### Algorithm 3 ###
        Simulate reconnaissance in the enclave over a fraction of time.

        :param enclave: The enclave to perform reconnaissance in
        :param time: The time to spend in the enclave
        :return: Time taken for reconnaissance
        """
        recon_time = int(self.r_reconnaissance * time)
        disvovered = enclave.infected_devices()
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
    
    def network_spread(self) -> float:
        """
        ### Algorithm 1 ###
        Simulate the spread of an attack in the network over time.
        """
        compromised_enclaves: List[Enclave] = [e for e in self.segmentation.enclaves if e.compromised] # Only Internet when initialized
        loss = 0
        while self.spent_time <= self.T:
            print(f"Time: [{self.spent_time}/{self.T}]")
            # For each compromised enclave, infect its neighbours
            for enclave in compromised_enclaves:
                for n in enclave.neighbours:
                    next = self.segmentation.enclaves[n]
                    if not next.compromised:
                        self.spent_time += 1 # TODO: Added
                        infect = random.random()
                        print(f"    Infecting [Enclave {next.id}] (v={next.vulnerability:.2f}) with probability {infect:.2f}.")
                        detected = self.network_detection()
                        if infect <= next.vulnerability:
                            if not detected:
                                print(f"    -- [√] Infection successful.")
                                loss += self.enclave_spread(next, self.times[n])
                                if next.compromised:
                                    compromised_enclaves.append(next)
                                self.spent_time += self.times[n]
                                if self.spent_time >= self.T:
                                    print(f"Time limit reached. Loss incurred: {loss:.2f}")
                                    return loss
                            else:
                                print(f"    -- [x] Infection detected and blocked.")
                        else:
                            print(f"    -- [x] Infection failed.")
            self.spent_time += 1
        print(f"Time limit reached. Loss incurred: {loss:.2f}")
        return loss
    
    def internet_spread(self) -> float:
        """
        ### Algorithm 5 ###
        Simulate the attack spreading from internally infected devices.
        """
        internal_loss = 0
        for e in self.segmentation.enclaves:
            if e.compromised:
                for d in e.devices:
                    if d.infected:
                        internal_loss += self.enclave_spread(e, self.times[e], d)
                        self.spent_time += self.times[e]
        return internal_loss
    

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
    
    def enclave_cleansing(self, enclave: Enclave) -> float:
        """Simulate the cleansing of the enclave."""
        enclave.compromised = False
        for device in enclave.devices:
            device.reset()
        print(f"        [Enclave {enclave.id}] cleansed. Threats removed.")
    
    def cleansing_loss_with_investigation(self, enclave: Enclave) -> int:
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
        return random.random() <= enclave.sensitivity * alert / len(enclave.devices)
    
    def regular_update(self, seg: Segmentation) -> bool:
        """
        Simulate a regular update of the network.

        :param network: The network to update
        :return: True if the update is successful, False otherwise.
        """
        if random.random() <= 1 - self.p_update:
            for enclave in seg.enclaves:
                self.enclave_cleansing(enclave) # Model scheduled updates so no loss
            return True
        return False

    def network_detection(self) -> bool:
        """
        Simulate the network-level IDS/IPS detection.

        :return: True if the network detects an intrusion, False otherwise.
        """
        return random.random() <= 1 - self.p_network_error
    

    
    