from typing import Dict, Tuple
import time
import json
from datetime import datetime
import os

from network import Segmentation
from optimization import map_elites, topology_neighbours, MapElitesConfig
from visualization import draw_grid_heatmap, draw_segmentation_topology
from parameters import MapElitesConfig, CONFIG_1, CONFIG_2, CONFIG_ADP, MY_CONFIG

def save_results(seg: Segmentation, grid: Dict[Tuple, Tuple[Segmentation, float]], batch: int, gen: int, config: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config}_batch{batch}_gen{gen}_{timestamp}"
    dir_path = f"segmentations/{config}"
    os.makedirs(dir_path, exist_ok=True) # Ensure directory exists

    # Save segmentation
    seg_path = f"{dir_path}/seg_{filename}.json"
    with open(seg_path, "w") as f:
        json.dump(seg.to_dict(), f, indent=2)
    print(f"Saved segmentation to {seg_path}")

    # Save grid
    serializable_grid = {
        str(key): {
            "fitness": fitness,
            "segmentation": segmentation.to_dict()
        }
        for key, (segmentation, fitness) in grid.items()
    }
    grid_path = f"{dir_path}/grid_{filename}.json"
    with open(grid_path, "w") as f:
        json.dump(serializable_grid, f, indent=2)
    print(f"Saved grid to {grid_path}")
    
    return filename

def load_segmentation(filename: str) -> Segmentation:
    with open(filename, "r") as f:
        data = json.load(f)
    return Segmentation.from_dict(data)

def load_grid(filename: str) -> Dict[Tuple, Tuple[Segmentation, float]]:
    from_dict = Segmentation.from_dict
    with open(filename, "r") as f:
        data = json.load(f)
    grid = {
        tuple(map(float, key.strip("()").split(","))): (from_dict(value["segmentation"]), value["fitness"])
        for key, value in data.items()
    }
    return grid

def run_segmentation(config: MapElitesConfig, infected_seg: Segmentation = None):
    # Generate the universe of graphs
    print("Generating universe of graphs...")
    n_enclaves = config.n_enclaves
    topology_list, neighbours_table, distances_table = topology_neighbours(config.universe, n_enclaves, K=5)

    # Run MAP-Elites for optimization
    start_time = time.time()
    (seg, fitness), grid = map_elites(topology_list, neighbours_table, distances_table, config, infected_seg)
    end_time = time.time()
    elapsed = end_time - start_time

    print("\nResults:")
    print(f"Execution time for MAP-Elite: {elapsed:.2f} seconds")
    print(f"Fitness: {fitness}")
    print(seg.topology.edges())
    for e in seg.enclaves:
        if e.id == 0:
            print(e)
            continue
        print(f"Enclave {e.id}")
        print(f"    Sensitivity: {e.sensitivity:.2f}")
        print(f"    Vulnerability: {e.vulnerability:.2f}")
        print(f"    Devices: {[d.name for d in e.devices]}")
    filename = save_results(seg, grid, config.batch_size, config.generations, config.name)
    # draw_segmentation_topology(seg, save_path=f"imgs/{config.name}/seg_{filename}.png")
    # draw_grid_heatmap(grid, n_enclaves=n_enclaves, dim_x=0, dim_y=1, descriptor_names=config.descriptors, save_path=f"imgs/{config.name}/grid_{filename}.png")
    draw_segmentation_topology(seg)
    draw_grid_heatmap(grid, n_enclaves=n_enclaves, dim_x=0, dim_y=1, descriptor_names=config.descriptors)

if __name__ == "__main__":
    run_segmentation(CONFIG_1)
    # run_segmentation(CONFIG_2)
    # run_segmentation(MY_CONFIG)

    # seg_name = "Adaptation/seg_Adaptation_batch400_gen30_20250604_025636"
    # seg = load_segmentation(f"segmentations/{seg_name}.json")
    # draw_segmentation_topology(seg, save_path=f"imgs/{seg_name}.png")

    # seg.enclaves[2].devices[0].infect()
    # # draw_segmentation_topology(seg)
    # run_segmentation(CONFIG_ADP, infected_seg=seg)

    # grid_name = "Adaptation/grid_Adaptation_batch400_gen30_20250604_025636" 
    # grid = load_grid(f"segmentations/{grid_name}.json")
    # # draw_grid_heatmap(grid, dim_x=0, dim_y=1, n_enclaves=5, descriptor_names=["std_devices", "std_web_facing"])
    # draw_grid_heatmap(grid, dim_x=0, dim_y=1, n_enclaves=5, descriptor_names=["nb_high_deg_nodes", "std_devices"], save_path=f"imgs/{grid_name}.png")
