import time
import argparse

from network import Segmentation
from mapElites import map_elites, topology_neighbours
from visualization import draw_grid_heatmap, draw_segmentation_topology
from config import MapElitesConfig, load_config_from_yaml, update_config_for_adptation, generate_filename, save_results, load_segmentation, load_grid

def run_segmentation(config: MapElitesConfig, infected_seg: Segmentation = None, save: bool = False):
    """
    Run the segmentation optimization process using MAP-Elites.

    :param config: Configuration object containing parameters for the optimization.
    :param infected_seg: Optional Segmentation object to use for adaptation.
    :param save: Whether to save the results to files.
    """
    # Generate the universe of graphs
    print("Generating universe of graphs...")
    topology_list, neighbours_table, distances_table = topology_neighbours(config.universe, config.n_enclaves, K=5)

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
    
    if save:
        # Save the results to files
        filename = generate_filename(config)
        save_results(seg, grid, config.name, filename)
        draw_segmentation_topology(seg, save_path=f"imgs/{config.name}/seg_{filename}.png")
        draw_grid_heatmap(grid, n_enclaves=config.n_enclaves, dim_x=0, dim_y=1, descriptor_names=config.descriptors, save_path=f"imgs/{config.name}/grid_{filename}.png")
    else:
        # Visualize the results without saving
        draw_segmentation_topology(seg)
        draw_grid_heatmap(grid, n_enclaves=config.n_enclaves, dim_x=0, dim_y=1, descriptor_names=config.descriptors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scalability.yaml", help="Path to config YAML")
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument("--seg", type=str, help="Path to segmentation JSON for adaptation")
    args = parser.parse_args()

    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = MapElitesConfig()

    if args.seg:
        seg = load_segmentation(args.seg)
        config = update_config_for_adptation(config)
        seg.randomly_infect(devices={"Employee computer": 1}) # Change to intended device to infect
        draw_segmentation_topology(seg)
        run_segmentation(config, infected_seg=seg, save=args.save)
    else:
        run_segmentation(config, save=args.save)
