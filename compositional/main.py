import time
import argparse
from typing import Optional

from mapElites import compositional_map_elites
from visualization import draw_segmentation_topology
from config import ConfigManager

def compositional_experiment(args, cm: ConfigManager):
    # Load segmentation for adaptation
    seg = cm.load_segmentation_node(args.seg) if args.seg else None

    # Run MAP-Elites
    start_time = time.time()
    final_seg = compositional_map_elites(cm.config, infected_node=seg)
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\nResults:")
    print(f"Execution time: {elapsed:.2f} seconds")
    print(f"Number of devices: {final_seg.seg.num_devices()}")
    final_seg.print_details()

    # Save results
    if args.save:
        filename = cm.save_results(final_seg)
        # draw_segmentation_topology(final_seg.seg, save_path=f"compositional/imgs/{cm.config.name}/seg_{filename}.png")
        # draw_grid_heatmap(grid, n_enclaves=cm.config.n_enclaves, dim_x=0, dim_y=1, descriptor_names=cm.config.descriptors, save_path=f"imgs/{cm.config.name}/grid_{filename}.png")
    # else:
        # draw_segmentation_topology(final_seg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default="compositional/configs/device_profiles.yaml", help="Path to device profiles YAML")
    parser.add_argument("--network", type=str, default="compositional/network/medium.yaml", help="Path to network structure YAML")
    parser.add_argument("--config", type=str, default="compositional/configs/config_compositional.yaml", help="Path to algorithm configuration YAML")
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument("--seg", type=str, help="Path to segmentation JSON for adaptation")
    args = parser.parse_args()
    
    cm = ConfigManager(args.devices, args.network, args.config)
    compositional_experiment(args, cm)

    # seg = cm.load_segmentation_node("compositional/segmentations/Compositional_Experiment/seg_Compositional_Experiment_medium_20250715_190242.json")
    # seg.print_details()
