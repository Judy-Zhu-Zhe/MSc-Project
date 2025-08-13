import time
import argparse

from mapElites import compositional_map_elites
from visualization import *
from config import ConfigManager, load_segmentation_node

def compositional_experiment(args, cm: ConfigManager):
    # Load segmentation for adaptation
    if args.seg:
        seg = load_segmentation_node(args.seg)
        print(f"Starting segmentation for adaptation: {args.seg}...")
    else:
        seg = None
        print(f"Starting segmentation {cm.config.name}...")

    # Run MAP-Elites
    start_time = time.time()
    final_seg = compositional_map_elites(cm.config, infected_node=seg, verbose=args.verbose)
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\nResults:")
    print(f"Execution time: {elapsed:.2f} seconds")
    print(f"Number of devices: {final_seg.seg.num_devices()}")
    # final_seg.print_details()

    # Save results
    if args.save:
        filename = cm.save_results(final_seg)
        # draw_grid_heatmap(grid, n_enclaves=cm.config.n_enclaves, dim_x=0, dim_y=1, descriptor_names=cm.config.descriptors, save_path=f"imgs/{cm.config.name}/grid_{filename}.png")
    
    draw_compositional_segmentation(final_seg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default="setups/device_profiles/device_profiles_enterprise.yaml", help="Path to device profiles YAML")
    parser.add_argument("--network", type=str, default="setups/networks/enterprise_medium.yaml", help="Path to network structure YAML")
    parser.add_argument("--config", type=str, default="setups/configs/config_test.yaml", help="Path to algorithm configuration YAML")
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument("--seg", type=str, help="Path to segmentation JSON for adaptation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output from simulation and MAP-Elites")
    args = parser.parse_args()
    
    cm = ConfigManager(args.devices, args.network, args.config, verbose=args.verbose)
    compositional_experiment(args, cm)

    # seg = load_segmentation_node("segmentations/Compositional_Experiment/seg_Compositional_Experiment_medium_20250729_111155.json")
    # seg.infect_devices_and_propagate(1, devices={"IOT device": 1})
    # seg = load_segmentation_node("segmentations/Compositional_Experiment/seg_Compositional_Experiment_medium_20250729_125646.json")
    # seg.print_details()
    # draw_compositional_segmentation(seg)
    # com_seg = compositional_map_elites(cm.config, infected_node=seg)
    # com_seg.print_details()
    # draw_compositional_segmentation(com_seg)


