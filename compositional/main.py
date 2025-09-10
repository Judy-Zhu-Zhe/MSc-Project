import time
import argparse
from typing import Optional

from mapElites import compositional_map_elites, fitness
from visualization import *
from config import ConfigManager, load_segmentation_node
from metrics_hierarchy import topology_distance
from kCut import run_kway_experiment

def compositional_experiment(args, cm: ConfigManager, seg=None, fitness_threshold=None):
    # Load segmentation for adaptation
    if args.seg:
        seg = load_segmentation_node(args.seg)
        print(f"Starting segmentation for adaptation: {args.seg} network...")
    else:
        print(f"Starting segmentation for optimization: {cm.config.name} network...")

    # Run MAP-Elites
    start_time = time.time()
    final_seg = compositional_map_elites(cm.config, infected_node=seg, fitness_threshold=fitness_threshold, verbose=args.verbose)
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\nResults:")
    print(f"Execution time: {elapsed:.2f} seconds")
    print(f"Number of devices: {final_seg.seg.num_devices()}")
    final_seg.print_details()

    # Save results
    if args.save:
        filename = cm.save_results(final_seg)
        # draw_grid_heatmap(grid, n_enclaves=cm.config.n_enclaves, dim_x=0, dim_y=1, descriptor_names=cm.config.descriptors, save_path=f"imgs/{cm.config.name}/grid_{filename}.png")
    
    draw_compositional_segmentation_circlify(final_seg)
    return final_seg

def seg_fitness(seg: SegmentationNode, reference_seg: Optional[SegmentationNode] = None):
    return fitness(cm.config.configs[0], seg.seg, reference_seg.seg if reference_seg else None, verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default="setups/device_profiles/device_profiles_enterprise.yaml", help="Path to device profiles YAML")
    parser.add_argument("--network", type=str, default="setups/networks/enterprise_medium.yaml", help="Path to network structure YAML")
    parser.add_argument("--config", type=str, default="setups/configs/config_medium.yaml", help="Path to algorithm configuration YAML")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--seg", type=str, help="Path to segmentation JSON for adaptation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output from simulation and MAP-Elites")
    args = parser.parse_args()
    
    cm = ConfigManager(args.devices, args.network, args.config, verbose=args.verbose)

    # compositional_experiment(args, cm)

    # seg = load_segmentation_node("segmentations/Compositional/structure/balanced.json")
    # seg.infect_devices_and_propagate(1, devices={"IOT device": 1, "Web server": 1})
    # # seg.print_details()
    # draw_compositional_segmentation_circlify(seg)
    # cm.save_results(seg, filename="infection/infected_mixed")

    # seg = load_segmentation_node("segmentations/Compositional/infection/infected_mixed.json")
    # draw_compositional_segmentation_circlify(seg)
    # seg_adj = compositional_experiment(args, cm, seg, fitness_threshold=-600.0)
    # draw_compositional_segmentation_circlify(seg_adj)

    # python main.py --network "setups/networks/enterprise_large.yaml" --config "setups/configs/config_large.yaml" --save
    # python main.py --devices "setups/device_profiles/device_profiles_ot.yaml" --network "setups/networks/ot_DuPont.yaml" --config "setups/configs/config_ot.yaml" --save

    # seg_path = "segmentations/Compositional/structure/zerotrust_fig.json"
    # seg = load_segmentation_node(seg_path)
    # # print(seg_path.split("/")[-1])
    # # fitness(cm.config.configs[0], seg.seg)
    # draw_compositional_segmentation_circlify(seg)

    # seg = run_kway_experiment(cm)
    # seg.print_details()
    # cm.save_results(seg)

    # seg = load_segmentation_node("segmentations/Compositional/structure/balanced.json")
    # seg1 = load_segmentation_node("segmentations/Compositional/infection/infected_mixed.json")
    # seg2 = load_segmentation_node("segmentations/Compositional/infection/adp_mixed.json")

    # print("--------------------------------")
    # seg_fitness(seg1, seg)
    # print("--------------------------------")
    # seg_fitness(seg2, seg)
    # print("--------------------------------")

    # print(f"Topology distance: {topology_distance(seg1, seg2):.3f}")


    seg = load_segmentation_node("segmentations/OT_DuPont/ot.json")
    seg.print_details()
    draw_compositional_segmentation_circlify(seg)
    seg_fitness(seg)