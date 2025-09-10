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


def load_and_plot_segmentation(seg_path: str, cm: ConfigManager = None):
    """
    Load a segmentation from a JSON file and plot it using circlify visualization.
    
    :param seg_path: Path to the segmentation JSON file
    :param cm: ConfigManager instance (optional, for saving results)
    :return: Loaded SegmentationNode
    """
    print(f"Loading segmentation from: {seg_path}")
    seg = load_segmentation_node(seg_path)
    print(f"Loaded segmentation: {seg_path.split('/')[-1]}")
    seg.print_details()
    draw_compositional_segmentation_circlify(seg)
    return seg


def load_infect_and_plot(seg_path: str, device_infections: dict, save_filename: str = None, cm: ConfigManager = None):
    """
    Load a segmentation, infect specified devices, propagate infection, plot the result, and save it.
    
    :param seg_path: Path to the segmentation JSON file
    :param device_infections: Dictionary mapping device types to number of devices to infect
    :param save_filename: Optional filename to save the infected segmentation
    :param cm: ConfigManager instance (required for saving)
    :return: Infected SegmentationNode
    """
    print(f"Loading segmentation from: {seg_path}")
    seg = load_segmentation_node(seg_path)
    
    print(f"Infecting devices: {device_infections}")
    seg.infect_devices_and_propagate(1, devices=device_infections)
    
    print("Plotting infected segmentation...")
    draw_compositional_segmentation_circlify(seg)
    
    if save_filename and cm:
        print(f"Saving infected segmentation as: {save_filename}")
        cm.save_results(seg, filename=save_filename)
    
    return seg


def compare_fitness(seg1_path: str, seg2_path: str, reference_seg_path: str = None, cm: ConfigManager = None):
    """
    Compare the fitness of two segmentations, optionally against a reference segmentation.
    
    :param seg1_path: Path to first segmentation JSON file
    :param seg2_path: Path to second segmentation JSON file
    :param reference_seg_path: Optional path to reference segmentation for comparison
    :param cm: ConfigManager instance (required for fitness calculation)
    :return: Tuple of (fitness1, fitness2, topology_distance)
    """
    if not cm:
        raise ValueError("ConfigManager is required for fitness comparison")
    
    print("Loading segmentations for comparison...")
    seg1 = load_segmentation_node(seg1_path)
    seg2 = load_segmentation_node(seg2_path)
    
    reference_seg = None
    if reference_seg_path:
        reference_seg = load_segmentation_node(reference_seg_path)
        print(f"Using reference segmentation: {reference_seg_path}")
    
    print("=" * 50)
    print(f"Fitness of {seg1_path.split('/')[-1]}:")
    fitness1 = seg_fitness(seg1, reference_seg)
    
    print("=" * 50)
    print(f"Fitness of {seg2_path.split('/')[-1]}:")
    fitness2 = seg_fitness(seg2, reference_seg)
    
    print("=" * 50)
    topology_dist = topology_distance(seg1, seg2)
    print(f"Topology distance between segmentations: {topology_dist:.3f}")
    
    return fitness1, fitness2, topology_dist


def run_kway_experiment_wrapper(cm: ConfigManager, n_enclaves: int = 5, save_results: bool = True):
    """
    Run the k-way minimum cut experiment and optionally save results.
    
    :param cm: ConfigManager instance
    :param n_enclaves: Number of enclaves for the k-way cut
    :param save_results: Whether to save the results
    :return: Generated SegmentationNode from k-way experiment
    """
    print("Running k-way minimum cut experiment...")
    seg = run_kway_experiment(cm, n_enclaves)
    
    print("K-way experiment results:")
    seg.print_details()
    
    if save_results:
        print("Saving k-way experiment results...")
        cm.save_results(seg)
    
    return seg


def run_adaptation_experiment(original_seg_path: str, fitness_threshold: float = -600.0, 
                            cm: ConfigManager = None, save_results: bool = True):
    """
    Run an adaptation experiment starting from an infected segmentation.
    
    :param original_seg_path: Path to the original (infected) segmentation
    :param fitness_threshold: Fitness threshold for the adaptation experiment
    :param cm: ConfigManager instance
    :param save_results: Whether to save the adapted results
    :return: Adapted SegmentationNode
    """
    if not cm:
        raise ValueError("ConfigManager is required for adaptation experiment")
    
    print(f"Loading original segmentation: {original_seg_path}")
    original_seg = load_segmentation_node(original_seg_path)
    
    print("Plotting original segmentation...")
    draw_compositional_segmentation_circlify(original_seg)
    
    print(f"Running adaptation experiment with fitness threshold: {fitness_threshold}")
    adapted_seg = compositional_experiment(None, cm, original_seg, fitness_threshold)
    
    print("Plotting adapted segmentation...")
    draw_compositional_segmentation_circlify(adapted_seg)
    
    return adapted_seg

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