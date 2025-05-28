from graphillion import GraphSet
import time
import json
from datetime import datetime

from network import Segmentation
from optimization import map_elites, topology_neighbours, MapElitesConfig
from visualization import draw_segmentation_topology
from parameters import CONFIG_1, CONFIG_2, MY_CONFIG

def save_segmentation(seg: Segmentation, batch: int, gen: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"segmentations/mapelites_batch{batch}_gen{gen}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(seg.to_dict(), f, indent=2)

def load_segmentation(filename: str) -> Segmentation:
    with open(filename, "r") as f:
        data = json.load(f)
    return Segmentation.from_dict(data)

def run_segmentation(config: MapElitesConfig):
    n_enclaves = config.n_enclaves
    # Generate all possible topologies
    universe = [(i, j) for i in range(n_enclaves) for j in range(n_enclaves) if i < j]
    GraphSet.set_universe(universe)
    degree_constraints = {i: [1, 2] if i == 0 else range(1, n_enclaves) for i in range(n_enclaves)} # Maximum 2 ISPs
    graphs = GraphSet.graphs(vertex_groups=[range(n_enclaves)], degree_constraints=degree_constraints)

    # Run MAP-Elites for optimization
    start_time = time.time()
    topology_list, neighbours_table, distances_table = topology_neighbours(graphs, n_enclaves, K=5)
    seg, fitness = map_elites(topology_list, neighbours_table, distances_table, config)
    end_time = time.time()
    elapsed = end_time - start_time

    print("\nNumber of graphs (topologies):", len(graphs))
    print("\nResults:")
    print(f"Execution time for MAP-Elite: {elapsed:.2f} seconds")
    print(f"Fitness: {fitness}")
    print(seg.topology.edges())
    for e in seg.enclaves:
        if e.id == 0:
            print(e)
            continue
        print(f"Enclave {e.id}")
        print(f"  Sensitivity: {e.sensitivity:.2f}")
        print(f"  Vulnerability: {e.vulnerability:.2f}")
        print(f"  Devices: {[d.name for d in e.devices]}")
    save_segmentation(seg, config.batch_size, config.generations)
    draw_segmentation_topology(seg)

if __name__ == "__main__":
    run_segmentation(MY_CONFIG)
