from graphillion import GraphSet
import time
import json
from datetime import datetime

from network import Segmentation
from optimization import map_elites, topology_neighbours
from visualization import plot_behavior_map, draw_segmentation_topology
from parameters import N_ENCLAVES, CONFIG_1

def save_segmentation(seg: Segmentation, batch: int, gen: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"segmentations/mapelites_batch{batch}_gen{gen}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(seg.to_dict(), f, indent=2)

def load_segmentation(filename: str) -> Segmentation:
    with open(filename, "r") as f:
        data = json.load(f)
    return Segmentation.from_dict(data)

def main():
    # Generate all possible topologies
    universe = [(i, j) for i in range(N_ENCLAVES) for j in range(N_ENCLAVES) if i < j]
    GraphSet.set_universe(universe)
    degree_constraints = {i: [1, 2] if i == 0 else range(1, N_ENCLAVES) for i in range(N_ENCLAVES)} # Maximum 2 ISPs
    graphs = GraphSet.graphs(vertex_groups=[range(N_ENCLAVES)], degree_constraints=degree_constraints)

    # Run MAP-Elites for optimization
    config = CONFIG_1
    start_time = time.time()
    topology_list, neighbours_table, distances_table = topology_neighbours(graphs, N_ENCLAVES, K=5)
    seg, fitness = map_elites(topology_list, neighbours_table, distances_table, config)
    end_time = time.time()
    elapsed = end_time - start_time

    # Visualize the results
    # print("Archive size:", len(archive))
    # print("Archive:", archive)
    # plot_behavior_map(archive)
    # for k in archive.values():
    #     draw_segmentation_topology(k[0])
    #     print(k[0].topology.adj_matrix)
    #     print("Fitness: ", k[1])

    print("\nNumber of graphs:", len(graphs))
    print("Graphs:", graphs)
    print("Results:")
    print(f"Execution time for MAP-Elite: {elapsed:.2f} seconds")
    print(f"Fitness: {fitness}")
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
    main()
