import matplotlib.pyplot as plt
import networkx as nx

from network import Segmentation

def plot_behavior_map(grid):
    xs, ys, scores = [], [], []

    for key, (_, fitness) in grid.items():
        x, y = key[0], key[1]  # or any other pair of dimensions
        xs.append(x)
        ys.append(y)
        scores.append(fitness)

    plt.scatter(xs, ys, c=scores, cmap="viridis")
    plt.xlabel("High Degree Nodes")
    plt.ylabel("Std. Enclave Size")
    plt.colorbar(label="Fitness")
    plt.title("MAP-Elites Behavior Space")
    plt.show()


def draw_segmentation_topology(seg: Segmentation):
    G = nx.Graph()
    n = seg.n_enclaves()

    # Add nodes with attributes
    for i in range(n):
        enclave = seg.enclaves[i]
        size = len(enclave.devices)
        if i == 0:
            label = "Internet"
        else:
            label = (
                f"Enclave {i}\n"
                f"s = {enclave.sensitivity:.2f}, v = {enclave.vulnerability:.2f}\n"
                f"Device num = {size}"
            )
        color = "orange" if i == 0 else "lightblue"
        G.add_node(i, label=label, size=size, color=color)

    # Add edges from topology matrix
    for i in range(n):
        for j in range(i + 1, n):
            if seg.topology.adj_matrix[i][j]:
                G.add_edge(i, j)

    pos = nx.spring_layout(G, seed=42)  # fixed layout for consistency
    sizes = [G.nodes[i]["size"] * 200 + 100 for i in G.nodes]  # avoid zero-size
    sizes[0] = 1500  # Internet node size
    colors = [G.nodes[i]["color"] for i in G.nodes]
    labels = {i: G.nodes[i]["label"] for i in G.nodes}

    nx.draw(G, pos, node_size=sizes, node_color=colors, with_labels=False, edge_color="gray")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title("Network Topology with Sensitivity and Device Count")
    plt.axis("off")
    # plt.tight_layout()
    plt.show()
