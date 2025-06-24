import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from collections import Counter

from network import Segmentation

def draw_grid_heatmap(
    grid: dict,
    n_enclaves: int,
    dim_x: int = 0,
    dim_y: int = 1,
    descriptor_names: list = None,
    save_path: str = None
):
    # Extract values
    points = [
        (key[dim_x], key[dim_y], -fitness)
        for key, (_, fitness) in grid.items()
    ]
    

    # Unique sorted bin values (used as axes)
    x_bins = sorted(set(p[0] for p in points))
    y_bins = sorted(set(p[1] for p in points))

    x_idx = {val: i for i, val in enumerate(x_bins)}
    y_idx = {val: i for i, val in enumerate(y_bins)}

    # Create heatmap matrix
    heatmap = np.full((len(y_bins), len(x_bins)), np.nan)
    for x, y, loss in points:
        heatmap[y_idx[y], x_idx[x]] = loss

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heatmap,
        origin='lower',
        cmap='viridis_r',
        aspect='auto'
    )

    # Set integer ticks if descriptor is "nb_..." or "distance_..."
    descriptor_x = descriptor_names[dim_x] if descriptor_names else f"Descriptor[{dim_x}]"
    descriptor_y = descriptor_names[dim_y] if descriptor_names else f"Descriptor[{dim_x}]"

    if "nb" in descriptor_x or "distance" in descriptor_x:
        xticks = list(range(n_enclaves))
        xticklabels = [str(x) for x in xticks]
        x_max = n_enclaves
    else:
        xticks = list(range(len(x_bins)))
        xticklabels = [f"{v:.2f}" for v in x_bins]
        x_max = len(x_bins)

    if "nb" in descriptor_y or "distance" in descriptor_y:
        yticks = list(range(n_enclaves))
        yticklabels = [str(y) for y in yticks]
        y_max = n_enclaves
    else:
        yticks = list(range(len(y_bins)))
        yticklabels = [f"{v:.2f}" for v in y_bins]
        y_max = len(y_bins)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Labels and formatting
    ax.set_xlabel(f"bins: {descriptor_x}")
    ax.set_ylabel(f"bins: {descriptor_y}")
    ax.set_title("MAP-Elites Fitness Heatmap")
    plt.colorbar(im, ax=ax, label="Loss")

    # Minor ticks and grid lines
    ax.set_xticks(np.arange(-0.5, x_max, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, y_max, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved heatmap to {save_path}")

    plt.show()


def draw_segmentation_topology(seg: Segmentation, save_path: str = None):
    G = nx.Graph()
    n = seg.topology.n_enclaves

    for i in range(n):
        enclave = seg.enclaves[i]
        size = len(enclave.devices)
        if i == 0:
            label = "Internet"
        else:
            device_counts = Counter(d.device_type for d in enclave.devices)
            device_lines = "\n".join(f"{dtype}: {count}" for dtype, count in device_counts.items())
            label = (
                f"Enclave {i}\n"
                f"s = {enclave.sensitivity:.2f}, v = {enclave.vulnerability:.2f}\n"
                f"Device num = {size}\n"
                f"{{{device_lines}}}"
            )
        color = "orange" if i == 0 else "lightblue"
        G.add_node(i, label=label, size=size, color=color)

    for i in range(n):
        for j in range(i + 1, n):
            if seg.topology.adj_matrix[i][j]:
                G.add_edge(i, j)

    pos = nx.spring_layout(G, seed=42)
    sizes = [G.nodes[i]["size"] * 1200 + 500 for i in G.nodes]
    sizes[0] = 2000
    colors = [G.nodes[i]["color"] for i in G.nodes]
    labels = {i: G.nodes[i]["label"] for i in G.nodes}

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    nx.draw(G, pos, node_size=sizes, node_color=colors, with_labels=False, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title("Network Segmentation Topology")
    ax.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f"Saved network diagram to {save_path}")

    plt.show()