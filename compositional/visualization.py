import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, Tuple
import plotly.graph_objects as go
import math

from network import SegmentationNode, Root
from config import load_segmentation_node

def draw_grid_heatmap(grid: dict, n_enclaves: int, dim_x: int = 0, dim_y: int = 1,
    descriptor_names: Optional[list] = None, save_path: Optional[str] = None):
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
    descriptor_y = descriptor_names[dim_y] if descriptor_names else f"Descriptor[{dim_y}]"

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


def min_enclosing_circle(points, min_radius=0.1):
    # Welzl's algorithm would be ideal, but for small n, brute force is fine
    # Returns (center_x, center_y, radius)
    if not points:
        return (0, 0, min_radius)
    if len(points) == 1:
        return (points[0][0], points[0][1], min_radius)
    # Try all pairs as diameter
    best = None
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            cx = (points[i][0] + points[j][0]) / 2
            cy = (points[i][1] + points[j][1]) / 2
            r = max(math.hypot(points[k][0] - cx, points[k][1] - cy) for k in range(len(points)))
            if best is None or r < best[2]:
                best = (cx, cy, r)
    # Try all triplets as circumcircle
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            for k in range(j+1, len(points)):
                A, B, C = points[i], points[j], points[k]
                # Calculate circumcenter
                a = B[0] - A[0]
                b = B[1] - A[1]
                c = C[0] - A[0]
                d = C[1] - A[1]
                e = a*(A[0]+B[0]) + b*(A[1]+B[1])
                f = c*(A[0]+C[0]) + d*(A[1]+C[1])
                g = 2*(a*(C[1]-B[1]) - b*(C[0]-B[0]))
                if g == 0:
                    continue
                cx = (d*e - b*f) / g
                cy = (a*f - c*e) / g
                r = max(math.hypot(points[m][0] - cx, points[m][1] - cy) for m in range(len(points)))
                if best is None or r < best[2]:
                    best = (cx, cy, r)
    if best is None:
        return (points[0][0], points[0][1], min_radius)
    return (best[0], best[1], max(best[2], min_radius))

def point_on_circle(cx, cy, r, tx, ty) -> Tuple[float, float]:
    """Return the point on the circle at (cx,cy) with radius r in the direction of (tx,ty)"""
    if abs(tx - cx) < 0.001 and abs(ty - cy) < 0.001:
        # If target is at center, return a point on the circle in positive x direction
        return (cx + r, cy)
    
    angle = math.atan2(ty - cy, tx - cx)
    return (cx + r * math.cos(angle), cy + r * math.sin(angle))

def get_infection_status(node: SegmentationNode) -> dict:
    """Get infection status information for a node."""
    infected_devices = node.seg.all_compromised_devices()
    total_devices = node.seg.num_devices()
    
    return {
        'infected_count': len(infected_devices),
        'total_count': total_devices,
        'infection_rate': len(infected_devices) / total_devices if total_devices > 0 else 0,
        'infected_devices': [d.name for d in infected_devices]
    }

def get_config_summary(node: SegmentationNode) -> str:
    """Get a summary of config information for display."""
    if not node.config:
        return "No config"
    
    summary = f"Config: {node.config.name}"
    if hasattr(node.config, 'level'):
        summary += f" (Level {node.config.level})"
    if hasattr(node.config, 'generations'):
        summary += f" - {node.config.generations} generations"
    if hasattr(node.config, 'n_simulations'):
        summary += f" - {node.config.n_simulations} simulations"
    
    return summary

def draw_compositional_segmentation(seg_node: SegmentationNode):
    """Draw compositional segmentation with Internet as an enclave circle."""
    
    fig = go.Figure()
    # Find max depth for scaling
    def find_max_level(node: SegmentationNode, lvl: int = 0) -> int:
        if not node.children:
            return lvl
        return max(find_max_level(child, lvl+1) for child in node.children.values())
    max_level = find_max_level(seg_node)
    
    # Root level: Create a single large circle for the outermost segmentation
    # Internet will be one of the child enclaves
    root_center = (0, 0)
    root_radius = 2.0  # Large radius for root level
    
    # Start the recursive drawing process
    _draw_node_recursive(seg_node, fig, max_level, root_center, root_radius)
    
    # Draw Internet enclave circle in the outermost layer
    internet_center = (-2, 0)  # Position Internet on the left
    internet_radius = 0.15
    draw_enclave_circle(
        fig,
        internet_center,
        internet_radius,
        fillcolor="rgba(128,128,128,0.3)",
        line_color="gray",
        line_width=2,
        hover_text="INTERNET",
        layer="below"
    )
    
    # Add Internet label
    fig.add_annotation(
        x=internet_center[0],
        y=internet_center[1],  # Position below the circle
        text="INTERNET",
        showarrow=False,
        font=dict(size=12, color="black"),
        xanchor="center",
        yanchor="top"
    )
    
    # Add edges between Internet and other enclaves based on topology
    topology_edges = seg_node.seg.topology.edges()
    print(f"Topology edges: {topology_edges}")
    edge_lines = []
    
    # Get positions of enclaves - handle both child enclaves and leaf node enclaves
    enclave_positions = {}
    enclave_radii = {}
    
    if len(seg_node.children) > 0:
        # Case 1: Outer enclave has child enclaves
        print("Outer enclave has child enclaves")
        n_children = len(seg_node.children)
        angle_step = 2 * math.pi / n_children
        for idx, (child_key, child) in enumerate(seg_node.children.items()):
            angle = idx * angle_step
            # Use same positioning as _draw_node_recursive
            child_center_x = root_center[0] + root_radius * 0.4 * math.cos(angle)
            child_center_y = root_center[1] + root_radius * 0.4 * math.sin(angle)
            child_center = (child_center_x, child_center_y)
            child_radius = root_radius * 0.25  # Same as in _draw_node_recursive
            enclave_positions[int(child_key)] = (child_center, child_radius)
            enclave_radii[int(child_key)] = child_radius
    else:
        # Case 2: Outer enclave is a leaf node - use actual enclaves from segmentation
        print("Outer enclave is a leaf node - using actual enclaves")
        n_enclaves = len(seg_node.seg.enclaves)
        # Skip Internet enclave (index 0) for positioning
        non_internet_enclaves = [e for e in seg_node.seg.enclaves if e.id != 0]
        n_non_internet = len(non_internet_enclaves)
        
        if n_non_internet > 0:
            angle_step = 2 * math.pi / n_non_internet
            enclave_radius = root_radius * 0.4  # Same as in _draw_node_recursive for leaf nodes
            
            for idx, enclave in enumerate(non_internet_enclaves):
                angle = idx * angle_step
                # Use same positioning as _draw_node_recursive for leaf nodes
                enclave_center_x = root_center[0] + enclave_radius * math.cos(angle)
                enclave_center_y = root_center[1] + enclave_radius * math.sin(angle)
                enclave_center = (enclave_center_x, enclave_center_y)
                enclave_radius_circle = enclave_radius * 0.3  # Same as in _draw_node_recursive
                enclave_positions[enclave.id] = (enclave_center, enclave_radius_circle)
                enclave_radii[enclave.id] = enclave_radius_circle
    
    print(f"Enclave positions: {enclave_positions}")
    
    # Draw all edges from topology
    for i, j in topology_edges:
        print(f"Processing edge: {i} -> {j}")
        
        if i == 0:  # Internet is connected to enclave j
            if j in enclave_positions:
                enclave_center, enclave_radius = enclave_positions[j]
                # Calculate points on circle peripheries
                p1 = point_on_circle(internet_center[0], internet_center[1], internet_radius, enclave_center[0], enclave_center[1])
                p2 = point_on_circle(enclave_center[0], enclave_center[1], enclave_radius, internet_center[0], internet_center[1])
                edge_lines.append([p1, p2])
                print(f"Added Internet -> {j} edge")
        elif j == 0:  # Enclave i is connected to Internet
            if i in enclave_positions:
                enclave_center, enclave_radius = enclave_positions[i]
                # Calculate points on circle peripheries
                p1 = point_on_circle(enclave_center[0], enclave_center[1], enclave_radius, internet_center[0], internet_center[1])
                p2 = point_on_circle(internet_center[0], internet_center[1], internet_radius, enclave_center[0], enclave_center[1])
                edge_lines.append([p1, p2])
                print(f"Added {i} -> Internet edge")
        else:  # Connection between two non-Internet enclaves
            if i in enclave_positions and j in enclave_positions:
                center_i, radius_i = enclave_positions[i]
                center_j, radius_j = enclave_positions[j]
                # Calculate points on circle peripheries
                p1 = point_on_circle(center_i[0], center_i[1], radius_i, center_j[0], center_j[1])
                p2 = point_on_circle(center_j[0], center_j[1], radius_j, center_i[0], center_i[1])
                edge_lines.append([p1, p2])
                print(f"Added {i} -> {j} edge")
    
    # Draw all edges
    if edge_lines:
        for edge_x, edge_y in edge_lines:
            fig.add_trace(go.Scatter(
                x=edge_x, 
                y=edge_y,
                mode="lines",
                line=dict(color="red", width=2),
                hoverinfo="none",
                showlegend=False
            ))
    
    # Add config summary to title if root has config
    title = "Compositional Segmentation"
    if seg_node.config:
        config_summary = get_config_summary(seg_node)
        title += f"<br><sub>{config_summary}</sub>"
    
    # Update layout for root level
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        width=1200, 
        height=800,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()
    
    return fig

def draw_enclave_circle(fig: go.Figure, center: Tuple[float, float], radius: float, 
                        fillcolor: str = "rgba(0,0,255,0.05)", line_color: str = "blue",
                        opacity: float = 0.2, layer: str = "below", line_width: float = 2,
                        hover_text: str = None, config_info: str = None):
    """Helper function to draw a circle for an enclave with optional hover text and config info"""
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=center[0] - radius * 2, 
        y0=center[1] - radius * 2, 
        x1=center[0] + radius * 2, 
        y1=center[1] + radius * 2,
        line_color=line_color,
        line_width=line_width,
        fillcolor=fillcolor,
        opacity=opacity,
        layer=layer
    )
    
    # Combine hover text with config info
    full_hover_text = hover_text or ""
    if config_info:
        full_hover_text += f"<br>{config_info}"
    
    # Add hover text by creating an invisible scatter trace over the circle
    if full_hover_text:
        fig.add_trace(go.Scatter(
            x=[center[0]], 
            y=[center[1]],
            mode="markers",
            marker=dict(size=radius*120, color="rgba(0,0,0,0)"),  # Invisible marker covering circle area
            hoverinfo="text",
            text=[full_hover_text],
            showlegend=False
        ))

# Device color map: device_type -> color, keeping close to trust level colors
DEVICE_COLOR_MAP = {
    # High trust devices (red-based)
    'Authentication server': 'darkred',
    'Domain controller': 'maroon',      # Dark red for highest trust level
    'Syslog server': 'crimson',
    'SQL database': 'firebrick',
    'Admin computer': 'indianred',
    
    # Medium trust devices (yellow-based)
    'E-mail server': 'goldenrod',
    'DNS server': 'darkgoldenrod',
    'Printer server': 'gold',
    'DHCP server': 'khaki',
    'Workstation': 'yellow',
    
    # Low trust devices (green-based)
    'Web server': 'darkgreen',
    'Printer': 'forestgreen',
    'Guest device': 'limegreen',
    'IOT device': 'mediumseagreen',
    
    # Default colors for unknown device types
    'low': 'green',
    'medium': 'yellow', 
    'high': 'red'
}

def _draw_node_recursive(node: SegmentationNode, fig: go.Figure, max_level: int, 
                        center: Tuple[float, float], radius: float) -> Tuple[float, float, float]:
    """
    Helper function to recursively draw nodes with proper coordinate translation.
    
    Returns:
        Tuple of (center_x, center_y, radius) of the enclosing circle for this node
    """
    child_positions = []
    device_positions = []
    child_radii = {}
    
    if node.seg.num_enclaves() == 0:
        # No enclaves, use default circle
        enclosing_center = center
        enclosing_radius = radius * 0.2
        
        # Draw this enclave's circle
        total_devices = node.seg.num_devices()
        hover_text = f"Enclave (level {node.level})<br>Devices: {total_devices}"
        
        # Add infection status
        infection_status = get_infection_status(node)
        if infection_status['infected_count'] > 0:
            hover_text += f"<br>⚠️ {infection_status['infected_count']} infected ({infection_status['infection_rate']:.1%})"
        
        # Add config info if available
        config_info = None
        if node.config:
            config_info = f"Config: {node.config.name}<br>Level: {node.config.level}<br>Generations: {node.config.generations}"
        
        draw_enclave_circle(fig, enclosing_center, enclosing_radius, fillcolor="rgba(0,255,0,0.05)", line_color="black", layer="below", hover_text=hover_text, config_info=config_info)

    elif not node.children:
        # Leaf node with devices - first draw enclaves, then distribute devices within their enclaves
        enclave_positions = []
        all_device_positions = []
        
        # First, draw enclaves and calculate their positions
        n_enclaves = len([e for e in node.seg.enclaves if e.id != 0])  # Exclude Internet
        if n_enclaves > 0:
            angle_step = 2 * math.pi / n_enclaves
            enclave_radius = radius * 0.4  # Enclave circles smaller than parent
            
            enclave_idx = 0
            for enclave in node.seg.enclaves:
                if enclave.id == 0:
                    continue  # skip Internet
                
                # Calculate position for this enclave
                angle = enclave_idx * angle_step
                enclave_center_x = center[0] + enclave_radius * math.cos(angle)
                enclave_center_y = center[1] + enclave_radius * math.sin(angle)
                enclave_center = (enclave_center_x, enclave_center_y)
                
                # Draw enclave circle
                enclave_hover_text = f"Enclave {enclave.id}<br>Devices: {len(enclave.devices)}"
                draw_enclave_circle(
                    fig, 
                    enclave_center, 
                    enclave_radius * 0.3,  # Smaller radius for enclave circles
                    fillcolor="rgba(0,255,0,0.1)", 
                    line_color="green",
                    line_width=1,
                    hover_text=enclave_hover_text,
                    layer="below"
                )
                
                enclave_positions.append((enclave_center, enclave_radius * 0.3))
                
                # Distribute devices within this enclave
                n_devices = len(enclave.devices)
                if n_devices > 0:
                    device_angle_step = 2 * math.pi / n_devices
                    device_radius = enclave_radius * 0.15  # Devices very close to enclave center
                    
                    for j, device in enumerate(enclave.devices):
                        d_angle = j * device_angle_step
                        dx = enclave_center_x + device_radius * math.cos(d_angle)
                        dy = enclave_center_y + device_radius * math.sin(d_angle)
                        device_positions.append((dx, dy, device))
                        all_device_positions.append((dx, dy))
                
                enclave_idx += 1
        
        # Calculate minimum enclosing circle for all enclaves and devices
        if enclave_positions:
            # Use enclave centers for enclosing circle calculation
            enclave_centers = [pos[0] for pos in enclave_positions]
            enclosing_center_x, enclosing_center_y, enclosing_radius = min_enclosing_circle(enclave_centers, min_radius=radius * 0.2)
            enclosing_center = (enclosing_center_x, enclosing_center_y)
        else:
            # No enclaves, use default circle
            enclosing_center = center
            enclosing_radius = radius * 0.2
        
        # Draw the parent enclave circle
        total_devices = node.seg.num_devices()
        hover_text = f"Enclave (level {node.level})<br>Devices: {total_devices}"
        
        # Add infection status
        infection_status = get_infection_status(node)
        if infection_status['infected_count'] > 0:
            hover_text += f"<br>⚠️ {infection_status['infected_count']} infected ({infection_status['infection_rate']:.1%})"
        
        # Add config info if available
        config_info = None
        if node.config:
            config_info = f"Config: {node.config.name}<br>Level: {node.config.level}<br>Generations: {node.config.generations}"
        
        draw_enclave_circle(fig, enclosing_center, enclosing_radius, fillcolor="rgba(0,255,0,0.05)", line_color="black", layer="below", hover_text=hover_text, config_info=config_info)
        
        # Draw devices as nodes with device-specific colors
        # Batch devices by color to reduce number of traces
        devices_by_color = {}
        for dx, dy, device in device_positions:
            if device.infected:
                color = 'red'
            else:
                color = DEVICE_COLOR_MAP.get(device.device_type, 'gray')
            if color not in devices_by_color:
                devices_by_color[color] = {'x': [], 'y': [], 'text': []}
            devices_by_color[color]['x'].append(dx)
            devices_by_color[color]['y'].append(dy)
            
            # Enhanced device hover text
            device_text = f"{device.name}<br>Type: {device.device_type}<br>Vulnerability: {device.vulnerability:.2f}"
            if device.infected:
                device_text += "<br>⚠️ INFECTED"
            if device.turned_down:
                device_text += "<br>🔴 TURNED DOWN"
            
            devices_by_color[color]['text'].append(device_text)
        
        # Create one trace per color
        for color, data in devices_by_color.items():
            fig.add_trace(go.Scatter(
                x=data['x'], 
                y=data['y'],
                mode="markers",
                marker=dict(size=8, color=color, line=dict(width=1, color='black')),
                hoverinfo="text",
                text=data['text'],
                showlegend=False
            ))
                
    else:
        # Distribute children inside the circle
        angle_step = 2 * math.pi / len(node.children)
        
        for idx, (child_key, child) in enumerate(node.children.items()):
            # Calculate position for this child inside the circle
            angle = idx * angle_step
            # Use a smaller radius (0.4) to keep children well inside the parent
            item_center_x = center[0] + radius * 0.4 * math.cos(angle)
            item_center_y = center[1] + radius * 0.4 * math.sin(angle)
            item_center = (item_center_x, item_center_y)
            item_radius = radius * 0.25  # Item circle much smaller than parent
            
            # Recurse into child
            child_cx, child_cy, child_r = _draw_node_recursive(
                child, 
                fig, 
                max_level, 
                item_center, 
                item_radius
            )
            child_positions.append((child_cx, child_cy))
            child_radii[child_key] = child_r
        
        # Compute minimum enclosing circle for all children
        if child_positions:
            enclosing_center_x, enclosing_center_y, enclosing_radius = min_enclosing_circle(child_positions, min_radius=radius * 0.2)
            enclosing_center = (enclosing_center_x, enclosing_center_y)
        else:
            # No children, use default circle
            enclosing_center = center
            enclosing_radius = radius * 0.2
        
        # Draw this enclave's circle
        total_devices = node.seg.num_devices()
        hover_text = f"Enclave (level {node.level})<br>Devices: {total_devices}"
        
        # Add infection status
        infection_status = get_infection_status(node)
        if infection_status['infected_count'] > 0:
            hover_text += f"<br>⚠️ {infection_status['infected_count']} infected ({infection_status['infection_rate']:.1%})"
        
        # Add config info if available
        config_info = None
        if node.config:
            config_info = f"Config: {node.config.name}<br>Level: {node.config.level}<br>Generations: {node.config.generations}"
        
        draw_enclave_circle(fig, enclosing_center, enclosing_radius, fillcolor="rgba(0,255,0,0.05)", line_color="black", layer="below", hover_text=hover_text, config_info=config_info)
        
        # Draw edges between child enclaves based on topology
        if len(child_positions) > 1:
            # Get topology edges once and cache them
            topology_edges = node.seg.topology.edges()
            edge_lines = []
            
            # Create mapping from enclave index to child position index
            # Use the same order as child_positions (enumeration order)
            child_keys_ordered = list(node.children.keys())  # Same order as enumeration
            # Both Root and SegmentationNode now use integer keys
            enclave_to_child_idx = {int(key): idx for idx, key in enumerate(child_keys_ordered)}
            
            for (i, j) in topology_edges:
                # Skip Internet connections as they're handled separately
                if i == 0 or j == 0:
                    continue
                
                # Check if both enclaves have children in this node
                if i not in enclave_to_child_idx or j not in enclave_to_child_idx:
                    continue
                
                child1_idx = enclave_to_child_idx[i]
                child2_idx = enclave_to_child_idx[j]
                
                child1_cx, child1_cy = child_positions[child1_idx]
                child2_cx, child2_cy = child_positions[child2_idx]
                # Both Root and SegmentationNode now use integer keys
                child1_r = child_radii[i]
                child2_r = child_radii[j]
                
                # Calculate points on circle peripheries
                p1 = point_on_circle(child1_cx, child1_cy, child1_r, child2_cx, child2_cy)
                p2 = point_on_circle(child2_cx, child2_cy, child2_r, child1_cx, child1_cy)
                
                # Only add edge if points are different
                if (abs(p1[0] - p2[0]) > 0.001 or abs(p1[1] - p2[1]) > 0.001):
                    edge_lines.append([p1, p2])
            
            # Batch all edges into one trace
            if edge_lines:
                edge_x = []
                edge_y = []
                for p1, p2 in edge_lines:
                    edge_x.extend([p1[0], p2[0], None])
                    edge_y.extend([p1[1], p2[1], None])
                
                fig.add_trace(go.Scatter(
                    x=edge_x, 
                    y=edge_y,
                    mode="lines",
                    line=dict(color="grey", width=1),
                    hoverinfo="none",
                    showlegend=False
                ))
    
    return enclosing_center[0], enclosing_center[1], enclosing_radius


import circlify

def draw_compositional_segmentation_circlify(seg_node: SegmentationNode):
    """
    Draw segmentation using circlify with preserved hierarchy and edge connections.
    """

    def build_hierarchy(node, path="root"):
        """Build hierarchy for circlify, including both enclaves and devices."""
        children = []
        
        if not node.children:
            # Leaf node - create enclave-level circles with devices as children
            enclave_data = []
            for enclave in node.seg.enclaves:
                if enclave.id == 0:  # Skip Internet
                    continue
                
                # Create device data for this enclave
                device_data = []
                for device in enclave.devices:
                    device_data.append({
                        'id': f"{path}/enclave_{enclave.id}/device_{device.name}",
                        'datum': 1  # Each device has size 1
                    })
                
                # If no devices, create a placeholder
                if not device_data:
                    device_data = [{'id': f"{path}/enclave_{enclave.id}/empty", 'datum': 1}]
                
                # Create enclave circle with devices as children
                enclave_data.append({
                    'id': f"{path}/enclave_{enclave.id}",
                    'datum': len(device_data),
                    'children': device_data
                })
            
            # If no enclaves, create a placeholder
            if not enclave_data:
                enclave_data = [{'id': f"{path}/empty", 'datum': 1}]
            
            return {
                'id': path,
                'datum': sum(enclave['datum'] for enclave in enclave_data),
                'children': enclave_data
            }
        else:
            # Non-leaf node - recurse into children
            for key, child in node.children.items():
                child_path = f"{path}/enclave_{key}"
                children.append(build_hierarchy(child, child_path))
            
            return {
                'id': path,
                'datum': sum(child['datum'] for child in children) if children else 1,
                'children': children
            }

    # Build device metadata mapping separately from circlify hierarchy
    device_metadata = {}
    def collect_device_metadata(node, path="root"):
        """Collect device metadata for later lookup."""
        if not node.children:
            # Leaf node - collect device metadata
            for enclave in node.seg.enclaves:
                if enclave.id == 0:  # Skip Internet
                    continue
                for device in enclave.devices:
                    device_path = f"{path}/enclave_{enclave.id}/device_{device.name}"
                    device_metadata[device_path] = {
                        'device': device,
                        'enclave_id': enclave.id
                    }
        else:
            # Non-leaf node - recurse into children
            for key, child in node.children.items():
                child_path = f"{path}/enclave_{key}"
                collect_device_metadata(child, child_path)
    
    collect_device_metadata(seg_node)
    
    hierarchy = [build_hierarchy(seg_node)]
    circles = circlify.circlify(
        hierarchy, show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    # Create mappings for easy lookup
    circle_map = {}
    device_map = {}
    # Map path to circle coordinates for each level
    path_to_circle = {}
    
    for c in circles:
        circle_map[c.ex['id']] = (c.x, c.y, c.r)
        path_to_circle[c.ex['id']] = (c.x, c.y, c.r)
        
        # Map devices to their circles using metadata
        if c.ex['id'] in device_metadata:
            device_map[device_metadata[c.ex['id']]['device'].name] = (c.x, c.y, c.r)

    fig = go.Figure()
    
    # Draw circles for enclaves and devices
    for path, (x, y, r) in circle_map.items():
        if 'device_' in path:
            # Device circle
            device_name = path.split('device_')[-1]
            device = None
            if path in device_metadata:
                device = device_metadata[path]['device']
            
            if device:
                color = 'red' if device.infected else DEVICE_COLOR_MAP.get(device.device_type, 'gray')
                device_text = f"{device.name}<br>Type: {device.device_type}<br>Vulnerability: {device.vulnerability:.2f}"
                if device.infected:
                    device_text += "<br>⚠️ INFECTED"
                if device.turned_down:
                    device_text += "<br>🔴 TURNED DOWN"
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers",
                    marker=dict(size=8, color=color, line=dict(width=1, color='black')),
                    hoverinfo="text",
                    text=[device_text],
                    showlegend=False
                ))
        elif 'enclave_' in path:
            # Enclave circle
            # Extract enclave ID from path like "root/enclave_1" or "root/enclave_1/device_device1"
            path_parts = path.split('/')
            enclave_part = None
            for part in path_parts:
                if part.startswith('enclave_'):
                    enclave_part = part
                    break
            if enclave_part:
                enclave_id = enclave_part.split('enclave_')[-1]
            else:
                enclave_id = "unknown"
            
            # Create hover text for enclave
            hover_text = f"Enclave {enclave_id}"
            
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x - r, y0=y - r,
                x1=x + r, y1=y + r,
                line_color="black",
                fillcolor="rgba(0,255,0,0.1)",
                line_width=1
            )
            
            # Add invisible scatter point for hover functionality
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=r*100, color="rgba(0,0,0,0)", line=dict(width=0)),
                hoverinfo="text",
                text=[hover_text],
                showlegend=False
            ))
        elif path == "root":
            # Skip drawing the root circle - we'll draw Internet separately
            pass

    # Draw edges at all levels of the hierarchy
    def draw_edges_for_node(node, current_path=""):
        """Recursively draw edges for each segmentation level."""
        if not node.children:
            # Leaf node - draw edges between enclaves at this level
            topology_edges = node.seg.topology.edges()
            edge_x, edge_y = [], []
            
            for i, j in topology_edges:
                # Skip Internet connections as they're handled separately
                if i == 0 or j == 0:
                    continue
                
                # Find the circles for these enclaves at this level
                # Look for enclave circles that are direct children of the current path
                enclave_i_path = f"{current_path}/enclave_{i}" if current_path else f"enclave_{i}"
                enclave_j_path = f"{current_path}/enclave_{j}" if current_path else f"enclave_{j}"
                
                # Check if these enclave paths exist in our circle map
                if enclave_i_path in path_to_circle and enclave_j_path in path_to_circle:
                    x1, y1, r1 = path_to_circle[enclave_i_path]
                    x2, y2, r2 = path_to_circle[enclave_j_path]
                    
                    # Calculate points on circle peripheries
                    p1 = point_on_circle(x1, y1, r1, x2, y2)
                    p2 = point_on_circle(x2, y2, r2, x1, y1)
                    
                    edge_x.extend([p1[0], p2[0], None])
                    edge_y.extend([p1[1], p2[1], None])
            
            if edge_x:
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode="lines",
                    line=dict(color="red", width=1),
                    hoverinfo="none",
                    showlegend=False
                ))
        else:
            # Non-leaf node - recurse into children
            for key, child in node.children.items():
                child_path = f"{current_path}/enclave_{key}" if current_path else f"enclave_{key}"
                draw_edges_for_node(child, child_path)

    # Start drawing edges from the root node
    draw_edges_for_node(seg_node)

    # Draw Internet circle separately (positioned outside the main visualization)
    internet_x, internet_y = 1.5, 0.0  # Position to the right of the main visualization
    internet_r = 0.15  # Smaller radius for Internet
    
    # Draw Internet circle
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=internet_x - internet_r, y0=internet_y - internet_r,
        x1=internet_x + internet_r, y1=internet_y + internet_r,
        line_color="orange",
        fillcolor="rgba(255,165,0,0.2)",
        line_width=2
    )
    
    # Add Internet label
    fig.add_trace(go.Scatter(
        x=[internet_x], y=[internet_y],
        mode="text",
        text=["Internet"],
        textfont=dict(size=12, color="orange"),
        hoverinfo="text",
        showlegend=False
    ))
    
    # Draw edges from Internet to enclaves that have Internet connections
    def draw_internet_edges(node, current_path=""):
        """Draw edges from Internet to enclaves that connect to it."""
        if not node.children:
            # Leaf node - check for Internet connections
            topology_edges = node.seg.topology.edges()
            edge_x, edge_y = [], []
            
            for i, j in topology_edges:
                # Check if one of the endpoints is Internet (enclave 0)
                if i == 0 or j == 0:
                    # Find the non-Internet enclave
                    enclave_id = j if i == 0 else i
                    
                    # Look for the enclave circle
                    enclave_path = f"{current_path}/enclave_{enclave_id}" if current_path else f"enclave_{enclave_id}"
                    
                    if enclave_path in path_to_circle:
                        x1, y1, r1 = path_to_circle[enclave_path]
                        
                        # Calculate point on enclave circle periphery towards Internet
                        p1 = point_on_circle(x1, y1, r1, internet_x, internet_y)
                        
                        edge_x.extend([internet_x, p1[0], None])
                        edge_y.extend([internet_y, p1[1], None])
            
            if edge_x:
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode="lines",
                    line=dict(color="orange", width=2, dash="dash"),
                    hoverinfo="none",
                    showlegend=False
                ))
        else:
            # Non-leaf node - recurse into children
            for key, child in node.children.items():
                child_path = f"{current_path}/enclave_{key}" if current_path else f"enclave_{key}"
                draw_internet_edges(child, child_path)
    
    # Draw Internet edges
    draw_internet_edges(seg_node)

    # Add config summary to title if root has config
    title = "Compositional Segmentation (circlify)"
    if seg_node.config:
        config_summary = get_config_summary(seg_node)
        title += f"<br><sub>{config_summary}</sub>"

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1200, height=800,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='white'
    )
    fig.show()
    
    return fig



seg = load_segmentation_node("segmentations\Compositional_Example\seg_Compositional_Example_medium_20250805_162047.json")
seg.print_details()
draw_compositional_segmentation(seg)
# draw_compositional_segmentation_circlify(seg)

