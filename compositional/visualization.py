from typing import Tuple
import plotly.graph_objects as go
import circlify
import math

from network import SegmentationNode

# Device color map: device_type -> color, keeping close to trust level colors
DEVICE_COLOR_MAP = {
    # High trust devices (purple-based to distinguish from infected red)
    'Authentication server': 'purple',
    'Domain controller': 'darkviolet', 
    'Syslog server': 'mediumpurple',
    'SQL database': 'blueviolet',
    'Admin computer': 'mediumorchid',
    
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
    
    # =========================
    # OT DEVICES BY ZONE
    # =========================
    
    # Business Zone (L4) - Blue-based colors
    'DUPONTNET Domain Controller': 'darkblue',
    'DUPONTNET Resource Domain Controllers': 'blue',
    'Manufacturing Message Bus Adaptors (SAP, EConnect)': 'steelblue',
    'Corporate Patch Management Server': 'royalblue',
    'Manufacturing Application Server (L4)': 'mediumblue',
    'Process Explorer Clients (L4)': 'lightblue',
    'PE Clients (L4)': 'lightblue',
    'Web.21 Server': 'skyblue',
    
    # Operations Management Zone (L3) - Green-based colors
    'IP 21 Server PM&C': 'darkgreen',
    'Manufacturing Application Server (L3)': 'forestgreen',
    'PE Clients (L3)': 'limegreen',
    
    # Process Control Zone (L2/L1) - Yellow-based colors
    'DCS AD Domain Controllers': 'gold',
    'DCS Application Server': 'goldenrod',
    'DCS Consoles': 'darkgoldenrod',
    'Process Explorer Clients (L2)': 'yellow',
    'DCS Controllers': 'orange',
    'Field Bus Gateway': 'darkorange',
    'Field Devices (PCZ)': 'khaki',
    
    # Safety System Zone - Purple-based colors
    'SIS (Safety Instrumented System)': 'purple',
    'Field Devices (Safety)': 'darkviolet',
    
    # Default colors for unknown device types
    'low': 'green',
    'medium': 'yellow', 
    'high': 'purple'
}

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
    if hasattr(node.config, 'generations'):
        summary += f" - {node.config.generations} generations"
    if hasattr(node.config, 'evaluation_metrics'):
        summary += f"\n - {node.config.evaluation_metrics}"
    
    return summary


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
    
    # Apply spacing adjustments to circles based on hierarchy level
    def get_hierarchy_level(path):
        """Determine the hierarchy level from the path."""
        if path == "root":
            return 0
        
        # Split the path and analyze the structure
        parts = path.split('/')
        
        # A path like "root/enclave_1/enclave_2/device_device1" should be level 2, not level 3
        enclave_count = 0
        for part in parts:
            if part.startswith('enclave_') and 'device_' not in part:
                enclave_count += 1
        
        return enclave_count
    
    def get_parent_path(path):
        """Get the parent path of a given path."""
        if path == "root":
            return None
        parts = path.split('/')
        if len(parts) <= 1:
            return None
        return '/'.join(parts[:-1])
    
    # First pass: calculate all adjusted radii
    radius_map = {}
    for c in circles:
        path = c.ex['id']
        level = get_hierarchy_level(path)
        
        # Calculate spacing based on hierarchy level
        base_spacing_factor = 0.05  # Base spacing between circles at same level
        level_spacing_factor = 0.1  # Additional spacing per hierarchy level
        
        # Apply radius reduction based on level
        original_radius = c.r
        total_reduction = base_spacing_factor + (level * level_spacing_factor)
        adjusted_radius = original_radius * (1.0 - total_reduction)
        radius_map[path] = adjusted_radius
    
    # Second pass: adjust positions to ensure proper containment
    for c in circles:
        path = c.ex['id']
        parent_path = get_parent_path(path)
        
        if parent_path and parent_path in radius_map:
            # This is a child circle - adjust position to stay within parent
            parent_x, parent_y, parent_r = circle_map.get(parent_path, (c.x, c.y, radius_map[parent_path]))
            child_radius = radius_map[path]
            
            # Calculate the maximum distance the child can be from parent center
            max_distance = parent_r - child_radius - 0.02  # Small buffer
            
            # Calculate current distance from parent center
            current_distance = math.sqrt((c.x - parent_x)**2 + (c.y - parent_y)**2)
            
            if current_distance > max_distance:
                # Child is outside parent boundary - move it inside
                if current_distance > 0:
                    # Normalize the direction vector
                    direction_x = (c.x - parent_x) / current_distance
                    direction_y = (c.y - parent_y) / current_distance
                    
                    # Position child at maximum allowed distance from parent center
                    adjusted_x = parent_x + direction_x * max_distance
                    adjusted_y = parent_y + direction_y * max_distance
                else:
                    # Child is at parent center - move it slightly
                    adjusted_x = parent_x + 0.01
                    adjusted_y = parent_y + 0.01
            else:
                # Child is already within parent boundary
                adjusted_x = c.x
                adjusted_y = c.y
        else:
            # Root or top-level circle - keep original position
            adjusted_x = c.x
            adjusted_y = c.y
        
        # Store the adjusted circle coordinates and radius
        adjusted_radius = radius_map[path]
        circle_map[path] = (adjusted_x, adjusted_y, adjusted_radius)
        path_to_circle[path] = (adjusted_x, adjusted_y, adjusted_radius)
        
        # Map devices to their circles using metadata
        if path in device_metadata:
            device_map[device_metadata[path]['device'].name] = (adjusted_x, adjusted_y, adjusted_radius)

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
        elif 'enclave_' in path and 'device_' not in path and 'empty' not in path:
            # Enclave circle - only for pure enclave paths, not device paths or empty placeholders
            # Extract enclave ID from path like "root/enclave_1"
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
            
            # Create enhanced hover text for enclave
            level = get_hierarchy_level(path)
            
            # Find the corresponding enclave in the segmentation
            enclave_info = ""
            try:
                enclave_id_int = int(enclave_id)
                # Navigate to the correct segmentation level
                current_node = seg_node
                for i in range(level):
                    if current_node.children:
                        # Get the first child (simplified navigation)
                        first_child_key = list(current_node.children.keys())[0]
                        current_node = current_node.children[first_child_key]
                
                # Find the enclave in the current segmentation
                if enclave_id_int < len(current_node.seg.enclaves):
                    enclave = current_node.seg.enclaves[enclave_id_int]
                    device_count = len(enclave.devices)
                    sensitivity = enclave.sensitivity
                    enclave_info = f"<br>Sensitivity: {sensitivity:.2f}<br>Devices: {device_count}"
                else:
                    enclave_info = f"<br>Enclave not found"
            except (ValueError, IndexError, KeyError):
                enclave_info = f"<br>Info unavailable"
            
            hover_text = f"Enclave {enclave_id} (level{level}){enclave_info}"
            
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x - r, y0=y - r,
                x1=x + r, y1=y + r,
                line_color="black",
                fillcolor="rgba(0,255,0,0.1)",
                line_width=1
            )
            
            # Add hover point at the top of the circle (increasing in y-axis from center)
            hover_x = x  # Same x-coordinate as circle center
            hover_y = y + r  # Top of the circle (center y + radius)
            
            # Add invisible scatter point for hover functionality at the top
            fig.add_trace(go.Scatter(
                x=[hover_x],
                y=[hover_y],
                mode="markers",
                marker=dict(size=5, color="rgba(0,0,0,0)", line=dict(width=0)),
                hoverinfo="text",
                text=[hover_text],
                showlegend=False
            ))
        elif path == "root":
            # Skip drawing the root circle - we'll draw Internet separately
            pass

    # Draw edges at all levels of the hierarchy
    def draw_edges_for_node(node, current_path="root", is_outermost=True):
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
                draw_edges_for_node(child, child_path, is_outermost=False)
            
            # Also draw edges at the outermost level if this is the root
            if is_outermost:
                topology_edges = node.seg.topology.edges()
                edge_x, edge_y = [], []
                
                for i, j in topology_edges:
                    # Skip Internet connections as they're handled separately
                    if i == 0 or j == 0:
                        continue
                    
                    # Find the circles for these enclaves at the root level
                    enclave_i_path = f"root/enclave_{i}"
                    enclave_j_path = f"root/enclave_{j}"
                    
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
                        line=dict(color="red", width=1),  # Different color for outermost edges
                        hoverinfo="none",
                        showlegend=False
                    ))

    # Start drawing edges from the root node
    draw_edges_for_node(seg_node)

    # Draw Internet circle separately (positioned outside the main visualization)
    internet_x, internet_y = -1, 0.6  # Position to the right of the main visualization
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
        textfont=dict(size=18, color="orange"),
        hoverinfo="text",
        showlegend=False
    ))
    
    # Draw edges from Internet to enclaves that have Internet connections
    def draw_internet_edges(node, current_path="", is_outermost=True):
        """Draw edges from Internet to enclaves that connect to it."""
        
        # Draw Internet connections for the current level if it's the outermost level
        if is_outermost:
            topology_edges = node.seg.topology.edges()
            edge_x, edge_y = [], []
            
            for i, j in topology_edges:
                # If one of the endpoints is Internet (enclave 0)
                if i == 0 or j == 0:
                    # Find the non-Internet enclave
                    enclave_id = j if i == 0 else i
                    
                    # Look for the enclave circle at the current path level
                    enclave_path = f"{current_path}/enclave_{enclave_id}" if current_path else f"enclave_{enclave_id}"
                    
                    if enclave_path in path_to_circle:
                        x1, y1, r1 = path_to_circle[enclave_path]
                        
                        # Connect to the orange Internet circle
                        p1 = point_on_circle(x1, y1, r1, internet_x, internet_y)
                        p2 = point_on_circle(internet_x, internet_y, internet_r, x1, y1)
                        edge_x.extend([p1[0], p2[0], None])
                        edge_y.extend([p1[1], p2[1], None])
                    else:
                        print(f"Enclave path {enclave_path} not found in path_to_circle")
            
            if edge_x:
                # Orange dashed lines for Internet connections
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode="lines",
                    line=dict(color="orange", width=2, dash="dash"),
                    hoverinfo="none",
                    showlegend=False
                ))
        
        # Draw parent-child gateway connections for inner levels
        if not is_outermost and not node.children:
            # This is a leaf node at an inner level - check for gateway connections
            topology_edges = node.seg.topology.edges()
            edge_x, edge_y = [], []
            
            for i, j in topology_edges:
                # If one of the endpoints is index 0, this indicates a gateway to parent
                if i == 0 or j == 0:
                    # Find the gateway enclave (the non-zero one)
                    enclave_id = j if i == 0 else i
                    
                    # Look for the gateway enclave circle
                    enclave_path = f"{current_path}/enclave_{enclave_id}" if current_path else f"enclave_{enclave_id}"
                    child_x, child_y, child_r = path_to_circle[enclave_path]
                    parent_x, parent_y, parent_r = path_to_circle[current_path]
                    
                    # Calculate the optimal line along the direction connecting centers
                    # Direction vector from child to parent center
                    dx = parent_x - child_x
                    dy = parent_y - child_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0:
                        # Normalize the direction vector
                        dx_normalized = dx / distance
                        dy_normalized = dy / distance
                        
                        # Child perimeter point: away from parent center
                        p1_x = child_x - dx_normalized * child_r
                        p1_y = child_y - dy_normalized * child_r
                        
                        # Parent perimeter point: closest to child center
                        p2_x = parent_x - dx_normalized * parent_r
                        p2_y = parent_y - dy_normalized * parent_r
                        
                        edge_x.extend([p1_x, p2_x, None])
                        edge_y.extend([p1_y, p2_y, None])
    
            if edge_x:
                # Orange lines for parent-child gateway connections
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode="lines",
                    line=dict(color="orange", width=2),
                    hoverinfo="none",
                    showlegend=False
                ))
        
        # Recurse into children (if any)
        if node.children:
            for key, child in node.children.items():
                child_path = f"{current_path}/enclave_{key}" if current_path else f"enclave_{key}"
                draw_internet_edges(child, child_path, is_outermost=False)
    
    # Draw Internet edges
    draw_internet_edges(seg_node, "root")

    # Add config summary to title if root has config
    title = "Compositional Network Segmentation"
    if seg_node.config:
        config_summary = get_config_summary(seg_node)
        # title += f"<br><sub>{config_summary}</sub>"

    fig.update_layout(
        # title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1000, height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='white'
    )
    fig.show()
    
    return fig


# seg = load_segmentation_node("segmentations\Compositional\seg_Compositional_medium_20250830_171816.json")
# seg.print_details()
# draw_compositional_segmentation_circlify(seg)

