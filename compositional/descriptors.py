from network import Segmentation
from config import MapElitesConfig

import statistics
from typing import List

def nb_high_deg_nodes(segmentation: Segmentation) -> int:
    """Returns the number of enclaves with more than 2 neighbours."""
    return sum(1 for n in segmentation.topology.adj_matrix if sum(n) > 3)

def std_devices(segmentation: Segmentation) -> float:
    """Returns the standard deviation of the number of devices in enclaves."""
    if len(segmentation.enclaves) > 1:
        return statistics.stdev([len(e.devices) for e in segmentation.enclaves])
    return 0.0

def std_high_value(segmentation: Segmentation) -> float:
    """Returns the standard deviation of the number of high-value devices in enclaves."""
    if len(segmentation.enclaves) > 1:
        return statistics.stdev([sum(1 for d in e.devices if d.compromise_value > 5) for e in segmentation.enclaves])
    return 0.0

def std_web_facing(segmentation: Segmentation) -> float:
    """Returns the standard deviation of the number of web-facing devices in enclaves."""
    if len(segmentation.enclaves) > 1:
        return statistics.stdev([sum(1 for d in e.devices if d.internet_sensitive) for e in segmentation.enclaves])
    return 0.0

def distance_high_val(segmentation: Segmentation) -> int:
    """Returns the sum of distances to the internet for high-value devices."""
    return sum(e.dist_to_internet for e in segmentation.enclaves for d in e.devices if d.compromise_value > 5 and e.dist_to_internet)

def behavior_descriptors(seg: Segmentation, descriptors: List[str]) -> List[float]:
    """Computes behavior descriptors dynamically based on selected keys."""
    values = []
    for name in descriptors:
        func = globals().get(name)
        if not callable(func):
            raise ValueError(f"Invalid descriptor function: '{name}' not found.")
        values.append(func(seg))
    return values

def bins(config: MapElitesConfig) -> List[float]:
    """Discretizes the behavior descriptor into bins."""
    # Better binning strategy?
    n_enclaves = config.n_enclaves
    num_std = sum(1 for d in config.descriptors if "std" in d)
    std_bin = max(2, int(round((config.batch_size * 3) ** (1 / num_std))))  # at least 2 per dim
    bin_widths = []
    for descriptor in config.descriptors:
        if "std" in descriptor:
            bin_widths.append(round((n_enclaves + 1) / std_bin, 2)) # Standard deviation descriptors
        elif "nb" in descriptor or "distance" in descriptor:
            bin_widths.append(1.0) # integer descriptors
        else:
            bin_widths.append(1.0)
    return bin_widths

def discretize(values: List[float], bin_widths: List[float]) -> List[float]:
    """Discretizes a list of values into bins based on bin widths."""
    assert len(values) == len(bin_widths), "Values and bin widths must have the same length"
    return [round(d // b * b, 2) for d, b in zip(values, bin_widths)]
