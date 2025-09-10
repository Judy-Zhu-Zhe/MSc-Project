# Compositional MAP-Elites for Hierarchical Network Segmentation

This project implements a **Compositional MAP-Elites** evolutionary algorithm for **hierarchical network segmentation** in Python. It extends the traditional MAP-Elites approach to support multi-level network segmentation, enabling both macro-level (coarse-grained) and micro-level (fine-grained) segmentation strategies. The system evaluates how different hierarchical topologies and device assignments impact security, performance, and resilience metrics under simulated attacks.

**Author**: Judy Zhu  
**Project**: MSc Research - Hierarchical Network Segmentation

---

## 🔍 Key Features

### **Hierarchical Segmentation**
- **Multi-level architecture**: Supports nested segmentation with multiple refinement levels
- **Compositional optimization**: Each level can be optimized independently while maintaining hierarchy
- **Adaptive granularity**: Coarse-grained segmentation at higher levels, fine-grained at lower levels

### **Advanced Evaluation Metrics**
- **Security Loss**: Recursive computation with cross-level propagation prevention
- **Performance Loss**: Macro-level evaluation focusing on Internet connectivity
- **Resilience Loss**: Independent evaluation at each hierarchy level
- **Topology Distance**: Hierarchical distance calculation with level weighting
- **Attack Surface Exposure**: Vulnerability-based penalty scoring
- **Trust Separation Score**: Compromise value mismatch penalties
- **Sensitivity Penalty**: High-value device placement optimization

### **Evolutionary Algorithm**
- **Compositional MAP-Elites**: Multi-level archive management
- **Hierarchical mutation**: Topology, partition, and sensitivity mutations
- **Adaptation support**: Internal attack simulation and response
- **K-way minimum cut**: Alternative segmentation generation

### **Comprehensive Visualization**
- **Hierarchical topology visualization**: Multi-level network representation
- **Device color coding**: Zone-based color schemes (Business, Operations, Process, Safety)
- **Fitness heatmaps**: Multi-dimensional performance visualization
- **Infection propagation**: Attack spread visualization

---

## 📁 Project Structure

### **Core Algorithm Files**
- **`mapElites.py`** - Compositional MAP-Elites algorithm, hierarchical mutation operators, and recursive optimization
- **`network.py`** - Data structures: `Device`, `Enclave`, `Topology`, `Segmentation`, `SegmentationNode`
- **`config.py`** - Configuration management, topology generation, and hierarchical config classes
- **`simulation.py`** - Attack simulation logic with infection propagation and cleansing mechanisms

### **Evaluation & Metrics**
- **`metrics.py`** - Base evaluation metrics (security, performance, resilience, topology distance)
- **`metrics_hierarchy.py`** - Hierarchical versions of core metrics with recursive computation
- **`descriptors.py`** - Behavior descriptors for MAP-Elites archive characterization

### **Alternative Algorithms**
- **`kCut.py`** - K-way minimum cut algorithm implementation using spectral clustering
- **`v1.py`** - Version 1 implementation (legacy)
- **`v2.py`** - Version 2 implementation (legacy)

### **Visualization & Analysis**
- **`visualization.py`** - Comprehensive visualization tools including hierarchical topology plots
- **`main.py`** - Entry point with utility functions for common operations

### **Configuration Files**
- **`setups/configs/`** - YAML configuration files for different experiments
  - `config_compositional.yaml` - Multi-level compositional experiments
  - `config_test.yaml` - Testing configurations
  - `config_large.yaml` - Large-scale network experiments
  - `config_ot.yaml` - Operational Technology (OT) network configurations
- **`setups/device_profiles/`** - Device profile definitions
  - `device_profiles_enterprise.yaml` - Enterprise network devices
  - `device_profiles_ot.yaml` - OT/ICS devices with DuPont model zones
- **`setups/networks/`** - Network topology definitions
  - `enterprise_medium.yaml` - Medium-scale enterprise network
  - `enterprise_large.yaml` - Large-scale enterprise network

### **Data & Results**
- **`data/`** - Network datasets (Cisco, LANL)
- **`segmentations/`** - Saved segmentation results organized by experiment type
- **`imgs/`** - Generated visualizations and plots

---

## 🚀 Getting Started

### **1. Installation**
This project requires Python 3.9 or higher. Install dependencies:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `networkx>=2.8.0` - Graph algorithms and network analysis
- `numpy>=1.21.0` - Numerical computations
- `scikit-learn>=1.0.0` - Machine learning algorithms (for k-way cut)
- `matplotlib` - Visualization
- `pyyaml` - Configuration file parsing

### **2. Configuration**

#### **Available Configurations:**
- **`config_compositional.yaml`** - Multi-level compositional experiments with hierarchical optimization
- **`config_test.yaml`** - Testing configurations for development
- **`config_large.yaml`** - Large-scale network experiments (4-level hierarchy)
- **`config_ot.yaml`** - Operational Technology networks with DuPont model (5-level hierarchy)

#### **Configuration Parameters:**
- **Hierarchy levels**: Number of segmentation refinement levels
- **Enclaves per level**: Number of enclaves at each hierarchy level
- **Generations**: Evolution iterations per level
- **Evaluation metrics**: Security, performance, resilience weights
- **Behavior descriptors**: Archive characterization dimensions
- **Topology constraints**: Maximum degree, connectivity limits

### **3. Running Experiments**

#### **Basic Compositional Experiment:**
```bash
python main.py --config setups/configs/config_compositional.yaml --save
```

#### **Large-Scale Network:**
```bash
python main.py --config setups/configs/config_large.yaml --network setups/networks/enterprise_large.yaml --save
```

#### **OT Network (DuPont Model):**
```bash
python main.py --config setups/configs/config_ot.yaml --devices setups/device_profiles/device_profiles_ot.yaml --save
```

#### **Adaptation Experiment:**
```bash
python main.py --config setups/configs/config_compositional.yaml --seg segmentations/Compositional/structure/balanced.json --save
```

### **4. Utility Functions**

The `main.py` file provides convenient utility functions:

#### **Load and Visualize Segmentation:**
```python
from main import load_and_plot_segmentation
seg = load_and_plot_segmentation("segmentations/Compositional/structure/balanced.json")
```

#### **Infection and Adaptation:**
```python
from main import load_infect_and_plot, run_adaptation_experiment
# Infect devices and save result
infected_seg = load_infect_and_plot(
    "segmentations/Compositional/structure/balanced.json",
    {"IOT device": 1, "Web server": 1},
    "infected_mixed",
    cm
)
# Run adaptation experiment
adapted_seg = run_adaptation_experiment("infected_seg.json", -500.0, cm)
```

#### **Fitness Comparison:**
```python
from main import compare_fitness
fitness1, fitness2, topology_dist = compare_fitness(
    "seg1.json", "seg2.json", "reference.json", cm
)
```

#### **K-way Minimum Cut:**
```python
from main import run_kway_experiment_wrapper
kway_seg = run_kway_experiment_wrapper(cm, n_enclaves=8)
```

---

## 📊 Output Files

### **Segmentation Results:**
- **Best Segmentation**: `segmentations/<experiment_type>/seg_<timestamp>.json`
- **Archive Grid**: `segmentations/<experiment_type>/grid_<timestamp>.json`
- **Infected Segmentations**: `segmentations/<experiment_type>/infection/`

### **Visualizations:**
- **Hierarchical Topology**: `imgs/<experiment_type>/seg_<timestamp>.png`
- **Fitness Heatmaps**: `imgs/<experiment_type>/grid_<timestamp>.png`
- **Infection Propagation**: `imgs/<experiment_type>/infection/`

### **Experiment Types:**
- **`Compositional/`** - Multi-level compositional experiments
- **`Test/`** - Testing and development results
- **`OT_DuPont/`** - Operational Technology experiments
- **`Large/`** - Large-scale network experiments

---

## 🔬 Advanced Features

### **Hierarchical Metrics**
The system supports both flat and hierarchical evaluation metrics:

```python
from metrics_hierarchy import hierarchical_security_loss, hierarchical_performance_loss, hierarchical_resilience_loss

# Hierarchical evaluation
security_loss = hierarchical_security_loss(node, simulations)
performance_loss = hierarchical_performance_loss(node)
resilience_loss = hierarchical_resilience_loss(node)
```

### **DuPont Model Support**
Specialized support for Operational Technology networks following the DuPont security model:
- **Business Zone (L4)**: Blue-based color scheme
- **Operations Management Zone (L3)**: Green-based colors
- **Process Control Zone (L2/L1)**: Yellow-based colors
- **Safety System Zone**: Purple-based colors

### **K-way Minimum Cut Algorithm**
Alternative segmentation generation using spectral clustering:
```python
from kCut import KWayMinimumCut
kway = KWayMinimumCut(config_manager)
segmentation = kway.generate_segmentation(n_enclaves=8)
```

---

## 🛠️ Customization

### **Adding New Device Types:**
1. Update `setups/device_profiles/device_profiles_*.yaml`
2. Add color mappings in `visualization.py`
3. Update device generation logic in `config.py`

### **New Evaluation Metrics:**
1. Add metric function to `metrics.py`
2. Implement hierarchical version in `metrics_hierarchy.py`
3. Update configuration YAML files

### **Custom Behavior Descriptors:**
1. Add descriptor functions to `descriptors.py`
2. Update configuration files with new descriptors
3. Modify archive initialization in `mapElites.py`

### **Network Topologies:**
1. Create new network YAML in `setups/networks/`
2. Define device counts and connectivity patterns
3. Update device profile mappings

---

## 📚 Research Applications

This system is designed for research in:
- **Network Security**: Multi-level segmentation strategies
- **Evolutionary Algorithms**: Compositional MAP-Elites optimization
- **Critical Infrastructure**: OT/ICS network protection
- **Zero Trust Architecture**: Micro-segmentation implementation
- **Attack Simulation**: Hierarchical infection propagation

---

## 🤝 Contributing

To extend the system:
1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings to new functions
3. Update configuration files for new features
4. Test with both small and large-scale networks
5. Update this README with new functionality

---

## 📄 License

This project is part of MSc research work. Please cite appropriately if used in academic work.
