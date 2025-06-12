# MAP-Elites for Secure Network Segmentation

This project implements a MAP-Elites evolutionary algorithm to optimise **network segmentation strategies** in python. It evaluates how different topologies and device assignments impact security, performance, and resilience metrics under simulated attacks.

Judy Zhu

---

## 🔍 Features

- **Multi-objective evaluation** of segmentations based on:
  - Security loss (from a number of simulated attacks)
  - Performance loss (latency from segmentation)
  - Resilience loss (impact of enclave failures)
  - Dissimilarity (Euclidean distance between topologies)
- **MAP-Elites algorithm** for evolving diverse, high-performing segmentations.
- **Support for adaptation** simulates the spread of attacks from internal compromised devices.
- **Visualization** of segmentations and descriptor-fitness heatmaps.

---

## 📁 Project Structure

- **network.py** Data structures for devices, enclaves, topology, and segmentation
- **config.py** Experiment config, device generation, saving/loading files
- **simulation.py** Attack simulation logic
- **metrics.py** Loss functions (security, performance, resilience)
- **mapElites.py** MAP-Elites algorithm and mutation operators
- **visualization.py** Heatmap and topology drawing
- **main.py** Entry point for running experiments

## 🚀 Running Code

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Configure Experiment

Create a YAML file under configs/. You can customize:
- Number of enclaves, devices, generations, etc.
- Descriptor weights and metrics
- Topology constraints (e.g. max_isps, max_degree)

Example configurations are:
- **example.yaml** -- A complete YAML template with all configurable parameters and comments. 
- **config_exp1.yaml** -- *Experiment 1*: A basic experiment with fewer generations, focused primarily on security.
- **config_exp1.yaml** -- *Experiment 2*: A more extensive run with additional generations, optimising security, performance, and resilience metrics.
- **config_adp.yaml** -- *Adaptation Scenario*: Configured for simulating internal attacks. Optimizes for security loss and topology dissimilarity.

### 3. Run Optimization
Run main.py with flags:
- **--config \<path\>**:
Specifies the YAML configuration file that defines your experiment.
- **--save**: Optional flag. If included, results (optimised segmentation and loss grid) will be saved to \segmentations in json format, and visualisations will be saved to \imgs.

Example:
```
python main.py --config configs/config_exp1.yaml --save
```

### 4. Adaptation Run (Optional)
Run main.py with an additional flags:
- **--seg \<path\>**: Specifies a previously saved segmentation JSON file. The algorithm will randomly infect some devices (can be modified) and run optimisation based on the given segmentation.

Example:
```
python main.py --config configs/config_exp1.yaml --seg segmentations/Exp1/seg_Exp1_batch400_gen30_20250603_155556.json --save
```

## 📊 Output saved
- Best Segmentation: in *segmentations/<config_name>/seg_<filename>.json*
- Loss Grid: in *segmentations/<config_name>/grid_<filename>.json*
- Segmentation Topology: in *imgs/<config_name>/seg_<filename>.png*
- Fitness Heatmap: in *imgs/<config_name>/grid_<filename>.png*

## 📘 Future updates
- Modify *DEVICE_TYPES* and *DEVICE_GROUPS* to include other devices.
- Modify *metrics.py* for other evaluation metrics.
- Add new behavior descriptors in *mapElites.py*.
- Adapt compositional MAP-Elites for micro-segmentation.
