# Laser Optimizer Suite 🚀

A comprehensive framework for optimizing toolpaths in two-layer laser machining systems (Slow Platform + Fast Galvo). 

This suite unifies three evolutionary stages of the algorithm—from a basic Python prototype to a high-performance C++ production engine—under a single management API.

## 📂 Project Structure

```text
LaserOptimizerSuite/
├── manager.py              # Central Command API (Installer & Launcher)
├── inputs/                 # Drop your .svg patterns here
├── outputs/                # Generated G-Code (.gcode) and Visualizations (.html)
├── docs/                    # Research documentation, essays, and thesis
│
└── engines/                # The Solvers
    ├── py_v1/              # Basic Prototype (Python)
    ├── py_v2/              # Research Model (Python + Co-Evolutionary PSO)
    └── cpp_v3/             # Production Engine (C++17 + OpenMP)
```

## 🧠 The Algorithm

The system solves the **Dual-Mode Path Planning Problem**, where a slow global stage moves to "Stay Points" and a fast local scanner processes vectors within a limited field of view.

*   **Macro Layer (The General):** Optimizes the sequence and location of Platform Stay Points to minimize travel time.
    *   *Methods:* Particle Swarm Optimization (PSO), Co-Evolutionary Genetic Algorithms, Weighted K-Means++.
*   **Micro Layer (The Soldier):** Optimizes the laser marking sequence within a specific scanning field.
    *   *Methods:* Greedy Nearest Neighbor, Directional Heuristics, 2-Opt Local Search.

## ⚡ Quick Start

### Prerequisites
*   **Python 3.8+**
*   **CMake 3.14+**
*   **C++ Compiler** (GCC, Clang, or MSVC)

### 1. Installation
The `manager.py` script handles Python dependency installation and C++ compilation automatically.

```bash
python3 manager.py install
```

### 2. Run a Job (Production Mode)
Use the C++ engine (`cpp`) for maximum speed (< 2 seconds for 50k vectors).

```bash
# Normalize input to 40cm x 80cm
python3 manager.py run input.svg --engine cpp --width 0.4 --height 0.8
```

### 3. Run a Job (Research Mode)
Use the Python engine (`v2`) for detailed research metrics and Matplotlib visualizations.

```bash
# Normalize to 15cm x 15cm and simulate tiling
python3 manager.py run input.svg --engine v2 --width 0.15 --height 0.15 --tile 3
```

## 🎮 Command Line Arguments

All commands are executed via `manager.py`, following are some frequently used:

| Flag | Description | Default |
| :--- | :--- | :--- |
| `install` | Compiles C++ engine and installs Python requirements. | - |
| `run` | Executes an optimization job. | - |
| `--engine` | Select backend: `cpp` (Fast), `v2` (Research), `v1` (Legacy). | `cpp` |
| `--width` `H` | Normalize input SVG to target width (meters). | - |
| `--height` `W` | Normalize input SVG to target height (meters). | - |
| `--platform_speed` | Speed of the XY Stage (m/s). | 0.25 |
| `--mark_speed` | Laser cutting speed (m/s). | 2.5 |
| `--jump_speed` | Galvo repositioning speed (m/s). | 5.0 |
| `--field_size` | Size of the scanning field (m). | 0.2 |

## 📄 Documentation

Detailed algorithmic analysis, mathematical proofs, and the final essay are located in the **`docs/`** directory.

*   **`docs/`**: Contains the theoretical basis and performance benchmarks.
*   **`outputs/`**: Contains the visual proof of optimization (HTML interactive dashboards).

## 📝 License
Open-source project designed for computational geometry research. The author is irresponsible for any incident.
