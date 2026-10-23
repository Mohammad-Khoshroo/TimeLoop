# Timeloop / LoopTree

[Persian Documentation (README-fa.md)](README-fa.md) | [Agent & Architecture Guide (AGENTS.md)](AGENTS.md)

## Overview

Timeloop is an analytical simulation, modeling, and mapping framework for deep learning architectures and tensor algebra accelerators.
This repository has been upgraded to the latest stable **LoopTree (Timeloop v3.0)** architecture, supporting fused-layer workloads, inter-layer data reuse, cross-layer tiling, and native integration with **Accelergy**.

---

## Repository Branches

This repository is maintained across two dedicated branches:

1. **`master` (Developer Branch):**
   - Full C++ source code, SCons build configuration, and the LoopTree analytical engine.
   - For researchers and hardware architects extending Timeloop or compiling from source.
2. **`user` (End-User Distribution Branch):**
   - Standalone, lightweight distribution containing precompiled binaries in `runner/` and input templates.
   - **No C++ compilation or build dependencies required.** Simply clone and run directly:
     ```bash
     git clone -b user https://github.com/Mohammad-Khoshroo/TimeLoop.git timeloop-user
     cd timeloop-user
     source env/setup-env.bash
     timeloop-model inputs/sample_mma_model.yaml
     ```

---

## Directory Structure

* **`runner/`**: Consolidated directory containing precompiled simulation executables (`looptree-model`, `timeloop-mapper`, `timeloop-model`, `timeloop-metrics`, `timeloop-design-space`, `timeloop-simple-mapper`, `einsumgraph`).
* **`lib/`**: Dynamic shared libraries (`libtimeloop-model.so`, `libtimeloop-mapper.so`).
* **`tech/`**: Multi-node semiconductor standard cell and technology characterization libraries (LEF, Tech LEF, Liberty, Verilog):
  - **`tech/2nm/`**: Georgia Tech GT2N 2nm Nanosheet GAAFET PDK.
  - **`tech/7nm/`**: ASU / ARM ASAP7 7.5T Predictive FinFET library.
  - **`tech/45nm/`**: NanGate FreePDK45 open cell library (typical, fast, slow corners).
  - **`tech/130nm/`**: SkyWater SKY130 High-Density (HD) planar CMOS library.
  - **`tech/180nm/`**: GlobalFoundries GF180MCU 7-track planar CMOS library.
* **`examples/`**: Reference archive of preconfigured accelerator architectures and workloads:
  - **`gemm/`**: General Matrix Multiply accelerator architecture, constraints, workloads, and unified sample configurations.
  - **`eyeriss/`**: Eyeriss 256/1024-PE spatial array specifications, split problem layers, and Accelergy configurations.
  - **`cnn_conv/`**: 7D CNN convolution layers, ASPLOS 2014 benchmarks, and mapper/model templates.
  - **`simba/`**: NVIDIA Simba multi-chiplet accelerator architecture specification.
  - **`mma_gemm_accelerator/`**: Dedicated MMA hardware accelerator project, RiskMLP neural network layers, and automated DSE suite.
* **`inputs/`**: Active simulation workspace directory (configured with the hardware MMA accelerator and RiskMLP workload):
  - `sample_mma_mapper.yaml`: Consolidated single-file specification for instant mapping exploration.
  - `sample_mma_model.yaml`: Consolidated single-file specification for deterministic evaluation.
  - `run_multi_tech_dse.py`: Comprehensive DSE execution script across all 5 semiconductor technology nodes.
  - `architectures/`: Modular hardware architecture definitions (`arch_mma.yaml`).
  - `constraints/`: Modular spatial and temporal mapping constraints (`constraints_mma.yaml`).
  - `workloads/`: Tensor shape and problem dimension specifications (`problem_risk_mlp.yaml`, `gemm_ABZ.yaml`).
  - `mapper/`: Search heuristics and convergence criteria (`mapper_mma.yaml`).
* **`right-out/`**: Multi-technology simulation outputs and analytical plots categorized by process node:
  - `2nm/`, `7nm/`, `45nm/`, `130nm/`, `180nm/`: Individual CSV summaries, candidate exploration dumps, XML reports, and 4 high-resolution plots (Pareto, Search Space, Power Breakdown, RiskMLP Execution).
  - `comparison/`: Cross-technology scaling comparison table (`tech_scaling_summary.md`), CSV dump, and comparative benchmark visualization (`tech_scaling_comparison.png`).
* **`outputs/`**: All simulation results, performance counters, mappings, XML dumps, and execution logs from direct runs are automatically directed to this directory.
* **`src/`**: Unified C++ source tree containing both implementation (`.cpp`) and header (`.hpp`) files side-by-side (`master` branch).

---

## Developer Setup and Build Instructions (`master` branch)

### 1. System Dependencies (Ubuntu / WSL)

Install core development tools and libraries:
```bash
sudo apt update && sudo apt install -y \
    scons build-essential libboost-all-dev libyaml-cpp-dev \
    libconfig++-dev libncurses-dev libtinfo-dev libgpm-dev \
    libgmp-dev libtool autoconf automake libisl-dev libpolylib64-dev
```

### 2. Polyhedral Libraries (NTL & Barvinok 0.41.6)

LoopTree leverages polyhedral integer-set analysis (ISL) for complex fused loop trees, requiring `NTL` and `Barvinok 0.41.6`:

```bash
# 1. Install NTL from system package or source
sudo apt install -y libntl-dev

# 2. Compile and install Barvinok 0.41.6
cd /tmp
wget https://barvinok.sourceforge.io/barvinok-0.41.6.tar.gz
tar -xzf barvinok-0.41.6.tar.gz
cd barvinok-0.41.6
./configure --enable-shared-barvinok --with-ntl=system
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 3. Accelergy Energy Framework Integration

Accelergy evaluates architectural energy and silicon area through technology-specific plug-ins:

```bash
pip install --break-system-packages accelergy accelergy-cacti-plug-in accelergy-table-based-plug-in
accelergy --version
```

### 4. Compilation with SCons

Build all simulation binaries with Accelergy enabled:
```bash
source env/setup-env.bash
scons -j$(nproc) --accelergy
```

Upon a successful build:
- All executables are automatically installed into `runner/`.
- The shared library `libtimeloop-model.so` is placed in `lib/`.
- The intermediate `build/` directory is automatically cleaned up to keep the repository lightweight.

---

## Running Simulations

### Deterministic Architecture Evaluation (`timeloop-model`)
```bash
source env/setup-env.bash
timeloop-model inputs/sample_mma_model.yaml
```

### Optimal Mapping Exploration (`timeloop-mapper`)
```bash
source env/setup-env.bash
timeloop-mapper inputs/sample_mma_mapper.yaml
```

### Multi-Technology DSE and Plot Generation
```bash
source env/setup-env.bash
python3 inputs/run_multi_tech_dse.py
```

### Modular Multi-File Exploration
```bash
source env/setup-env.bash
timeloop-mapper \
    inputs/architectures/arch_mma.yaml \
    inputs/constraints/constraints_mma.yaml \
    inputs/workloads/problem_risk_mlp.yaml \
    inputs/mapper/mapper_mma.yaml
```

### Running Archived Examples
You can also run models directly from the `examples/` directory:
```bash
# CNN 7D Convolution exploration
timeloop-mapper examples/cnn_conv/samples/sample_mapper.yaml

# Accelerated MMA GEMM DSE suite
cd examples/mma_gemm_accelerator
python3 run_mma_dse.py
python3 plot_mma_dse.py
```

All generated summary tables (`*.stats.txt`), optimal tile mappings (`*.map.txt`), and structured logs (`*.map+stats.xml`) will be stored directly inside `outputs/`.
