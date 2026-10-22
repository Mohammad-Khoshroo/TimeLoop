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
     timeloop-model inputs/sample_model_gemm.yaml
     ```

---

## Directory Structure

* **`runner/`**: Consolidated directory containing precompiled simulation executables (`looptree-model`, `timeloop-mapper`, `timeloop-model`, `timeloop-metrics`, `timeloop-design-space`, `timeloop-simple-mapper`, `einsumgraph`).
* **`lib/`**: Dynamic shared libraries (`libtimeloop-model.so`, `libtimeloop-mapper.so`).
* **`tech/`**: Semiconductor technology characterization libraries (Nangate 45nm LEF/Liberty/Verilog in `tech/45nm/`).
* **`examples/`**: Reference archive of preconfigured accelerator architectures and workloads:
  - **`gemm/`**: General Matrix Multiply accelerator architecture, constraints, workloads, and unified sample configurations.
  - **`eyeriss/`**: Eyeriss 256/1024-PE spatial array specifications, split problem layers, and Accelergy configurations.
  - **`cnn_conv/`**: 7D CNN convolution layers, ASPLOS 2014 benchmarks, and mapper/model templates.
  - **`simba/`**: NVIDIA Simba multi-chiplet accelerator architecture specification.
  - **`mma_gemm_accelerator/`**: Dedicated MMA hardware accelerator project, RiskMLP neural network layers, and automated DSE suite.
* **`inputs/`**: Active simulation workspace directory (configured by default with the GEMM accelerator):
  - `sample_gemm.yaml`: Consolidated single-file specification for instant mapping exploration.
  - `sample_model_gemm.yaml`: Consolidated single-file specification for deterministic evaluation.
  - `architectures/`: Modular hardware architecture definitions.
  - `constraints/`: Modular spatial and temporal mapping constraints.
  - `workloads/`: Tensor shape and problem dimension specifications.
  - `mapper/`: Search heuristics and convergence criteria.
* **`outputs/`**: All simulation results, performance counters, mappings, XML dumps, and execution logs are automatically directed to this directory.
* **`src/`**: Unified C++ source tree containing both implementation (`.cpp`) and header (`.hpp`) files side-by-side (`master` branch).

> **Switching Active Architecture:** To evaluate any other accelerator model, copy its files from `examples/<model>/*` into `inputs/` (e.g. `cp -r examples/eyeriss/* inputs/`).

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
timeloop-model inputs/sample_model_gemm.yaml
```

### Optimal Mapping Exploration (`timeloop-mapper`)
```bash
source env/setup-env.bash
timeloop-mapper inputs/sample_gemm.yaml
```

### Modular Multi-File Exploration
```bash
source env/setup-env.bash
timeloop-mapper \
    inputs/architectures/gemm_arch.yaml \
    inputs/constraints/gemm_constraints.yaml \
    inputs/workloads/gemm_problem.yaml \
    inputs/mapper/gemm_mapper.yaml
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
