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
     timeloop-model input/sample_model.cfg
     ```

---

## Directory Structure

* **`runner/`**: Consolidated directory containing precompiled simulation executables (`looptree-model`, `timeloop-mapper`, `timeloop-model`, `timeloop-metrics`, `timeloop-design-space`, `timeloop-simple-mapper`, `einsumgraph`).
* **`input/`**: Unified repository for all YAML specifications and configurations:
  - `sample_model.cfg`: Sample architecture and mapping specification for deterministic evaluation.
  - `sample_mapper.yaml`: Comprehensive architecture, mapping constraints, and mapper search settings.
  - `sample_accelergy.yaml`: Accelergy integration configuration for accurate energy modeling.
  - `cnn_layer.yaml`, `gemm_ABZ.yaml`: Tensor problem definitions.
  - `TechFile/`: Semiconductor technology characterization files (Nangate 45nm LEF/Liberty).
  - `mma_gemm_accelerator/`: Concrete accelerator design project and DSE suite.
* **`outputs/`**: All simulation results, performance counters, mappings, XML dumps, and execution logs are automatically directed to this directory.
* **`src/` & `include/`**: C++ analytical simulation engine, polyhedral analysis pipeline, and application entry points (`master` branch).
* **`lib/`**: Dynamic shared libraries (`libtimeloop-model.so`, `libtimeloop-mapper.so`).

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

### Single-Layer Evaluation (`timeloop-model`)
```bash
source env/setup-env.bash
timeloop-model input/sample_model.cfg
```

### Mapping Exploration (`timeloop-mapper`)
```bash
source env/setup-env.bash
timeloop-mapper input/sample_mapper.yaml
```

### Accelergy-Enabled Exploration (`timeloop-mapper` with Accelergy)
```bash
source env/setup-env.bash
timeloop-mapper input/sample_accelergy.yaml
```

All generated summary tables (`*.stats.txt`), optimal tile mappings (`*.map.txt`), and structured logs (`*.map+stats.xml`) will be stored directly inside `outputs/`.
