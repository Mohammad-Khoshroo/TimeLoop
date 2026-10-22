# Timeloop Standalone Binary Distribution (User Branch)

[Persian Documentation (README-fa.md)](README-fa.md)

This branch provides a ready-to-run, precompiled binary distribution of the Timeloop / LoopTree simulation framework.
No C++ compiler, build tools, or SCons are required.

---

## Quick Start (3 Steps)

### 1. Install Runtime Dependencies
Only standard Linux shared libraries are needed:
```bash
sudo apt update && sudo apt install -y \
    libyaml-cpp0.8 \
    libboost-iostreams1.83.0 \
    libncurses6 \
    libtinfo6 \
    libisl23
```

### 2. Set Up Environment
Run once per shell session:
```bash
source env/setup-env.bash
```
This adds `runner/` to your `PATH` and sets `LD_LIBRARY_PATH` to `lib/`.

### 3. Run Simulation

#### Evaluate a Fixed Mapping
```bash
timeloop-model input/sample_model.cfg
```

#### Explore and Find Optimal Mapping
```bash
timeloop-mapper input/sample_mapper.yaml
```

#### Explore with Accelergy Energy Modeling
```bash
timeloop-mapper input/sample_accelergy.yaml
```

All simulation outputs (`*.stats.txt`, `*.map.txt`, `*.map+stats.xml`) will automatically be generated in the **`outputs/`** directory.

---

## Directory Structure

* **`runner/`**: Precompiled simulation executables:
  - `looptree-model`: Fused-layer and loop tree simulation engine.
  - `timeloop-model`: Direct mapping performance and energy evaluator.
  - `timeloop-mapper`: Heuristic mapper exploring optimal spatial and temporal tiling.
  - `timeloop-metrics`: Workload-independent silicon area and access energy extractor.
  - `timeloop-design-space`: Hardware design space exploration tool.
  - `timeloop-simple-mapper`: Lightweight mapper utility.
  - `einsumgraph`: Einsum graph generator and parser.
* **`lib/`**: Precompiled dynamic shared libraries (`libtimeloop-model.so`, `libtimeloop-mapper.so`).
* **`input/`**: Unified repository for all YAML specifications and configurations:
  - `sample_model.cfg`: Model evaluation configuration.
  - `sample_mapper.yaml`: Mapping exploration configuration.
  - `sample_accelergy.yaml`: Accelergy integration configuration.
  - `cnn_layer.yaml`, `gemm_ABZ.yaml`: Tensor problem definitions.
  - `TechFile/`: Technology characterization files (Nangate 45nm).
  - `mma_gemm_accelerator/`: MMA GEMM accelerator design and DSE suite.
* **`outputs/`**: Default output directory for all simulation reports, maps, and logs.
* **`env/`**: Shell environment configuration (`env/setup-env.bash`).

---

## Need Source Code or Customization?

If you need to modify the C++ simulation engine, add new hardware models, or build from source, switch to the developer branch:
```bash
git checkout master
```
See `README.md` on the `master` branch for developer build and Accelergy setup instructions.
