# Timeloop Standalone Binary Distribution (User Branch)

[**📖 راهنمای فارسی کاربر نهایی (README-fa.md)**](README-fa.md)

This branch provides a **ready-to-run, precompiled distribution** of the Timeloop / LoopTree simulation framework.
**No C++ compiler, build tools, or SCons are required.**

---

## ⚡ Quick Start (3 Steps)

### 1. Install Runtime Dependencies
Only standard shared libraries are needed:
```bash
sudo apt update && sudo apt install -y \
    libyaml-cpp0.8 \
    libboost-iostreams1.83.0 \
    libncurses6 \
    libtinfo6
```

### 2. Set Up Environment
Run once per shell session:
```bash
source env/setup-env.bash
```
This adds `runner/` to your `PATH` and sets `LD_LIBRARY_PATH` to `lib/`.

### 3. Run Simulation
```bash
# Evaluate a fixed mapping (direct model evaluation)
timeloop-model inputs/configs/model/sample.cfg

# Search and explore optimal mapping (mapper heuristic)
timeloop-mapper inputs/configs/mapper/sample.yaml
```

All simulation outputs (`*.stats.txt`, `*.map.txt`, `*.map+stats.xml`) will automatically be generated in the **`outputs/`** directory.

---

## 📂 Directory Structure

```
.
├── runner/                 # Precompiled executables
│   ├── timeloop-model      # Evaluates a specific mapping
│   ├── timeloop-mapper     # Explores mapspace for optimal mapping
│   ├── timeloop-metrics    # Workload-independent area/energy characteristics
│   ├── timeloop-design-space
│   └── timeloop-simple-mapper
├── lib/                    # Precompiled shared library (libtimeloop-model.so)
├── inputs/                 # Architecture, problem, and tech configs
│   ├── configs/            # Architecture & mapper YAMLs
│   ├── problem-shapes/     # Mathematical problem definitions (CNN, GEMM)
│   └── TechFile/           # Physical characterization files (Nangate 45nm)
├── outputs/                # Default destination for all simulation outputs
├── env/
│   └── setup-env.bash      # Path configuration script
└── README.md
```

---

## 🛠️ Need Source Code or Customization?
If you need to modify the C++ simulation engine, add new hardware models, or build from source, switch to the developer branch:
```bash
git checkout master
```
See `README.md` on the `master` branch for full developer build instructions.
