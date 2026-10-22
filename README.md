# Timeloop / LoopTree

[**📖 مطالعه دانشنامه و مستندات فارسی (README-fa.md)**](README-fa.md) | [**🤖 Agent & Architecture Guide (AGENTS.md)**](AGENTS.md)

## About
Timeloop is an analytical modeling and mapping framework for deep learning and tensor algebra accelerators.
This repository is updated to the latest stable **LoopTree (Timeloop v3.0)** architecture with fused-layer evaluation and Accelergy integration.

## Repository Branches

1. **`master` (Developer Branch):**
   - Contains full C++ source code, SCons build system, and LoopTree model engines.
   - For researchers and developers extending Timeloop or compiling from source.
2. **`user` (End-User Distribution Branch):**
   - Lightweight, standalone distribution with precompiled binaries in `runner/`.
   - **No C++ compiler or build steps needed.** Clone and run directly:
     ```bash
     git clone -b user https://github.com/Mohammad-Khoshroo/timeloop.git timeloop-user
     cd timeloop-user
     source env/setup-env.bash
     timeloop-model inputs/configs/model/sample.yaml
     ```

## Directory Structure
* **`runner/`**: Clean distribution of compiled binaries (`looptree-model`, `timeloop-mapper`, `timeloop-model`, etc.).
* **`inputs/`**: Unified repository for all YAML architecture specs, constraints, and problem definitions:
  - `inputs/configs/`: Architecture and mapper configurations.
  - `inputs/problem-shapes/`: Tensor problem definitions (CNN, GEMM).
  - `inputs/TechFile/`: Technology characterization libraries (e.g. 45nm).
* **`outputs/`**: All simulation statistics, maps, logs, and XML dumps are automatically routed into this folder.
* **`src/` & `include/`**: C++ core simulation engine and applications (`master` branch).
* **`lib/`**: Dynamic shared libraries (`libtimeloop-model.so`).

## Developer Build Instructions (`master` branch)

### 1. Prerequisites
```bash
sudo apt update && sudo apt install -y \
    scons build-essential libboost-all-dev libyaml-cpp-dev \
    libconfig++-dev libncurses-dev libtinfo-dev libgpm-dev \
    libgmp-dev libtool autoconf automake
```

### 2. Polyhedral Libraries (NTL & Barvinok 0.41.6)
```bash
# NTL
cd /tmp && wget https://libntl.org/ntl-11.5.1.tar.gz && tar -xzf ntl-11.5.1.tar.gz && cd ntl-11.5.1/src
./configure NTL_GMP_LIP=on SHARED=on && make -j$(nproc) && sudo make install

# Barvinok
cd /tmp && wget https://barvinok.sourceforge.io/barvinok-0.41.6.tar.gz && tar -xzf barvinok-0.41.6.tar.gz && cd barvinok-0.41.6
./configure --enable-shared-barvinok --with-ntl=system && make -j$(nproc) && sudo make install && sudo ldconfig
```

### 3. Accelergy Energy Model
```bash
pip install --break-system-packages accelergy accelergy-cacti-plug-in accelergy-table-based-plug-in
```

### 4. Build with SCons
```bash
source env/setup-env.bash
scons -j$(nproc) --accelergy
```
*Binaries will automatically install into `runner/` and the temporary `build/` directory is automatically cleaned upon success.*
