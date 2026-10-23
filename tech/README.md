# Open-Source Standard Cell & Technology Libraries (`tech/`)

This directory provides a multi-node archive of open-source standard cell libraries, technology LEFs, Liberty (.lib) timing/power models, and structural/functional Verilog representations spanning cutting-edge 2nm GAAFETs to mature 180nm planar CMOS. These libraries are curated for silicon-level analytical energy/area modeling with Accelergy and Timeloop, as well as logic synthesis and physical design flows with Yosys and OpenROAD.

---

## 1. Multi-Node Technology Overview

| Node | PDK / Cell Library | Device Architecture | Origin / Maintainer | Nominal Voltage | Primary Collaterals |
|---|---|---|---|:---:|---|
| **2nm** | GT2N 6-Track | Nanosheet GAAFET (BSPDN) | Georgia Institute of Technology | 0.70 V | Tech LEF, Cell LEF, Liberty (LVT TT), Verilog |
| **7nm** | ASAP7 7.5-Track | FinFET (Predictive) | Arizona State University / ARM | 0.70 V | Tech LEF (1x), Cell LEF (1x RVT), NLDM Liberty (TT), Verilog |
| **45nm** | NanGate Open Cell Library | Planar Bulk CMOS | NanGate / NCSU FreePDK45 | 1.10 V | Tech LEF, Cell LEF, Liberty (Typical, Fast, Slow), Verilog |
| **130nm** | SkyWater SKY130 HD | Planar Bulk CMOS | SkyWater Technology / Google | 1.80 V | Tech LEF, Merged Cell LEF, Liberty (TT 25C 1.8V), Verilog |
| **180nm** | GlobalFoundries GF180MCU 7T | Planar Bulk CMOS | GlobalFoundries / Google | 5.0 V / 1.8 V | Tech LEF, Cell LEF (5LM), Liberty (TT 5.0V & 1.8V), Verilog |

---

## 2. Directory Structure and Component Breakdown

```
tech/
├── 2nm/                    # Georgia Tech GT2N 2nm Nanosheet GAAFET PDK
│   ├── gt2_tech.lef        # Technology LEF with layer rules and DRC pitch
│   ├── gt2_6t_w31_lvt.lef  # Standard cell macro layout abstract (6T, W=31nm, LVT)
│   ├── gt2_6t_w31_lvt_tt_0p7v25c.lib # Liberty timing and power model (0.7V, 25C, TT)
│   ├── gt2_6t_w13_lvt_tt_0p7v25c.lib # Liberty timing and power model (0.7V, 25C, TT, W=13nm)
│   ├── cells_clkgate.v     # Integrated clock-gating behavioral model
│   └── README.md           # GT2N architecture documentation and citation info
│
├── 7nm/                    # ASU / ARM ASAP7 7.5T Predictive FinFET Library
│   ├── asap7_tech_1x_201209.lef     # 1x true-scale 7nm technology LEF
│   ├── asap7sc7p5t_28_R_1x_220121a.lef # Regular VT (RVT) standard cell macro LEF
│   ├── asap7sc7p5t_SIMPLE_RVT_TT.lib   # Primitive logic gates Liberty model (TT)
│   ├── asap7sc7p5t_INVBUF_RVT_TT.lib   # Inverter and buffer Liberty model (TT)
│   ├── asap7sc7p5t_SEQ_RVT_TT.lib      # Flip-flops and sequential elements Liberty (TT)
│   ├── asap7sc7p5t_AO_RVT_TT.lib       # AND-OR complex gates Liberty model (TT)
│   ├── asap7sc7p5t_OA_RVT_TT.lib       # OR-AND complex gates Liberty model (TT)
│   ├── asap7sc7p5t_SIMPLE_RVT_TT.v     # Gate-level structural Verilog (simple gates)
│   ├── asap7sc7p5t_INVBUF_RVT_TT.v     # Inverters/buffers structural Verilog
│   ├── asap7sc7p5t_SEQ_RVT_TT.v        # Sequential cells structural Verilog
│   └── README.md                       # ASAP7 platform documentation
│
├── 45nm/                   # NanGate FreePDK45 Open Cell Library
│   ├── NangateOpenCellLibrary.tech.lef # FreePDK45 technology LEF
│   ├── NangateOpenCellLibrary.lef      # Complete standard cell macro abstract
│   ├── NangateOpenCellLibrary_typical.lib # Liberty timing & power (1.10V, 25C, TT)
│   ├── NangateOpenCellLibrary_fast.lib    # Liberty timing & power (1.25V, 0C, FF)
│   ├── NangateOpenCellLibrary_slow.lib    # Liberty timing & power (0.95V, 125C, SS)
│   └── NangateOpenCellLibrary.v           # Complete behavioral and functional Verilog
│
├── 130nm/                  # SkyWater SKY130 High-Density (HD) Standard Cell Library
│   ├── sky130_fd_sc_hd.tlef           # Technology LEF for SKY130 HD
│   ├── sky130_fd_sc_hd_merged.lef     # Consolidated cell macro LEF
│   ├── sky130_fd_sc_hd__tt_025C_1v80.lib # Liberty timing & power (1.80V, 25C, TT)
│   ├── cells_adders_hd.v              # High-density adders Verilog representation
│   ├── cells_clkgate_hd.v             # High-density clock gating Verilog
│   └── cells_latch_hd.v               # Latches Verilog representation
│
└── 180nm/                  # GlobalFoundries GF180MCU 7-Track Standard Cell Library
    ├── gf180mcu_5LM_1TM_9K_7t_tech.lef # Technology LEF for 5 metal layer stack
    ├── gf180mcu_5LM_1TM_9K_7t_sc.lef   # Standard cell macro LEF (7-track)
    ├── gf180mcu_fd_sc_mcu7t5v0__tt_025C_5v00.lib # Liberty (5.00V, 25C, TT)
    ├── gf180mcu_fd_sc_mcu7t5v0__tt_025C_1v80.lib # Liberty (1.80V core mode, 25C, TT)
    ├── cells_adders.v                  # Multi-bit and single-bit adders Verilog
    └── cells_latch.v                   # Transparent latches Verilog
```

---

## 3. Integration with Timeloop and Accelergy

When performing hardware design space exploration (DSE) or evaluating an analytical model:
* **Accelergy Area Table Reference:** Extract cell dimensions and pin densities from the `.lef` files to parameterize buffer footprints and arithmetic unit layouts.
* **Dynamic and Leakage Energy Scaling:** Use the Liberty (`.lib`) characterization tables to obtain operating voltage, static leakage, and dynamic transition energy numbers under specific PVT (Process, Voltage, Temperature) corners.
* **Technology Migration Comparison:** Compare identical accelerator architectures across 180nm, 130nm, 45nm, 7nm, and 2nm to assess cycle energy scaling, area scaling factors, and memory access power trade-offs.
