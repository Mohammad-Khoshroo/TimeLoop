#!/usr/bin/env python3
# ==============================================================================
# موتور DSE ابعاد عمق درخت جمع‌کننده (Adder Tree Depth) و ظرفیت بافرها (Buffer Capacity)
# شتاب‌دهنده سخت‌افزاری MMA مبتنی بر ۱۰۰٪ Accelergy ERT/ART
# ==============================================================================

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.name == "inputs") else SCRIPT_DIR
TIMELOOP_MAPPER = REPO_DIR / "runner" / "timeloop-mapper"
RIGHT_OUT_DIR = REPO_DIR / "right-out"
TECH_BASE_DIR = REPO_DIR / "tech"
COMPONENTS_DIR = REPO_DIR / "inputs" / "components"

TREE_DSE_OUT_DIR = RIGHT_OUT_DIR / "tree_and_buffer_dse"
TREE_DSE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# فناوری‌های منتخب برای کاوش عمق درخت و بافر
# ۴۵ نانومتر به عنوان نقطه طلایی صنعتی و ۷ نانومتر برای رده بالا
EVAL_TECHS = {
    "45nm": {
        "name": "NanGate FreePDK45 Open Cell",
        "tech_node_nm": 45,
        "voltage_v": 1.10,
        "freq_mhz": 500.0,
        "cycle_sec": 2.0e-9,
    },
    "7nm": {
        "name": "ASAP7 7.5T Predictive FinFET",
        "tech_node_nm": 7,
        "voltage_v": 0.70,
        "freq_mhz": 1000.0,
        "cycle_sec": 1.0e-9,
    }
}

# سطوح عمق درخت جمع‌کننده و تعداد ضرب‌کننده‌ها
# L: عمق درخت جمع‌کننده (Levels) -> K_reduction = 2^L
TREE_CONFIGS = {
    "L1_K2":  {"L": 1, "K_red": 2,  "label": "L=1 (2 MACs/col, 8 MACs total)"},
    "L2_K4":  {"L": 2, "K_red": 4,  "label": "L=2 (4 MACs/col, 16 MACs total - Baseline)"},
    "L3_K8":  {"L": 3, "K_red": 8,  "label": "L=3 (8 MACs/col, 32 MACs total)"},
    "L4_K16": {"L": 4, "K_red": 16, "label": "L=4 (16 MACs/col, 64 MACs total)"},
}

# بازه ظرفیت بافرها برای کاوش
BUFFER_CAPACITIES = [64, 128, 256, 512, 1024, 2048]

# بنچمارک‌های ارزیابی
EVAL_MODELS = {
    "risk_mlp": {
        "title": "RiskMLP (Automotive Collision Risk)",
        "layers": {
            "FC1": {"M": 1, "K": 16, "N": 64},
            "FC2": {"M": 1, "K": 64, "N": 64},
            "FC3": {"M": 1, "K": 64, "N": 4},
        }
    },
    "mnist_mlp": {
        "title": "MNIST-MLP (LeNet-300-100)",
        "layers": {
            "FC1": {"M": 1, "K": 784, "N": 256},
            "FC2": {"M": 1, "K": 256, "N": 128},
            "FC3": {"M": 1, "K": 128, "N": 16},
        }
    }
}

def get_or_generate_accelergy_tables(tech_key, bt_count, k_red, buf_size, cache_dir):
    """تولید جداول دقیق ERT و ART توسط Accelergy بر مبنای عمق درخت و اندازه بافر"""
    tech_info = EVAL_TECHS[tech_key]
    tech_nm = tech_info["tech_node_nm"]
    cycle_sec = tech_info["cycle_sec"]
    
    config_dir = cache_dir / f"{tech_key}_BT{bt_count}_K{k_red}_BUF{buf_size}"
    config_dir.mkdir(parents=True, exist_ok=True)
    ert_file = config_dir / "ERT.yaml"
    art_file = config_dir / "ART.yaml"
    
    if ert_file.exists() and art_file.exists() and ert_file.stat().st_size > 100:
        return ert_file, art_file
        
    oa_depth = max(1, buf_size // 4)
    bt_end_idx = bt_count - 1
    bt_range = f"0..{bt_end_idx}" if bt_end_idx > 0 else "0"
    mac_end_idx = k_red - 1
    mac_range = f"0..{mac_end_idx}" if mac_end_idx > 0 else "0"
    
    acc_arch_yaml = f"""architecture:
  version: 0.4
  subtree:
    - name: system
      attributes:
        technology: {tech_nm}
        global_cycle_seconds: {cycle_sec}
      local:
        - name: DRAM
          class: mma_dram
          attributes:
            technology: {tech_nm}
            type: "LPDDR4"
            width: 64
            block_size: 16
            word_bits: 8
            global_cycle_seconds: {cycle_sec}
      subtree:
        - name: MMA_core
          local:
            - name: IA_BUF
              class: smartbuffer_SRAM
              attributes:
                technology: {tech_nm}
                width: 8
                depth: {buf_size}
                global_cycle_seconds: {cycle_sec}
          subtree:
            - name: BT[{bt_range}]
              local:
                - name: W_BUF
                  class: smartbuffer_SRAM
                  attributes:
                    technology: {tech_nm}
                    width: 8
                    depth: {buf_size}
                    global_cycle_seconds: {cycle_sec}
                - name: OA_BUF
                  class: smartbuffer_SRAM
                  attributes:
                    technology: {tech_nm}
                    width: 32
                    depth: {oa_depth}
                    global_cycle_seconds: {cycle_sec}
                - name: acc_reg
                  class: mma_regfile
                  attributes:
                    technology: {tech_nm}
                    width: 32
                    depth: 1
                    global_cycle_seconds: {cycle_sec}
                - name: MACs[{mac_range}]
                  class: mma_mac
                  attributes:
                    technology: {tech_nm}
                    datawidth: 8
                    global_cycle_seconds: {cycle_sec}
"""
    arch_file = config_dir / "acc_arch.yaml"
    arch_file.write_text(acc_arch_yaml, encoding="utf-8")
    
    cmd = [
        "accelergy",
        str(arch_file),
        str(COMPONENTS_DIR / "mma_components.yaml"),
        str(COMPONENTS_DIR / "smartbuffer_SRAM.yaml"),
        "-o", str(config_dir)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [خطای Accelergy برای {tech_key} BT={bt_count} K={k_red} BUF={buf_size}]: {res.stderr}")
        
    return ert_file, art_file

def generate_mapping_config(bt_count, k_red, buf_size, layer_spec, output_path):
    """تولید کانفیگ نگاشت تیم‌لوپ با در نظر گرفتن عمق درخت و اندازه بافر"""
    num_macs = bt_count * k_red
    spatial_n = min(bt_count, layer_spec["N"])
    
    # ضریب K مکانی بر اساس عمق درخت و ابعاد لایه
    spatial_k = min(k_red, layer_spec["K"])
    
    config = {
        "mapper": {
            "algorithm": "linear-pruned",
            "optimization-metrics": ["energy", "delay"],
            "search-size": 150,
            "victory-condition": 40,
            "timeout": 40,
            "live-status": False,
            "log-suboptimal": False,
            "log-stats": True,
            "diagnostics": False
        },
        "arch": {
            "arithmetic": {
                "name": "MACs",
                "instances": num_macs,
                "meshX": bt_count,
                "meshY": k_red,
                "word-bits": 8
            },
            "storage": [
                {
                    "name": "acc_reg",
                    "entries": 1,
                    "instances": bt_count,
                    "meshX": bt_count,
                    "word-bits": 32,
                    "read_bandwidth": 2,
                    "write_bandwidth": 2
                },
                {
                    "name": "OA_BUF",
                    "entries": max(1, buf_size // 4),
                    "instances": bt_count,
                    "meshX": bt_count,
                    "word-bits": 32,
                    "read_bandwidth": 2,
                    "write_bandwidth": 2
                },
                {
                    "name": "W_BUF",
                    "entries": buf_size,
                    "instances": bt_count,
                    "meshX": bt_count,
                    "word-bits": 8,
                    "read_bandwidth": k_red,
                    "write_bandwidth": 2
                },
                {
                    "name": "IA_BUF",
                    "entries": buf_size,
                    "instances": 1,
                    "word-bits": 8,
                    "read_bandwidth": k_red,
                    "write_bandwidth": 4
                },
                {
                    "name": "DRAM",
                    "technology": "DRAM",
                    "instances": 1,
                    "word-bits": 8,
                    "block-size": 16,
                    "bandwidth": 8.0
                }
            ]
        },
        "mapspace": {
            "constraints": [
                {"target": "acc_reg", "type": "datatype", "keep": ["Z"], "bypass": ["A", "B"]},
                {"target": "OA_BUF",  "type": "datatype", "keep": ["Z"], "bypass": ["A", "B"]},
                {"target": "W_BUF",   "type": "datatype", "keep": ["B"], "bypass": ["A", "Z"]},
                {"target": "IA_BUF",  "type": "datatype", "keep": ["A"], "bypass": ["B", "Z"]},
                {"target": "DRAM",    "type": "datatype", "keep": ["A", "B", "Z"]},
                {
                    "target": "IA_BUF",
                    "type": "spatial",
                    "factors": f"M1 N{spatial_n} K1",
                    "permutation": "MNK"
                },
                {
                    "target": "acc_reg",
                    "type": "spatial",
                    "factors": f"M1 N1 K{spatial_k}",
                    "permutation": "MNK"
                }
            ]
        },
        "problem": {
            "shape": "gemm-ABZ",
            "M": layer_spec["M"],
            "K": layer_spec["K"],
            "N": layer_spec["N"]
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)

def parse_stats_file(stats_path, tech_info, layer_spec):
    """استخراج نتایج دقیق از فایل آماری خروجی Timeloop"""
    if not stats_path.exists():
        return None
        
    text = stats_path.read_text(encoding="utf-8", errors="ignore")
    
    cycles_m = re.search(r'Cycles:\s*(\d+)', text)
    energy_m = re.search(r'Energy:\s*([0-9.]+)\s*uJ', text)
    util_m = re.search(r'Utilization:\s*([0-9.]+)', text)
    
    cycles = int(cycles_m.group(1)) if cycles_m else 0
    energy_uj = float(energy_m.group(1)) if energy_m else 0.0
    utilization = float(util_m.group(1)) if util_m else 0.0
    
    computes = layer_spec["M"] * layer_spec["K"] * layer_spec["N"]
    
    # خواندن fJ/Compute برای هر جزء
    fj_matches = re.findall(r'^\s*([A-Za-z0-9_]+)\s*=\s*([0-9.]+)\s*$', text, re.MULTILINE)
    fj_dict = {k: float(v) for k, v in fj_matches}
    
    mac_pj = (fj_dict.get("MACs", 0.0) * computes) * 1e-3
    acc_pj = (fj_dict.get("acc_reg", 0.0) * computes) * 1e-3
    oa_pj = (fj_dict.get("OA_BUF", 0.0) * computes) * 1e-3
    w_pj = (fj_dict.get("W_BUF", 0.0) * computes) * 1e-3
    ia_pj = (fj_dict.get("IA_BUF", 0.0) * computes) * 1e-3
    dram_pj = (fj_dict.get("DRAM", 0.0) * computes) * 1e-3
    sram_pj = oa_pj + w_pj + ia_pj
    
    # مساحت از ART
    area_matches = re.findall(r'Area(?:\s*\(total\))?\s*:\s*([0-9.]+)\s*um\^2', text)
    area_um2_total = sum(float(a) for a in area_matches)
    area_mm2 = area_um2_total * 1e-6
    
    cycle_time = tech_info["cycle_sec"]
    latency_sec = cycles * cycle_time
    latency_us = latency_sec * 1e6
    
    power_mw = (energy_uj / latency_sec) * 1e-3 if latency_sec > 0 else 0.0
    onchip_energy_uj = (mac_pj + acc_pj + sram_pj) * 1e-6
    core_power_mw = (onchip_energy_uj / latency_sec) * 1e-3 if latency_sec > 0 else 0.0
    dram_power_mw = power_mw - core_power_mw
    
    return {
        "cycles": cycles,
        "energy_uj": energy_uj,
        "utilization": utilization,
        "latency_us": latency_us,
        "power_mw": power_mw,
        "core_power_mw": core_power_mw,
        "dram_power_mw": dram_power_mw,
        "area_mm2": area_mm2,
        "mac_pj": mac_pj,
        "sram_pj": sram_pj,
        "dram_pj": dram_pj,
        "edp_js": (energy_uj * 1e-6) * latency_sec
    }

def plot_tree_depth_exploration(df_results, out_dir):
    """رسم نمودارهای تحلیل اثر عمق درخت جمع‌کننده بر عملکرد، انرژی و مساحت"""
    # ۱. نمودار اثر عمق درخت بر تاخیر و بهره‌وری (به تفکیک مدل‌ها در ۴۵ نانومتر با بافر ۲۵۶)
    df_sub = df_results[(df_results["technology"] == "45nm") & (df_results["buf_size_B"] == 256)]
    
    models = df_sub["model"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for m in models:
        df_m = df_sub[df_sub["model"] == m].sort_values("k_reduction")
        axes[0].plot(df_m["k_reduction"], df_m["latency_us"], marker="o", linewidth=2.5, markersize=8, label=m.upper())
        axes[1].plot(df_m["k_reduction"], df_m["utilization"] * 100, marker="s", linewidth=2.5, markersize=8, label=m.upper())
        
    axes[0].set_title("Inference Latency vs. Adder Tree Reduction Width (K_red)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Adder Tree Reduction Width (K_red = 2^Tree_Depth)", fontsize=11)
    axes[0].set_ylabel("End-to-End Latency (us)", fontsize=11)
    axes[0].set_xticks([2, 4, 8, 16])
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)
    
    axes[1].set_title("Compute Array Utilization vs. Adder Tree Depth", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Adder Tree Reduction Width (K_red = 2^Tree_Depth)", fontsize=11)
    axes[1].set_ylabel("Compute Utilization (%)", fontsize=11)
    axes[1].set_xticks([2, 4, 8, 16])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / "adder_tree_depth_latency_utilization.png", dpi=300)
    plt.close()
    
    # ۲. نمودار اثر عمق درخت بر مساحت و توان هسته
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in models:
        df_m = df_sub[df_sub["model"] == m].sort_values("k_reduction")
        axes[0].plot(df_m["k_reduction"], df_m["area_mm2"], marker="^", linewidth=2.5, markersize=8, label=m.upper())
        axes[1].plot(df_m["k_reduction"], df_m["core_power_mw"], marker="d", linewidth=2.5, markersize=8, label=m.upper())
        
    axes[0].set_title("Silicon Area vs. Adder Tree Reduction Width", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Adder Tree Reduction Width (K_red)", fontsize=11)
    axes[0].set_ylabel("Total Area (mm^2)", fontsize=11)
    axes[0].set_xticks([2, 4, 8, 16])
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)
    
    axes[1].set_title("On-Chip Core Power vs. Adder Tree Reduction Width", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Adder Tree Reduction Width (K_red)", fontsize=11)
    axes[1].set_ylabel("Core Power (mW)", fontsize=11)
    axes[1].set_xticks([2, 4, 8, 16])
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / "adder_tree_depth_area_power.png", dpi=300)
    plt.close()

def plot_buffer_capacity_exploration(df_results, out_dir):
    """رسم نمودارهای کاوش ظرفیت بافرها بر ترافیک DRAM و بازدهی انرژی"""
    # در عمق درخت پایه K=4 و فناوری ۴۵ نانومتر
    df_sub = df_results[(df_results["technology"] == "45nm") & (df_results["k_reduction"] == 4)]
    
    models = df_sub["model"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for m in models:
        df_m = df_sub[df_sub["model"] == m].sort_values("buf_size_B")
        axes[0].plot(df_m["buf_size_B"], df_m["dram_pj"] * 1e-6, marker="o", linewidth=2.5, markersize=8, label=m.upper())
        axes[1].plot(df_m["buf_size_B"], df_m["edp_js"] * 1e12, marker="s", linewidth=2.5, markersize=8, label=m.upper())
        
    axes[0].set_title("DRAM Energy vs. On-Chip Buffer Capacity", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Buffer Capacity per SRAM Bank (Bytes)", fontsize=11)
    axes[0].set_ylabel("DRAM Energy (uJ)", fontsize=11)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([64, 128, 256, 512, 1024, 2048])
    axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)
    
    axes[1].set_title("Energy-Delay Product (EDP) vs. Buffer Capacity", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Buffer Capacity per SRAM Bank (Bytes)", fontsize=11)
    axes[1].set_ylabel("EDP (pJ * s)", fontsize=11)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([64, 128, 256, 512, 1024, 2048])
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / "buffer_capacity_dram_energy_edp.png", dpi=300)
    plt.close()

def main():
    print("=" * 80)
    print("آغاز فرآیند جامع کاوش DSE بر روی عمق درخت جمع‌کننده و ابعاد بافرها")
    print("مبتنی بر فریم‌ورک استاندارد ۱۰۰٪ Accelergy ERT/ART")
    print("=" * 80)
    
    temp_work_dir = REPO_DIR / "outputs" / "dse_tree_temp"
    temp_work_dir.mkdir(parents=True, exist_ok=True)
    
    acc_cache_dir = REPO_DIR / "outputs" / "accelergy_tree_cache"
    acc_cache_dir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["PATH"] = f"/usr/local/bin:/usr/bin:/bin:{REPO_DIR / 'runner'}"
    env["TIMELOOP_PROBLEM_SHAPE_DIR"] = str(REPO_DIR / "inputs" / "workloads")
    env["LD_LIBRARY_PATH"] = str(REPO_DIR / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    
    results = []
    bt_count = 4  # تعداد بهینه ستون‌ها (BT=4) بر اساس DSE مرحله قبل
    
    total_steps = len(EVAL_TECHS) * len(EVAL_MODELS) * (len(TREE_CONFIGS) + len(BUFFER_CAPACITIES))
    curr_step = 0
    
    for tech_key, tech_info in EVAL_TECHS.items():
        print(f"\n[گره فناوری {tech_key}]:")
        
        for m_key, m_info in EVAL_MODELS.items():
            print(f"  --> مدل بنچمارک: {m_info['title']}")
            
            # ۱. کاوش عمق درخت جمع‌کننده (در بافر ثابت ۲۵۶ بایت)
            buf_fixed = 256
            for tree_key, tree_cfg in TREE_CONFIGS.items():
                curr_step += 1
                k_red = tree_cfg["K_red"]
                ert_file, art_file = get_or_generate_accelergy_tables(tech_key, bt_count, k_red, buf_fixed, acc_cache_dir)
                
                # اجرای ارزیابی به ازای هر لایه مدل
                layer_stats = []
                for l_name, l_spec in m_info["layers"].items():
                    tag = f"{tech_key}_{m_key}_K{k_red}_BUF{buf_fixed}_{l_name}"
                    cfg_file = temp_work_dir / f"{tag}.yaml"
                    log_file = temp_work_dir / f"{tag}.log"
                    
                    generate_mapping_config(bt_count, k_red, buf_fixed, l_spec, cfg_file)
                    cmd = [str(TIMELOOP_MAPPER), str(cfg_file), str(ert_file), str(art_file)]
                    with open(log_file, "w") as out_f:
                        subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, env=env, cwd=REPO_DIR)
                        
                    out_stats = REPO_DIR / "outputs" / "timeloop-mapper.stats.txt"
                    if out_stats.exists() and out_stats.stat().st_size > 0:
                        parsed = parse_stats_file(out_stats, tech_info, l_spec)
                        if parsed:
                            layer_stats.append(parsed)
                            
                if layer_stats:
                    tot_cycles = sum(s["cycles"] for s in layer_stats)
                    tot_energy_uj = sum(s["energy_uj"] for s in layer_stats)
                    avg_util = np.mean([s["utilization"] for s in layer_stats])
                    tot_lat_us = sum(s["latency_us"] for s in layer_stats)
                    tot_area_mm2 = layer_stats[0]["area_mm2"]
                    tot_core_pwr = np.mean([s["core_power_mw"] for s in layer_stats])
                    tot_dram_pwr = np.mean([s["dram_power_mw"] for s in layer_stats])
                    tot_dram_pj = sum(s["dram_pj"] for s in layer_stats)
                    tot_sram_pj = sum(s["sram_pj"] for s in layer_stats)
                    tot_mac_pj = sum(s["mac_pj"] for s in layer_stats)
                    tot_edp = (tot_energy_uj * 1e-6) * (tot_lat_us * 1e-6)
                    
                    results.append({
                        "experiment": "tree_depth_sweep",
                        "technology": tech_key,
                        "model": m_key,
                        "tree_depth_L": tree_cfg["L"],
                        "k_reduction": k_red,
                        "mac_count": bt_count * k_red,
                        "buf_size_B": buf_fixed,
                        "cycles": tot_cycles,
                        "latency_us": tot_lat_us,
                        "throughput_fps": 1e6 / tot_lat_us if tot_lat_us > 0 else 0,
                        "utilization": avg_util,
                        "energy_uj": tot_energy_uj,
                        "core_power_mw": tot_core_pwr,
                        "dram_power_mw": tot_dram_pwr,
                        "area_mm2": tot_area_mm2,
                        "dram_pj": tot_dram_pj,
                        "sram_pj": tot_sram_pj,
                        "mac_pj": tot_mac_pj,
                        "edp_js": tot_edp
                    })
                    print(f"    [عمق درخت {tree_key}]: Latency={tot_lat_us:.2f}us | Util={avg_util*100:.1f}% | Area={tot_area_mm2:.4f}mm2 | Energy={tot_energy_uj:.3f}uJ")
                    
            # ۲. کاوش ظرفیت بافرها (در عمق درخت پایه K=4)
            k_fixed = 4
            for buf_cap in BUFFER_CAPACITIES:
                curr_step += 1
                ert_file, art_file = get_or_generate_accelergy_tables(tech_key, bt_count, k_fixed, buf_cap, acc_cache_dir)
                
                layer_stats = []
                for l_name, l_spec in m_info["layers"].items():
                    tag = f"{tech_key}_{m_key}_K{k_fixed}_BUF{buf_cap}_{l_name}"
                    cfg_file = temp_work_dir / f"{tag}.yaml"
                    log_file = temp_work_dir / f"{tag}.log"
                    
                    generate_mapping_config(bt_count, k_fixed, buf_cap, l_spec, cfg_file)
                    cmd = [str(TIMELOOP_MAPPER), str(cfg_file), str(ert_file), str(art_file)]
                    with open(log_file, "w") as out_f:
                        subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, env=env, cwd=REPO_DIR)
                        
                    out_stats = REPO_DIR / "outputs" / "timeloop-mapper.stats.txt"
                    if out_stats.exists() and out_stats.stat().st_size > 0:
                        parsed = parse_stats_file(out_stats, tech_info, l_spec)
                        if parsed:
                            layer_stats.append(parsed)
                            
                if layer_stats:
                    tot_cycles = sum(s["cycles"] for s in layer_stats)
                    tot_energy_uj = sum(s["energy_uj"] for s in layer_stats)
                    avg_util = np.mean([s["utilization"] for s in layer_stats])
                    tot_lat_us = sum(s["latency_us"] for s in layer_stats)
                    tot_area_mm2 = layer_stats[0]["area_mm2"]
                    tot_core_pwr = np.mean([s["core_power_mw"] for s in layer_stats])
                    tot_dram_pwr = np.mean([s["dram_power_mw"] for s in layer_stats])
                    tot_dram_pj = sum(s["dram_pj"] for s in layer_stats)
                    tot_sram_pj = sum(s["sram_pj"] for s in layer_stats)
                    tot_mac_pj = sum(s["mac_pj"] for s in layer_stats)
                    tot_edp = (tot_energy_uj * 1e-6) * (tot_lat_us * 1e-6)
                    
                    results.append({
                        "experiment": "buffer_capacity_sweep",
                        "technology": tech_key,
                        "model": m_key,
                        "tree_depth_L": 2,
                        "k_reduction": k_fixed,
                        "mac_count": bt_count * k_fixed,
                        "buf_size_B": buf_cap,
                        "cycles": tot_cycles,
                        "latency_us": tot_lat_us,
                        "throughput_fps": 1e6 / tot_lat_us if tot_lat_us > 0 else 0,
                        "utilization": avg_util,
                        "energy_uj": tot_energy_uj,
                        "core_power_mw": tot_core_pwr,
                        "dram_power_mw": tot_dram_pwr,
                        "area_mm2": tot_area_mm2,
                        "dram_pj": tot_dram_pj,
                        "sram_pj": tot_sram_pj,
                        "mac_pj": tot_mac_pj,
                        "edp_js": tot_edp
                    })
                    print(f"    [ظرفیت بافر {buf_cap:4d}B]: Latency={tot_lat_us:.2f}us | DRAM Energy={tot_dram_pj*1e-6:.3f}uJ | EDP={tot_edp*1e12:.2f} pJ*s")

    df_all = pd.DataFrame(results)
    csv_path = TREE_DSE_OUT_DIR / "tree_and_buffer_dse_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n[+] نتایج شبیه‌سازی در {csv_path} ذخیره شد.")
    
    # تولید نمودارها
    plot_tree_depth_exploration(df_all[df_all["experiment"] == "tree_depth_sweep"], TREE_DSE_OUT_DIR)
    plot_buffer_capacity_exploration(df_all[df_all["experiment"] == "buffer_capacity_sweep"], TREE_DSE_OUT_DIR)
    print(f"[+] نمودارهای تحلیلی با رزولوشن بالا در {TREE_DSE_OUT_DIR} تولید شدند.")
    
    # پاک‌سازی خودکار پوشه خروجی مطابق قوانین
    shutil.rmtree(temp_work_dir, ignore_errors=True)
    shutil.rmtree(acc_cache_dir, ignore_errors=True)
    for p in (REPO_DIR / "outputs").glob("timeloop-mapper*"):
        try:
            p.unlink()
        except:
            pass
    print("[+] پوشه outputs کاملاً پاک‌سازی شد و قوانین مخزن رعایت گردید.")

if __name__ == "__main__":
    main()
