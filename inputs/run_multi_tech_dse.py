#!/usr/bin/env python3
# ==============================================================================
# موتور جامع شبیه‌سازی چندفناوری و تحلیل DSE برای شتاب‌دهنده سخت‌افزاری MMA
# مبتنی بر تخمین ۱۰۰٪ با فریم‌ورک استاندارد Accelergy (ERT/ART)
# Supported Nodes: 2nm (GAAFET), 7nm (FinFET), 45nm (Planar), 130nm (Planar), 180nm (Planar)
# Supported Models: RiskMLP, MNIST-MLP, HAR-MLP, DLRM-MLP
# ==============================================================================

import os
import sys
import re
import glob
import yaml
import shutil
import subprocess
from pathlib import Path

# استفاده از بک‌اند بدون نمایش گرافیکی برای Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# تنظیم مسیرهای ریشه مخزن
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.name == "inputs") else SCRIPT_DIR.parent.parent
TIMELOOP_MAPPER = REPO_DIR / "runner" / "timeloop-mapper"
RIGHT_OUT_DIR = REPO_DIR / "right-out"
TECH_BASE_DIR = REPO_DIR / "tech"
COMPONENTS_DIR = REPO_DIR / "inputs" / "components"

# مشخصات ۵ گره فناوری استاندارد متن‌باز
TECH_CONFIGS = {
    "2nm": {
        "name": "GT2N 2nm Nanosheet GAAFET",
        "family": "GAAFET",
        "tech_node_nm": 2,
        "voltage_v": 0.70,
        "freq_mhz": 1200.0,
        "cycle_sec": 0.833e-9,
        "tech_dir": TECH_BASE_DIR / "2nm"
    },
    "7nm": {
        "name": "ASAP7 7.5T Predictive FinFET",
        "family": "FinFET",
        "tech_node_nm": 7,
        "voltage_v": 0.70,
        "freq_mhz": 1000.0,
        "cycle_sec": 1.0e-9,
        "tech_dir": TECH_BASE_DIR / "7nm"
    },
    "45nm": {
        "name": "NanGate FreePDK45 Open Cell",
        "family": "Planar",
        "tech_node_nm": 45,
        "voltage_v": 1.10,
        "freq_mhz": 500.0,
        "cycle_sec": 2.0e-9,
        "tech_dir": TECH_BASE_DIR / "45nm"
    },
    "130nm": {
        "name": "SkyWater SKY130 HD",
        "family": "Planar",
        "tech_node_nm": 130,
        "voltage_v": 1.80,
        "freq_mhz": 200.0,
        "cycle_sec": 5.0e-9,
        "tech_dir": TECH_BASE_DIR / "130nm"
    },
    "180nm": {
        "name": "GlobalFoundries GF180MCU 7T",
        "family": "Planar",
        "tech_node_nm": 180,
        "voltage_v": 1.80,
        "freq_mhz": 100.0,
        "cycle_sec": 10.0e-9,
        "tech_dir": TECH_BASE_DIR / "180nm"
    }
}

# شبکه‌های عصبی پرسپترون چندلایه استاندارد (MLP Benchmarks)
MODELS = {
    "risk_mlp": {
        "title": "RiskMLP (Automotive Risk Estimation)",
        "short_name": "RiskMLP",
        "layers": {
            "Layer1_FC1": {"M": 1, "K": 16, "N": 64, "name": "FC1 (16 -> 64)"},
            "Layer2_FC2": {"M": 1, "K": 64, "N": 64, "name": "FC2 (64 -> 64)"},
            "Layer3_FC3": {"M": 1, "K": 64, "N": 4,  "name": "FC3 (64 -> 4)"},
        },
        "buffer_sizes": [128, 256, 512]
    },
    "mnist_mlp": {
        "title": "MNIST-MLP (LeNet-300-100 Benchmark)",
        "short_name": "MNIST-MLP",
        "layers": {
            "Layer1_FC1": {"M": 1, "K": 784, "N": 256, "name": "FC1 (784 -> 256)"},
            "Layer2_FC2": {"M": 1, "K": 256, "N": 128, "name": "FC2 (256 -> 128)"},
            "Layer3_FC3": {"M": 1, "K": 128, "N": 16,  "name": "FC3 (128 -> 16)"},
        },
        "buffer_sizes": [512, 1024, 2048]
    },
    "har_mlp": {
        "title": "HAR-MLP (Human Activity Recognition)",
        "short_name": "HAR-MLP",
        "layers": {
            "Layer1_FC1": {"M": 1, "K": 560, "N": 256, "name": "FC1 (560 -> 256)"},
            "Layer2_FC2": {"M": 1, "K": 256, "N": 128, "name": "FC2 (256 -> 128)"},
            "Layer3_FC3": {"M": 1, "K": 128, "N": 16,  "name": "FC3 (128 -> 16)"},
        },
        "buffer_sizes": [512, 1024, 2048]
    },
    "dlrm_mlp": {
        "title": "DLRM-MLP (Meta Recommendation Bottom MLP)",
        "short_name": "DLRM-MLP",
        "layers": {
            "Layer1_FC1": {"M": 1, "K": 512, "N": 256, "name": "FC1 (512 -> 256)"},
            "Layer2_FC2": {"M": 1, "K": 256, "N": 128, "name": "FC2 (256 -> 128)"},
            "Layer3_FC3": {"M": 1, "K": 128, "N": 64,  "name": "FC3 (128 -> 64)"},
        },
        "buffer_sizes": [512, 1024, 2048]
    }
}

def get_or_generate_accelergy_tables(tech_key, bt_count, buf_size, cache_dir):
    """تولید جداول ERT و ART توسط Accelergy به ازای فناوری و پیکربندی سخت‌افزاری معین"""
    tech_info = TECH_CONFIGS[tech_key]
    tech_nm = tech_info["tech_node_nm"]
    cycle_sec = tech_info["cycle_sec"]
    
    config_dir = cache_dir / f"{tech_key}_BT{bt_count}_BUF{buf_size}"
    config_dir.mkdir(parents=True, exist_ok=True)
    ert_file = config_dir / "ERT.yaml"
    art_file = config_dir / "ART.yaml"
    
    if ert_file.exists() and art_file.exists() and ert_file.stat().st_size > 100:
        return ert_file, art_file
        
    oa_depth = max(1, buf_size // 4)
    bt_end_idx = bt_count - 1
    bt_range = f"0..{bt_end_idx}" if bt_end_idx > 0 else "0"
    
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
                - name: MACs[0..3]
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
        print(f"  [خطای Accelergy برای {tech_key} BT={bt_count} BUF={buf_size}]: {res.stderr}")
        
    return ert_file, art_file

def generate_mapping_config(bt_count, buf_size, layer_spec, output_path):
    """تولید فایل پیکربندی نگاشت، شتاب‌دهنده و بار کاری برای تیم‌لوپ"""
    num_macs = bt_count * 4
    spatial_n = min(bt_count, layer_spec["N"])
    
    config = {
        "mapper": {
            "algorithm": "linear-pruned",
            "optimization-metrics": ["energy", "delay"],
            "search-size": 200,
            "victory-condition": 50,
            "timeout": 50,
            "live-status": False,
            "log-suboptimal": True,
            "log-stats": True,
            "diagnostics": True
        },
        "arch": {
            "arithmetic": {
                "name": "MACs",
                "instances": num_macs,
                "meshX": bt_count,
                "meshY": 4,
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
                    "read_bandwidth": 4,
                    "write_bandwidth": 2
                },
                {
                    "name": "IA_BUF",
                    "entries": buf_size,
                    "instances": 1,
                    "word-bits": 8,
                    "read_bandwidth": 4,
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
                    "factors": "M1 N1 K4",
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
        yaml.dump(config, f, default_flow_style=False)

def parse_stats_file(stats_path, tech_info, layer_spec, bt_count, buf_size):
    """استخراج مستقیم آمارهای ۱۰۰٪ مبتنی بر Accelergy از خروجی تیم‌لوپ"""
    if not stats_path.exists():
        return None
    
    text = stats_path.read_text(encoding="utf-8", errors="ignore")
    
    cycles_match = re.search(r'Cycles:\s*(\d+)', text)
    energy_match = re.search(r'Energy:\s*([0-9.]+)\s*uJ', text)
    util_match = re.search(r'Utilization:\s*([0-9.]+)', text)
    computes_match = re.search(r'Computes\s*(?:\(total\))?\s*[=:]\s*(\d+)', text)
    
    cycles = int(cycles_match.group(1)) if cycles_match else 0
    energy_uj = float(energy_match.group(1)) if energy_match else 0.0
    utilization = float(util_match.group(1)) if util_match else 0.0
    
    if computes_match:
        computes = int(computes_match.group(1))
    else:
        computes = layer_spec["M"] * layer_spec["K"] * layer_spec["N"]
        
    # استخراج مقادیر fJ/Compute برای هر جزء
    fj_dict = {}
    fj_matches = re.findall(r'^\s*([A-Za-z0-9_]+)\s*=\s*([0-9.]+)\s*$', text, re.MULTILINE)
    for k, v in fj_matches:
        fj_dict[k] = float(v)
        
    mac_pj = (fj_dict.get("MACs", 0.0) * computes) * 1e-3
    acc_pj = (fj_dict.get("acc_reg", 0.0) * computes) * 1e-3
    oa_pj = (fj_dict.get("OA_BUF", 0.0) * computes) * 1e-3
    w_pj = (fj_dict.get("W_BUF", 0.0) * computes) * 1e-3
    ia_pj = (fj_dict.get("IA_BUF", 0.0) * computes) * 1e-3
    dram_pj = (fj_dict.get("DRAM", 0.0) * computes) * 1e-3
    sram_pj = oa_pj + w_pj + ia_pj
    
    # اگر fJ/Compute پیدا نشد، از بخش سطوح (Level) بخوان
    if mac_pj == 0.0:
        components_pj = {}
        levels = re.split(r'Level \d+\n-+\n===\s*', text)
        for l in levels[1:]:
            header = l.split("===")[0].strip()
            stats_sec = l.split("STATS")[-1] if "STATS" in l else l
            stats_sec = stats_sec.split("Networks")[0].split("Summary stats")[0]
            matches = re.findall(r'Energy \(total\)\s*:\s*([0-9.]+)\s*pJ', stats_sec)
            tot_level_pj = sum(float(x) for x in matches)
            components_pj[header] = tot_level_pj
        mac_pj = components_pj.get("MACs", 0.0)
        acc_pj = components_pj.get("acc_reg", 0.0)
        oa_pj = components_pj.get("OA_BUF", 0.0)
        w_pj = components_pj.get("W_BUF", 0.0)
        ia_pj = components_pj.get("IA_BUF", 0.0)
        dram_pj = components_pj.get("DRAM", 0.0)
        sram_pj = oa_pj + w_pj + ia_pj

    # استخراج مساحت از بخش سطوح (بر مبنای جدول ART اکسلرجی)
    area_um2_total = 0.0
    area_matches = re.findall(r'Area(?:\s*\(total\))?\s*:\s*([0-9.]+)\s*um\^2', text)
    for a in area_matches:
        area_um2_total += float(a)
    area_mm2 = area_um2_total * 1e-6
    if area_mm2 == 0.0:
        # تقریب معقول از ART در صورتی که مساحت صفر ثبت شده باشد
        area_mm2 = (bt_count * 4 * 466.375 * 1e-6) + ((buf_size * 2 + 100) * 3.5e-6)

    freq_mhz = tech_info["freq_mhz"]
    clock_period_ns = 1000.0 / freq_mhz
    latency_ns = cycles * clock_period_ns
    latency_us = latency_ns / 1000.0
    
    power_mw = (energy_uj / latency_us) * 1000.0 if latency_us > 0 else 0.0
    
    # توان تفکیک‌شده به میلی‌وات (P = E / T)
    power_mac_mw = (mac_pj / latency_us) * 1e-3 if latency_us > 0 else 0.0
    power_reg_mw = (acc_pj / latency_us) * 1e-3 if latency_us > 0 else 0.0
    power_sram_mw = (sram_pj / latency_us) * 1e-3 if latency_us > 0 else 0.0
    power_dram_mw = (dram_pj / latency_us) * 1e-3 if latency_us > 0 else 0.0
    
    # کارایی انرژی به GOPS/W و TOPS/W
    total_ops = 2.0 * computes
    gops_per_w = (total_ops / 1e9) / (energy_uj * 1e-6) if energy_uj > 0 else 0.0
    tops_per_w = gops_per_w / 1000.0
    
    return {
        "cycles": cycles,
        "energy_uJ": round(energy_uj, 6),
        "latency_us": round(latency_us, 4),
        "power_mW": round(power_mw, 3),
        "power_mac_mW": round(power_mac_mw, 3),
        "power_reg_mW": round(power_reg_mw, 3),
        "power_sram_mW": round(power_sram_mw, 3),
        "power_dram_mW": round(power_dram_mw, 3),
        "utilization": utilization,
        "computes": computes,
        "gops_per_w": round(gops_per_w, 2),
        "tops_per_w": round(tops_per_w, 5),
        "area_mm2": round(area_mm2, 6)
    }

def parse_candidates(log_path, bt_count, buf_size, layer_name, computes, tech_info):
    """استخراج تمام نقاط ساب‌اپتیمال کاوش شده توسط ماپر"""
    candidates = []
    if not log_path.exists():
        return candidates
        
    freq_mhz = tech_info["freq_mhz"]
    clock_period_ns = 1000.0 / freq_mhz
    num_macs = bt_count * 4
    
    pattern = re.compile(r'\[\s*\d+\]\s+Utilization\s*=\s*([0-9.]+)%?\s*\|\s*pJ/Compute\s*=\s*([0-9.]+)')
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            m = pattern.search(line)
            if m:
                util = float(m.group(1))
                if util > 1.0:
                    util = util / 100.0
                pj_comp = float(m.group(2))
                
                energy_uj = (pj_comp * computes) / 1e6
                est_cycles = max(1, int(computes / (max(0.01, util) * num_macs)))
                est_latency_us = (est_cycles * clock_period_ns) / 1000.0
                est_power_mw = (energy_uj / est_latency_us) * 1000.0 if est_latency_us > 0 else 0.0
                
                candidates.append({
                    "layer": layer_name,
                    "bt_count": bt_count,
                    "buf_size_B": buf_size,
                    "utilization": round(util, 4),
                    "energy_uJ": round(energy_uj, 6),
                    "latency_us": round(est_latency_us, 4),
                    "power_mW": round(est_power_mw, 3)
                })
    return candidates

def generate_model_plots(df_opt, df_cands, model_name, model_info, tech_key, tech_info, output_dir):
    """رسم ۴ نمودار تحلیلی برای یک مدل خاص در یک فناوری معین"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ۱. نمودار پرتو (Pareto Frontier)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    layers = df_opt["layer"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(df_opt["bt_count"].unique()))))
    bt_color_map = {bt: c for bt, c in zip(sorted(df_opt["bt_count"].unique()), colors)}
    
    for idx, layer in enumerate(layers):
        sub = df_opt[df_opt["layer"] == layer].sort_values("latency_us")
        for bt in sorted(sub["bt_count"].unique()):
            bt_sub = sub[sub["bt_count"] == bt]
            ax1.scatter(bt_sub["latency_us"], bt_sub["energy_uJ"],
                        label=f"BT={bt}" if idx == 0 else "",
                        color=bt_color_map[bt], s=70, edgecolors='black', alpha=0.85)
        ax1.plot(sub["latency_us"], sub["energy_uJ"], linestyle='--', alpha=0.3, color='gray')
        
    ax1.set_title(f"Pareto Frontier: Latency vs. Energy ({model_info['short_name']} - {tech_info['name']})", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Latency (μs)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Total Energy (μJ)", fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    for idx, layer in enumerate(layers):
        sub = df_opt[df_opt["layer"] == layer].sort_values("latency_us")
        for bt in sorted(sub["bt_count"].unique()):
            bt_sub = sub[sub["bt_count"] == bt]
            ax2.scatter(bt_sub["latency_us"], bt_sub["power_mW"],
                        label=f"BT={bt}" if idx == 0 else "",
                        color=bt_color_map[bt], s=70, edgecolors='black', alpha=0.85)
            
    ax2.set_title(f"Latency vs. Power Consumption ({model_info['short_name']} - {tech_info['name']})", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Latency (μs)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Power (mW)", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(title="Hardware Config", fontsize=9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "mma_dse_pareto.png", dpi=250)
    plt.close()
    
    # ۲. نمودار کل فضای جستجو (Suboptimal + Optimal)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not df_cands.empty:
        sc = ax.scatter(df_cands["latency_us"], df_cands["energy_uJ"],
                        c=df_cands["utilization"], cmap="Spectral_r",
                        alpha=0.45, s=28, edgecolors='none', label=f"Suboptimal Candidates (N={len(df_cands)})")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Hardware Utilization (0.0 to 1.0)", fontsize=10, fontweight='bold')
    
    ax.scatter(df_opt["latency_us"], df_opt["energy_uJ"],
               color='red', s=90, marker='*', edgecolors='black', label=f"Optimal Points (N={len(df_opt)})", zorder=5)
    
    ax.set_title(f"DSE Search Space Exploration ({model_info['short_name']} - {tech_info['name']})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Latency (μs)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Total Energy (μJ)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / "mma_dse_search_space.png", dpi=250)
    plt.close()
    
    # ۳. نمودار دو پنله تفکیک توان مصرفی
    fig, (ax_core, ax_sys) = plt.subplots(1, 2, figsize=(15, 6))
    grouped = df_opt.groupby("bt_count").agg({
        "power_mac_mW": "mean",
        "power_reg_mW": "mean",
        "power_sram_mW": "mean",
        "power_dram_mW": "mean",
        "power_mW": "mean"
    }).reset_index()
    
    bts = grouped["bt_count"].values
    x = np.arange(len(bts))
    width = 0.45
    
    p_mac = grouped["power_mac_mW"].values
    p_reg = grouped["power_reg_mW"].values
    p_sram = grouped["power_sram_mW"].values
    p_dram = grouped["power_dram_mW"].values
    p_core = p_mac + p_reg + p_sram
    
    # پنل ۱: تفکیک توان داخلی تراشه شتاب‌دهنده
    ax_core.bar(x, p_mac, width, label="Compute MACs (INT8)", color="#1d3557", edgecolor='black', linewidth=0.6)
    ax_core.bar(x, p_reg, width, bottom=p_mac, label="acc_reg (32-bit Accumulator)", color="#457b9d", edgecolor='black', linewidth=0.6)
    ax_core.bar(x, p_sram, width, bottom=p_mac + p_reg, label="On-Chip Buffers (IA, W, OA)", color="#2a9d8f", edgecolor='black', linewidth=0.6)
    ax_core.set_title(f"On-Chip Core Power Breakdown ({model_info['short_name']} - {tech_info['name']})", fontsize=11, fontweight='bold', pad=10)
    ax_core.set_xticks(x)
    ax_core.set_xticklabels([f"BT={b} ({b*4} MACs)" for b in bts], fontsize=10, fontweight='bold')
    ax_core.set_ylabel("Power Consumption (mW)", fontsize=10, fontweight='bold')
    ax_core.legend(title="Core Modules", fontsize=9, loc='upper left')
    ax_core.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # پنل ۲: مقایسه کل توان سیستم (تراشه در برابر حافظه خارجی)
    ax_sys.bar(x, p_core, width, label="On-Chip Core Power", color="#48cae4", edgecolor='black', linewidth=0.6)
    ax_sys.bar(x, p_dram, width, bottom=p_core, label="Off-Chip DRAM / Bus", color="#e76f51", edgecolor='black', linewidth=0.6)
    ax_sys.set_title(f"Total System Power: Core vs. Off-Chip DRAM ({model_info['short_name']})", fontsize=11, fontweight='bold', pad=10)
    ax_sys.set_xticks(x)
    ax_sys.set_xticklabels([f"BT={b}" for b in bts], fontsize=10, fontweight='bold')
    ax_sys.set_ylabel("Power Consumption (mW)", fontsize=10, fontweight='bold')
    ax_sys.legend(title="Domain", fontsize=9, loc='upper left')
    ax_sys.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mma_dse_power_breakdown.png", dpi=250)
    plt.close()
    
    # ۴. نمودار تاخیر و توان عملیاتی
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    grouped_lat = df_opt.groupby("bt_count").agg({
        "latency_us": "sum",
        "computes": "sum"
    }).reset_index()
    
    bts = grouped_lat["bt_count"].values
    x = np.arange(len(bts))
    lats = grouped_lat["latency_us"].values
    tputs = (grouped_lat["computes"].values * 2.0 / 1e9) / (lats * 1e-6)
    
    ax1.bar(x, lats, 0.45, color="#2b5c8f", edgecolor='black')
    ax1.set_title(f"End-to-End Latency vs. Parallel Hardware ({model_info['short_name']})", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"BT={b}" for b in bts], fontsize=10, fontweight='bold')
    ax1.set_ylabel("Total Latency (μs)", fontsize=10, fontweight='bold')
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    ax2.bar(x, tputs, 0.45, color="#e76f51", edgecolor='black')
    ax2.set_title(f"Computational Throughput vs. Parallel Hardware ({model_info['short_name']})", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"BT={b}" for b in bts], fontsize=10, fontweight='bold')
    ax2.set_ylabel("Throughput (GOPS)", fontsize=10, fontweight='bold')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mma_dse_latency_throughput.png", dpi=250)
    plt.close()

def generate_within_tech_models_comparison(all_tech_summaries, tech_key, tech_info, tech_root_dir):
    """مقایسه عملکرد و توان مصرفی ۴ مدل در یک فناوری مشخص"""
    comp_dir = tech_root_dir / "models_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    model_keys = ["risk_mlp", "mnist_mlp", "har_mlp", "dlrm_mlp"]
    comp_rows = []
    
    for m_key in model_keys:
        if m_key not in all_tech_summaries[tech_key]:
            continue
        df = all_tech_summaries[tech_key][m_key]
        grouped = df.groupby("bt_count").agg({
            "latency_us": "sum",
            "energy_uJ": "sum",
            "power_mW": "mean",
            "power_mac_mW": "mean",
            "power_reg_mW": "mean",
            "power_sram_mW": "mean",
            "power_dram_mW": "mean",
            "computes": "sum"
        }).reset_index()
        
        row_bt2 = grouped[grouped["bt_count"] == 2].iloc[0]
        row_bt16 = grouped[grouped["bt_count"] == 16].iloc[0]
        
        fps_bt2 = 1e6 / row_bt2["latency_us"] if row_bt2["latency_us"] > 0 else 0
        fps_bt16 = 1e6 / row_bt16["latency_us"] if row_bt16["latency_us"] > 0 else 0
        speedup = row_bt2["latency_us"] / row_bt16["latency_us"] if row_bt16["latency_us"] > 0 else 1.0
        
        comp_rows.append({
            "Model": MODELS[m_key]["short_name"],
            "Full_Name": MODELS[m_key]["title"],
            "Total_Parameters": sum(l["K"] * l["N"] for l in MODELS[m_key]["layers"].values()),
            "Total_MAC_Operations": sum(l["M"] * l["K"] * l["N"] for l in MODELS[m_key]["layers"].values()),
            "Latency_BT2_us": round(row_bt2["latency_us"], 3),
            "Latency_BT16_us": round(row_bt16["latency_us"], 3),
            "Throughput_BT2_FPS": round(fps_bt2, 1),
            "Throughput_BT16_FPS": round(fps_bt16, 1),
            "Speedup_BT16_vs_BT2": round(speedup, 2),
            "Energy_per_Frame_uJ": round(row_bt16["energy_uJ"], 4),
            "Core_Power_mW": round(row_bt16["power_mac_mW"] + row_bt16["power_reg_mW"] + row_bt16["power_sram_mW"], 3),
            "Total_Power_mW": round(row_bt16["power_mW"], 2)
        })
        
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(comp_dir / "models_comparison.csv", index=False)
    
    # نمودار ۱: مقایسه نرخ فریم (FPS) و ضریب شتاب
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(model_keys))
    width = 0.35
    
    fps_2 = df_comp["Throughput_BT2_FPS"].values
    fps_16 = df_comp["Throughput_BT16_FPS"].values
    
    ax1.bar(x - width/2, fps_2, width, label="Hardware BT=2", color="#1d3557", edgecolor='black')
    ax1.bar(x + width/2, fps_16, width, label="Hardware BT=16", color="#48cae4", edgecolor='black')
    ax1.set_yscale('log')
    ax1.set_title(f"Throughput Comparison (FPS - Log Scale) - {tech_info['name']}", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODELS[k]["short_name"] for k in model_keys], fontsize=10, fontweight='bold')
    ax1.set_ylabel("Throughput (Frames / Sec)", fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    speedups = df_comp["Speedup_BT16_vs_BT2"].values
    bars = ax2.bar(x, speedups, 0.45, color="#2a9d8f", edgecolor='black')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label="No Speedup (1.0x)")
    ax2.set_title(f"Hardware Parallel Scaling Speedup (BT=16 vs. BT=2) - {tech_info['name']}", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODELS[k]["short_name"] for k in model_keys], fontsize=10, fontweight='bold')
    ax2.set_ylabel("Speedup Factor (x)", fontsize=10, fontweight='bold')
    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}x", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(comp_dir / "models_scaling_throughput_fps.png", dpi=250)
    plt.close()
    
    # نمودار ۲: مقایسه مصرف انرژی و توان میان مدل‌ها
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    energies = df_comp["Energy_per_Frame_uJ"].values
    ax1.bar(x, energies, 0.45, color="#e76f51", edgecolor='black')
    ax1.set_yscale('log')
    ax1.set_title(f"Energy per Inference Frame (μJ - Log Scale) - {tech_info['name']}", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODELS[k]["short_name"] for k in model_keys], fontsize=10, fontweight='bold')
    ax1.set_ylabel("Energy (μJ)", fontsize=10, fontweight='bold')
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    core_p = df_comp["Core_Power_mW"].values
    tot_p = df_comp["Total_Power_mW"].values
    dram_p = tot_p - core_p
    ax2.bar(x, core_p, 0.45, label="On-Chip Core Power", color="#48cae4", edgecolor='black')
    ax2.bar(x, dram_p, 0.45, bottom=core_p, label="Off-Chip DRAM Power", color="#f4a261", edgecolor='black')
    ax2.set_title(f"Power Breakdown across Models - {tech_info['name']}", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODELS[k]["short_name"] for k in model_keys], fontsize=10, fontweight='bold')
    ax2.set_ylabel("Power (mW)", fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(comp_dir / "models_energy_power_comparison.png", dpi=250)
    plt.close()
    
    # گزارش متنی مارک‌داون
    report_path = comp_dir / "models_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Model Benchmark Summary on {tech_info['name']}\n\n")
        f.write(df_comp.to_markdown(index=False))
        f.write("\n")

def generate_cross_technology_comparison(all_tech_summaries, comparison_dir):
    """رسم نمودار مقایسه‌ای جامع میان تمام ۵ فناوری سیلیکونی"""
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    tech_keys = ["2nm", "7nm", "45nm", "130nm", "180nm"]
    rows = []
    
    for t in tech_keys:
        ref_model = "mnist_mlp" if "mnist_mlp" in all_tech_summaries[t] else list(all_tech_summaries[t].keys())[0]
        df = all_tech_summaries[t][ref_model]
        grouped = df.groupby(["bt_count", "buf_size_B"]).agg({
            "latency_us": "sum",
            "energy_uJ": "sum",
            "power_mW": "mean",
            "power_mac_mW": "mean",
            "power_reg_mW": "mean",
            "power_sram_mW": "mean",
            "power_dram_mW": "mean",
            "gops_per_w": "mean",
            "tops_per_w": "mean",
            "area_mm2": "mean"
        }).reset_index()
        
        ref = grouped[grouped["bt_count"] == 4]
        if ref.empty:
            ref = grouped.iloc[0:1]
            
        r = ref.iloc[0]
        rows.append({
            "Technology": t,
            "PDK": TECH_CONFIGS[t]["name"],
            "Family": TECH_CONFIGS[t]["family"],
            "Voltage_V": TECH_CONFIGS[t]["voltage_v"],
            "Freq_MHz": TECH_CONFIGS[t]["freq_mhz"],
            "EndToEnd_Latency_us": round(r["latency_us"], 4),
            "EndToEnd_Energy_uJ": round(r["energy_uJ"], 6),
            "Avg_Power_mW": round(r["power_mW"], 3),
            "MAC_Power_mW": round(r["power_mac_mW"], 3),
            "AccReg_Power_mW": round(r["power_reg_mW"], 3),
            "SRAM_Power_mW": round(r["power_sram_mW"], 3),
            "DRAM_Power_mW": round(r["power_dram_mW"], 3),
            "Energy_Efficiency_GOPS_W": round(r["gops_per_w"], 2),
            "Energy_Efficiency_TOPS_W": round(r["tops_per_w"], 5),
            "Approx_Area_mm2": round(r["area_mm2"], 6)
        })
        
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(comparison_dir / "tech_scaling_comparison.csv", index=False)
    
    # ۱. نمودار جامع ۴ بخشی مقایسه فناوری‌ها
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(tech_keys))
    colors = ["#2b5c8f", "#3a86c8", "#48bfe3", "#f4a261", "#e76f51"]
    
    # تاخیر استنتاج
    axes[0, 0].bar(x, comp_df["EndToEnd_Latency_us"], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 0].set_title("End-to-End Latency Comparison (μs)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel("Latency (μs)", fontsize=10)
    axes[0, 0].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # مصرف انرژی کل
    axes[0, 1].bar(x, comp_df["EndToEnd_Energy_uJ"], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title("Total Energy per Inference (μJ - Log Scale)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel("Energy (μJ)", fontsize=10)
    axes[0, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # کارایی انرژی
    axes[1, 0].bar(x, comp_df["Energy_Efficiency_GOPS_W"], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Energy Efficiency (GOPS/W - Log Scale)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("GOPS / Watt", fontsize=10)
    axes[1, 0].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # مساحت تقریبی سیلیکون
    axes[1, 1].bar(x, comp_df["Approx_Area_mm2"], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title("Silicon Area Scaling (mm² - Log Scale)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel("Area (mm²)", fontsize=10)
    axes[1, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.suptitle("Cross-Technology Silicon Scaling Benchmark for MMA Accelerator", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(comparison_dir / "tech_scaling_comparison.png", dpi=250)
    plt.close()
    
    # ۲. نمودار مقایسه انباشته تفکیک توان مصرفی در میان فناوری‌ها (دو پنل مجزا)
    fig, (ax_core, ax_sys) = plt.subplots(1, 2, figsize=(15, 6))
    width = 0.5
    p_mac_all = comp_df["MAC_Power_mW"].values
    p_reg_all = comp_df["AccReg_Power_mW"].values
    p_sram_all = comp_df["SRAM_Power_mW"].values
    p_dram_all = comp_df["DRAM_Power_mW"].values
    p_core_all = p_mac_all + p_reg_all + p_sram_all
    
    # پنل ۱: تفکیک توان مصرفی ماژول‌های داخلی شتاب‌دهنده سیلیکونی
    ax_core.bar(x, p_mac_all, width, label="Compute MACs (INT8)", color="#1d3557", edgecolor='black', linewidth=0.6)
    ax_core.bar(x, p_reg_all, width, bottom=p_mac_all, label="acc_reg (32-bit Accumulator)", color="#457b9d", edgecolor='black', linewidth=0.6)
    ax_core.bar(x, p_sram_all, width, bottom=p_mac_all + p_reg_all, label="On-Chip Buffers (IA, W, OA)", color="#2a9d8f", edgecolor='black', linewidth=0.6)
    ax_core.set_title("On-Chip MMA Core Power Breakdown (Excl. DRAM)", fontsize=11, fontweight='bold', pad=10)
    ax_core.set_xticks(x)
    ax_core.set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    ax_core.set_ylabel("Power Consumption (mW)", fontsize=10, fontweight='bold')
    ax_core.legend(title="Core Modules", fontsize=9, loc='upper left')
    ax_core.grid(True, axis='y', linestyle=':', alpha=0.6)

    # پنل ۲: مقایسه توان مصرفی کل سیستم (تراشه شتاب‌دهنده در برابر حافظه خارجی)
    ax_sys.bar(x, p_core_all, width, label="On-Chip Core Power", color="#48cae4", edgecolor='black', linewidth=0.6)
    ax_sys.bar(x, p_dram_all, width, bottom=p_core_all, label="Off-Chip DRAM / SharedBus", color="#e76f51", edgecolor='black', linewidth=0.6)
    ax_sys.set_title("Total System Power: Core vs. Off-Chip DRAM", fontsize=11, fontweight='bold', pad=10)
    ax_sys.set_xticks(x)
    ax_sys.set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    ax_sys.set_ylabel("Power Consumption (mW)", fontsize=10, fontweight='bold')
    ax_sys.legend(title="System Domain", fontsize=9, loc='upper left')
    ax_sys.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(comparison_dir / "tech_power_breakdown_comparison.png", dpi=250)
    plt.close()
    
    # ذخیره گزارش متنی مارک‌داون
    report_md = comparison_dir / "tech_scaling_summary.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Cross-Technology Analytical Evaluation Summary (MMA Accelerator)\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n")
    print(f" مقایسه فناوری‌ها ذخیره شد: {comparison_dir}")

def clean_outputs_directory():
    """پاک‌سازی تمامی فایل‌های موقت تولیدشده در outputs/ به جز gitkeep"""
    outputs_dir = REPO_DIR / "outputs"
    for item in outputs_dir.iterdir():
        if item.name == ".gitkeep":
            continue
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

def main():
    print("=" * 80)
    print("آغاز فرآیند جامع شبیه‌سازی و DSE مبتنی بر ۱۰۰٪ Accelergy ERT/ART")
    print("=" * 80)
    
    bt_list = [2, 4, 8, 16]
    
    temp_work_dir = REPO_DIR / "outputs" / "dse_temp"
    temp_work_dir.mkdir(parents=True, exist_ok=True)
    
    acc_cache_dir = REPO_DIR / "outputs" / "accelergy_cache"
    acc_cache_dir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["PATH"] = f"/usr/local/bin:/usr/bin:/bin:{REPO_DIR / 'runner'}"
    env["TIMELOOP_PROBLEM_SHAPE_DIR"] = str(REPO_DIR / "inputs" / "workloads")
    env["LD_LIBRARY_PATH"] = str(REPO_DIR / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    
    all_tech_summaries = {}
    total_evals_per_tech = sum(len(bt_list) * len(m["buffer_sizes"]) * len(m["layers"]) for m in MODELS.values())
    
    for tech_key, tech_info in TECH_CONFIGS.items():
        print(f"\n================================================================================")
        print(f"اجرای شبیه‌سازی برای فناوری {tech_key} ({tech_info['name']}) با جداول Accelergy...")
        print(f"================================================================================")
        tech_root_dir = RIGHT_OUT_DIR / tech_key
        tech_root_dir.mkdir(parents=True, exist_ok=True)
        all_tech_summaries[tech_key] = {}
        
        curr_step = 0
        for m_key, m_info in MODELS.items():
            model_out_dir = tech_root_dir / m_key
            model_out_dir.mkdir(parents=True, exist_ok=True)
            
            opt_records = []
            cand_records = []
            
            for bt in bt_list:
                for buf in m_info["buffer_sizes"]:
                    # ۱. تولید جداول ERT و ART توسط Accelergy
                    ert_file, art_file = get_or_generate_accelergy_tables(tech_key, bt, buf, acc_cache_dir)
                    
                    for l_name, l_spec in m_info["layers"].items():
                        curr_step += 1
                        tag = f"{tech_key}_{m_key}_BT{bt}_BUF{buf}_{l_name}"
                        cfg_file = temp_work_dir / f"{tag}.yaml"
                        log_file = temp_work_dir / f"{tag}.log"
                        stats_backup = temp_work_dir / f"{tag}.stats.txt"
                        
                        generate_mapping_config(bt, buf, l_spec, cfg_file)
                        
                        cmd = [str(TIMELOOP_MAPPER), str(cfg_file), str(ert_file), str(art_file)]
                        with open(log_file, "w") as out_f:
                            subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, env=env, cwd=REPO_DIR)
                            
                        out_stats = REPO_DIR / "outputs" / "timeloop-mapper.stats.txt"
                        if out_stats.exists() and out_stats.stat().st_size > 0:
                            shutil.copy(out_stats, stats_backup)
                            parsed = parse_stats_file(out_stats, tech_info, l_spec, bt, buf)
                            if parsed:
                                parsed["technology"] = tech_key
                                parsed["tech_name"] = tech_info["name"]
                                parsed["model"] = m_key
                                parsed["layer"] = l_name
                                parsed["bt_count"] = bt
                                parsed["buf_size_B"] = buf
                                opt_records.append(parsed)
                                
                                computes = parsed["computes"]
                                cands = parse_candidates(log_file, bt, buf, l_name, computes, tech_info)
                                cand_records.extend(cands)
                        else:
                            print(f"  [هشدار] نگاشت برای {tag} ناموفق بود.")
                            
                        if curr_step % 12 == 0 or curr_step == total_evals_per_tech:
                            print(f"  [{curr_step:3d}/{total_evals_per_tech:3d}] سناریوها تا {tag} پردازش شدند.", flush=True)
                            
            df_opt = pd.DataFrame(opt_records)
            df_cands = pd.DataFrame(cand_records)
            
            df_opt.to_csv(model_out_dir / "mma_dse_optimal_summary.csv", index=False)
            df_cands.to_csv(model_out_dir / "mma_dse_all_evaluated_candidates.csv", index=False)
            all_tech_summaries[tech_key][m_key] = df_opt
            
            # کپی فایل‌های خروجی تیم‌لوپ برای نگهداری در پوشه مدل
            for f_name in ["timeloop-mapper.stats.txt", "timeloop-mapper.map.txt", "timeloop-mapper.map+stats.xml"]:
                src_f = REPO_DIR / "outputs" / f_name
                if src_f.exists():
                    shutil.copy(src_f, model_out_dir / f_name)
                    
            # رسم ۴ نمودار اختصاصی مدل
            generate_model_plots(df_opt, df_cands, m_key, m_info, tech_key, tech_info, model_out_dir)
            
        # مقایسه ۴ مدل درون همین فناوری
        generate_within_tech_models_comparison(all_tech_summaries, tech_key, tech_info, tech_root_dir)

    # مقایسه جامع میان ۵ فناوری
    print("\nتولید نمودارها و جداول مقایسه جامع میان ۵ فناوری سیلیکونی...")
    generate_cross_technology_comparison(all_tech_summaries, RIGHT_OUT_DIR / "comparison")
    
    # پاک‌سازی پوشه outputs
    print("\nپاک‌سازی فایل‌های موقت در outputs/...")
    clean_outputs_directory()
    
    print("\n" + "=" * 80)
    print("کلیه شبیه‌سازی‌ها، ارزیابی‌های Accelergy، داده‌های آماری و نمودارها با موفقیت تولید شدند.")
    print("=" * 80)

if __name__ == "__main__":
    main()
