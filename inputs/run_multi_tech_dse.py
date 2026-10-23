#!/usr/bin/env python3
# ==============================================================================
# موتور جامع شبیه‌سازی چندفناوری و تحلیل DSE برای شتاب‌دهنده سخت‌افزاری MMA
# Multi-Node Technology DSE & Analytical Simulation Engine for MMA GEMM Accelerator
# Supported Nodes: 2nm (GAAFET), 7nm (FinFET), 45nm (Planar), 130nm (Planar), 180nm (Planar)
# ==============================================================================

import os
import sys
import re
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

# مشخصات ۵ گره فناوری متن‌باز
TECH_CONFIGS = {
    "2nm": {
        "name": "GT2N 2nm Nanosheet GAAFET",
        "family": "GAAFET",
        "voltage_v": 0.70,
        "freq_mhz": 1200.0,
        "energy_scale": 0.038,       # مقیاس انرژی نسبت به 45nm
        "area_scale": 0.005,         # مقیاس مساحت نسبت به 45nm
        "tech_dir": TECH_BASE_DIR / "2nm"
    },
    "7nm": {
        "name": "ASAP7 7.5T Predictive FinFET",
        "family": "FinFET",
        "voltage_v": 0.70,
        "freq_mhz": 1000.0,
        "energy_scale": 0.100,       # مقیاس انرژی نسبت به 45nm
        "area_scale": 0.040,         # مقیاس مساحت نسبت به 45nm
        "tech_dir": TECH_BASE_DIR / "7nm"
    },
    "45nm": {
        "name": "NanGate FreePDK45 Open Cell",
        "family": "Planar",
        "voltage_v": 1.10,
        "freq_mhz": 500.0,
        "energy_scale": 1.000,       # گره مرجع
        "area_scale": 1.000,
        "tech_dir": TECH_BASE_DIR / "45nm"
    },
    "130nm": {
        "name": "SkyWater SKY130 HD",
        "family": "Planar",
        "voltage_v": 1.80,
        "freq_mhz": 200.0,
        "energy_scale": 6.800,       # مقیاس انرژی نسبت به 45nm
        "area_scale": 8.300,
        "tech_dir": TECH_BASE_DIR / "130nm"
    },
    "180nm": {
        "name": "GlobalFoundries GF180MCU 7T",
        "family": "Planar",
        "voltage_v": 1.80,
        "freq_mhz": 100.0,
        "energy_scale": 14.500,      # مقیاس انرژی نسبت به 45nm
        "area_scale": 16.000,
        "tech_dir": TECH_BASE_DIR / "180nm"
    }
}

# لایه‌های شبکه عصبی پرسپترون چندلایه (RiskMLP)
RISK_MLP_LAYERS = {
    "Layer1_FC1": {"M": 1, "K": 16, "N": 64, "name": "FC1 (16 -> 64)"},
    "Layer2_FC2": {"M": 1, "K": 64, "N": 64, "name": "FC2 (64 -> 64)"},
    "Layer3_FC3": {"M": 1, "K": 64, "N": 3,  "name": "FC3 (64 -> 3)"},
}

def generate_mapping_config(bt_count, buf_size, layer_spec, output_path):
    """تولید فایل کانفیگ مستقل برای تیم‌لوپ"""
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
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def parse_stats_file(stats_path, tech_info, layer_spec):
    """استخراج آمارهای شبیه‌سازی و اعمال ضرایب مقیاس فناوری"""
    if not stats_path.exists():
        return None
    
    text = stats_path.read_text(encoding="utf-8", errors="ignore")
    
    cycles_match = re.search(r'Cycles:\s*(\d+)', text)
    energy_match = re.search(r'Energy:\s*([0-9.]+)\s*uJ', text)
    util_match = re.search(r'Utilization:\s*([0-9.]+)', text)
    computes_match = re.search(r'Computes\s*(?:\(total\))?\s*[=:]\s*(\d+)', text)
    
    cycles = int(cycles_match.group(1)) if cycles_match else 0
    raw_energy_uj = float(energy_match.group(1)) if energy_match else 0.0
    utilization = float(util_match.group(1)) if util_match else 0.0
    
    if computes_match:
        computes = int(computes_match.group(1))
    else:
        computes = layer_spec["M"] * layer_spec["K"] * layer_spec["N"]
    
    # مقیاس‌بندی انرژی با فناوری مربوطه
    energy_scale = tech_info["energy_scale"]
    energy_uj = raw_energy_uj * energy_scale
    
    freq_mhz = tech_info["freq_mhz"]
    clock_period_ns = 1000.0 / freq_mhz
    latency_ns = cycles * clock_period_ns
    latency_us = latency_ns / 1000.0
    
    power_mw = (energy_uj / latency_us) * 1000.0 if latency_us > 0 else 0.0
    
    # کارایی انرژی به GOPS/W و TOPS/W
    total_ops = 2.0 * computes
    gops_per_w = (total_ops / 1e9) / (energy_uj * 1e-6) if energy_uj > 0 else 0.0
    tops_per_w = gops_per_w / 1000.0
    
    # تخمین مساحت تقریبی سیلیکون (mm^2)
    base_area_mm2 = 0.12 * (cycles / 260.0)
    scaled_area_mm2 = base_area_mm2 * tech_info["area_scale"]
    
    return {
        "cycles": cycles,
        "energy_uJ": round(energy_uj, 6),
        "latency_us": round(latency_us, 4),
        "power_mW": round(power_mw, 3),
        "utilization": utilization,
        "computes": computes,
        "gops_per_w": round(gops_per_w, 2),
        "tops_per_w": round(tops_per_w, 5),
        "area_mm2": round(scaled_area_mm2, 6)
    }

def parse_candidates(log_path, bt_count, buf_size, layer_name, computes, tech_info):
    """استخراج تمام نقاط ساب‌اپتیمال جستجو شده"""
    candidates = []
    if not log_path.exists():
        return candidates
        
    freq_mhz = tech_info["freq_mhz"]
    clock_period_ns = 1000.0 / freq_mhz
    num_macs = bt_count * 4
    energy_scale = tech_info["energy_scale"]
    
    # الگوی تطبیق خروجی لاگ تیم‌لوپ برای هر کاندید
    pattern = re.compile(r'\[\s*\d+\]\s+Utilization\s*=\s*([0-9.]+)%?\s*\|\s*pJ/Compute\s*=\s*([0-9.]+)')
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            m = pattern.search(line)
            if m:
                util = float(m.group(1))
                if util > 1.0:
                    util = util / 100.0
                pj_comp = float(m.group(2)) * energy_scale
                
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

def generate_tech_plots(df_opt, df_cands, tech_key, tech_info, output_dir):
    """رسم نمودارهای کامل برای هر فناوری"""
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
        
    ax1.set_title(f"Pareto Frontier: Latency vs. Energy ({tech_info['name']})", fontsize=11, fontweight='bold')
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
            
    ax2.set_title(f"Latency vs. Power Consumption ({tech_info['name']})", fontsize=11, fontweight='bold')
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
               color='gold', s=140, marker='*', edgecolors='black', linewidths=1.2,
               label="Optimal Points", zorder=10)
    
    ax.set_title(f"DSE Search Space Candidates ({tech_info['name']})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Latency (μs)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Total Energy (μJ)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "mma_dse_search_space.png", dpi=250)
    plt.close()
        
    # ۳. نمودار تفکیک توان لایه‌ها
    fig, ax = plt.subplots(figsize=(10, 5))
    sub = df_opt[df_opt["buf_size_B"] == 256].copy()
    if sub.empty:
        sub = df_opt.copy()
    labels = [f"{row['layer'].split('_')[0]}\nBT={row['bt_count']}" for _, row in sub.iterrows()]
    x = np.arange(len(labels))
    p_tot = sub["power_mW"].values
    
    ax.bar(x, p_tot, 0.55, color="#1f77b4", edgecolor='black', alpha=0.85)
    ax.set_title(f"Power Dissipation across Hardware Configurations ({tech_info['name']})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Workload Layer & BT Configuration", fontsize=10, fontweight='bold')
    ax.set_ylabel("Power (mW)", fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mma_dse_power_breakdown.png", dpi=250)
    plt.close()
    
    # ۴. استنتاج کل شبکه RiskMLP
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    grouped = df_opt.groupby(["bt_count", "buf_size_B"]).agg({
        "latency_us": "sum",
        "energy_uJ": "sum",
        "power_mW": "mean"
    }).reset_index()
    
    ref_buf = 256 if 256 in grouped["buf_size_B"].values else grouped["buf_size_B"].iloc[0]
    sub = grouped[grouped["buf_size_B"] == ref_buf].sort_values("bt_count")
    
    latencies = sub["latency_us"].values
    fps = [1e6 / lat if lat > 0 else 0 for lat in latencies]
    
    ax1_twin = ax1.twinx()
    ax1.bar([str(bt) for bt in sub["bt_count"]], latencies, color="#1d3557", width=0.45, label="Latency (μs)")
    ax1_twin.plot([str(bt) for bt in sub["bt_count"]], fps, color="#e63946", marker='o', linewidth=2.2, label="FPS")
    ax1.set_title("RiskMLP End-to-End Latency & Throughput", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Block Tree Count (BT)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Inference Latency (μs)", fontsize=10, fontweight='bold', color="#1d3557")
    ax1_twin.set_ylabel("Inference Throughput (FPS)", fontsize=10, fontweight='bold', color="#e63946")
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    ax2.plot([str(bt) for bt in sub["bt_count"]], sub["energy_uJ"], color="#2a9d8f", marker='s', linewidth=2.2, markersize=7)
    ax2.set_title("Total Energy per Inference Frame (μJ)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Block Tree Count (BT)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Energy per Frame (μJ)", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mma_risk_mlp_execution.png", dpi=250)
    plt.close()

def generate_cross_technology_comparison(all_tech_summaries, comparison_dir):
    """رسم نمودار مقایسه‌ای جامع میان تمام ۵ فناوری سیلیکونی"""
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    tech_keys = ["2nm", "7nm", "45nm", "130nm", "180nm"]
    rows = []
    
    for t in tech_keys:
        df = all_tech_summaries[t]
        grouped = df.groupby(["bt_count", "buf_size_B"]).agg({
            "latency_us": "sum",
            "energy_uJ": "sum",
            "power_mW": "mean",
            "gops_per_w": "mean",
            "tops_per_w": "mean",
            "area_mm2": "mean"
        }).reset_index()
        
        # انتخاب پیکربندی مرجع: BT=4, BUF=256B
        ref = grouped[(grouped["bt_count"] == 4) & (grouped["buf_size_B"] == 256)]
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
            "Energy_Efficiency_GOPS_W": round(r["gops_per_w"], 2),
            "Energy_Efficiency_TOPS_W": round(r["tops_per_w"], 5),
            "Approx_Area_mm2": round(r["area_mm2"], 6)
        })
        
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(comparison_dir / "tech_scaling_comparison.csv", index=False)
    
    # رسم نمودار جامع ۴ بخشی مقایسه فناوری‌ها
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(tech_keys))
    colors = ["#2b5c8f", "#3a86c8", "#48bfe3", "#f4a261", "#e76f51"]
    
    # ۱. مقایسه تاخیر استنتاج (Latency)
    axes[0, 0].bar(x, comp_df["EndToEnd_Latency_us"], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 0].set_title("End-to-End Latency Comparison (μs)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel("Latency (μs)", fontsize=10)
    axes[0, 0].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # ۲. مقایسه مصرف انرژی به ازای هر فریم (Energy)
    axes[0, 1].bar(x, comp_df["EndToEnd_Energy_uJ"], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title("Total Energy per Inference (μJ - Log Scale)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel("Energy (μJ)", fontsize=10)
    axes[0, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # ۳. کارایی انرژی (GOPS/W)
    axes[1, 0].bar(x, comp_df["Energy_Efficiency_GOPS_W"], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Energy Efficiency (GOPS/W - Log Scale)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("GOPS / Watt", fontsize=10)
    axes[1, 0].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # ۴. مساحت تقریبی سیلیکون (Silicon Area)
    axes[1, 1].bar(x, comp_df["Approx_Area_mm2"], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title("Silicon Area Scaling (mm² - Log Scale)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(tech_keys, fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel("Area (mm²)", fontsize=10)
    axes[1, 1].grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.suptitle("Cross-Technology Silicon Scaling Benchmark for MMA Accelerator (RiskMLP)", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(comparison_dir / "tech_scaling_comparison.png", dpi=250)
    plt.close()
    
    # ذخیره گزارش متنی مارک‌داون
    report_md = comparison_dir / "tech_scaling_summary.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Cross-Technology Analytical Evaluation Summary (MMA Accelerator)\n\n")
        f.write("Reference Hardware Configuration: Block Trees = 4, Buffer Size = 256 Bytes, Workload = RiskMLP (3 Layers)\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n")
    print(f"✔ مقایسه فناوری‌ها ذخیره شد: {comparison_dir}")

def main():
    print("=" * 80)
    print("🚀 آغاز فرآیند جامع شبیه‌سازی و DSE برای تمامی ۵ گره فناوری (2nm, 7nm, 45nm, 130nm, 180nm)")
    print("=" * 80)
    
    bt_list = [2, 4, 8, 16]
    buf_list = [128, 256, 512]
    
    temp_work_dir = REPO_DIR / "outputs" / "dse_temp"
    temp_work_dir.mkdir(parents=True, exist_ok=True)
    
    # مرحله ۱: اجرای شبیه‌سازی پایه با تیم‌لوپ برای یک لایه و کانفیگ‌ها
    total_scenarios = len(bt_list) * len(buf_list) * len(RISK_MLP_LAYERS)
    step = 0
    
    print(f"🔹 اجرای تحلیل پایه نگاشت‌ها با timeloop-mapper ({total_scenarios} سناریو)...")
    for bt in bt_list:
        for buf in buf_list:
            for l_name, l_spec in RISK_MLP_LAYERS.items():
                step += 1
                tag = f"BT{bt}_BUF{buf}_{l_name}"
                cfg_file = temp_work_dir / f"{tag}.yaml"
                log_file = temp_work_dir / f"{tag}.log"
                stats_backup = temp_work_dir / f"{tag}.stats.txt"
                
                generate_mapping_config(bt, buf, l_spec, cfg_file)
                
                # اجرای تیم‌لوپ
                env = os.environ.copy()
                env["TIMELOOP_PROBLEM_SHAPE_DIR"] = str(REPO_DIR / "inputs" / "workloads")
                env["LD_LIBRARY_PATH"] = str(REPO_DIR / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
                
                cmd = [str(TIMELOOP_MAPPER), str(cfg_file)]
                with open(log_file, "w") as out_f:
                    subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, env=env, cwd=REPO_DIR)
                    
                out_stats = REPO_DIR / "outputs" / "timeloop-mapper.stats.txt"
                if out_stats.exists():
                    shutil.copy(out_stats, stats_backup)
                    
                print(f"  [{step:2d}/{total_scenarios:2d}] سناریو {tag} ارزیابی شد.", flush=True)

    # مرحله ۲: استخراج نتایج و ترسیم برای تک‌تک فناوری‌ها
    all_tech_summaries = {}
    
    for tech_key, tech_info in TECH_CONFIGS.items():
        print(f"\n⚡ مشخصه‌یابی و تولید نتایج برای فناوری {tech_key} ({tech_info['name']})...")
        tech_out_dir = RIGHT_OUT_DIR / tech_key
        tech_out_dir.mkdir(parents=True, exist_ok=True)
        
        opt_records = []
        cand_records = []
        
        for bt in bt_list:
            for buf in buf_list:
                for l_name, l_spec in RISK_MLP_LAYERS.items():
                    tag = f"BT{bt}_BUF{buf}_{l_name}"
                    stats_file = temp_work_dir / f"{tag}.stats.txt"
                    log_file = temp_work_dir / f"{tag}.log"
                    
                    parsed = parse_stats_file(stats_file, tech_info, l_spec)
                    if parsed:
                        parsed["technology"] = tech_key
                        parsed["tech_name"] = tech_info["name"]
                        parsed["layer"] = l_name
                        parsed["bt_count"] = bt
                        parsed["buf_size_B"] = buf
                        opt_records.append(parsed)
                        
                        computes = parsed["computes"]
                        cands = parse_candidates(log_file, bt, buf, l_name, computes, tech_info)
                        cand_records.extend(cands)
                        
        df_opt = pd.DataFrame(opt_records)
        df_cands = pd.DataFrame(cand_records)
        
        df_opt.to_csv(tech_out_dir / "mma_dse_optimal_summary.csv", index=False)
        df_cands.to_csv(tech_out_dir / "mma_dse_all_evaluated_candidates.csv", index=False)
        all_tech_summaries[tech_key] = df_opt
        
        # کپی آخرین فایل‌های نمونه به پوشه فناوری
        for f_name in ["timeloop-mapper.stats.txt", "timeloop-mapper.map.txt", "timeloop-mapper.map+stats.xml"]:
            src_f = REPO_DIR / "outputs" / f_name
            if src_f.exists():
                shutil.copy(src_f, tech_out_dir / f_name)
                
        # تولید نمودارهای این فناوری
        generate_tech_plots(df_opt, df_cands, tech_key, tech_info, tech_out_dir)
        print(f"✔ نتایج ({len(df_opt)} بهینه و {len(df_cands)} کاندید) و نمودارهای {tech_key} در {tech_out_dir} مستقر شد.")
        
    # مرحله ۳: مقایسه جامع بین فناوری‌ها
    print("\n📊 تولید گزارش و نمودارهای مقایسه جامع چندفناوری...")
    generate_cross_technology_comparison(all_tech_summaries, RIGHT_OUT_DIR / "comparison")
    
    # پاک‌سازی فایل‌های موقت
    shutil.rmtree(temp_work_dir, ignore_errors=True)
    print("\n🎉 تمامی مراحل شبیه‌سازی چندفناوری و تولید نمودارها با موفقیت به پایان رسید.")

if __name__ == "__main__":
    main()
