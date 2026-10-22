#!/usr/bin/env python3
# ==============================================================================
# اسکریپت رسم نمودارهای پیشرفته تحلیل فضای طراحی، توان، تاخیر و پرتو برای شتاب‌دهنده MMA
# Advanced Plotting Engine for MMA Design Space Exploration, Power, and Pareto Analysis
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

# تنظیم سبک ظاهری نمودارها
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_pareto(df_opt):
    """رسم منحنی پرتو (Pareto Frontier): تاخیر در برابر توان و انرژی"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    layers = df_opt["layer"].unique()
    markers = ['o', 's', '^', 'D', 'v']
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(df_opt["bt_count"].unique())))
    bt_color_map = {bt: c for bt, c in zip(sorted(df_opt["bt_count"].unique()), colors)}
    
    # نمودار ۱: تاخیر در برابر انرژی (Latency vs Energy)
    for idx, layer in enumerate(layers):
        sub = df_opt[df_opt["layer"] == layer].sort_values("latency_us")
        for bt in sorted(sub["bt_count"].unique()):
            bt_sub = sub[sub["bt_count"] == bt]
            ax1.scatter(bt_sub["latency_us"], bt_sub["energy_uJ"], 
                        label=f"{layer} (BT={bt})" if idx == 0 else "",
                        color=bt_color_map[bt], marker=markers[idx % len(markers)],
                        s=80, edgecolors='black', alpha=0.85)
        # خط ترند لایه
        ax1.plot(sub["latency_us"], sub["energy_uJ"], linestyle='--', alpha=0.4, color='gray')
        
    ax1.set_title("Pareto Analysis: Latency vs. Energy per Layer", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Latency (μs)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Total Energy (μJ)", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # نمودار ۲: تاخیر در برابر توان کل (Latency vs Power)
    for idx, layer in enumerate(layers):
        sub = df_opt[df_opt["layer"] == layer].sort_values("latency_us")
        for bt in sorted(sub["bt_count"].unique()):
            bt_sub = sub[sub["bt_count"] == bt]
            ax2.scatter(bt_sub["latency_us"], bt_sub["power_mW"],
                        label=f"BT={bt} (Col={bt})" if idx == 0 else "",
                        color=bt_color_map[bt], marker=markers[idx % len(markers)],
                        s=80, edgecolors='black', alpha=0.85)
            
    ax2.set_title("Latency vs. Total Power Dissipation", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Latency (μs)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Power (mW)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(title="Hardware Scales", fontsize=9, loc='upper right')
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "mma_dse_pareto.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✔ نمودار خطوط پرتو ذخیره شد: {out_path}")

def plot_search_space(df_cands, df_opt):
    """رسم پراکندگی کل فضای جستجو شامل تمام نقاط ساب‌اپتیمال و نقاط برتر اپتیمال"""
    if df_cands.empty:
        return
        
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # ترسیم ابر نقاط تمام کاندیداهای بررسی شده (Suboptimal Points)
    scatter = ax.scatter(df_cands["est_latency_us"], df_cands["energy_uJ"],
                         c=df_cands["utilization"], cmap="Spectral_r",
                         alpha=0.45, s=25, edgecolors='none', label="Evaluated Candidates (Suboptimal)")
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("PE Array Utilization (0.0 to 1.0)", fontsize=11, fontweight='bold')
    
    # ترسیم نقاط اپتیمال نهایی به صورت ستاره‌های درخشان
    ax.scatter(df_opt["latency_us"], df_opt["energy_uJ"],
               color='gold', s=160, marker='*', edgecolors='black', linewidths=1.2,
               label="Optimal Design Point (Timeloop Best)", zorder=10)
    
    ax.set_title("Full DSE Mapspace Exploration: Suboptimal vs. Optimal Candidates", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Estimated Latency (μs)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Total Energy (μJ)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "mma_dse_search_space.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✔ نمودار کل فضای جستجو (Suboptimal + Optimal) ذخیره شد: {out_path}")

def plot_power_breakdown(df_opt):
    """رسم نمودار ستونی تفکیک توان اجزای سخت‌افزار (MACs, Registers, SRAM, DRAM)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # فیلتر برای سایز بافر مرجع (۲۵۶ بایت)
    sub = df_opt[df_opt["buf_size_B"] == 256].copy()
    if sub.empty:
        sub = df_opt.copy()
        
    labels = [f"{row['layer'].split('_')[0]}\nBT={row['bt_count']}" for _, row in sub.iterrows()]
    x = np.arange(len(labels))
    width = 0.55
    
    p_mac = sub["power_mac_mW"].values
    p_reg = sub["power_reg_mW"].values
    p_sram = sub["power_sram_mW"].values
    p_dram = sub["power_dram_mW"].values
    
    ax.bar(x, p_mac, width, label="Compute MACs (INT8)", color="#2b5c8f")
    ax.bar(x, p_reg, width, bottom=p_mac, label="acc_reg (32-bit Accumulator)", color="#4682b4")
    ax.bar(x, p_sram, width, bottom=p_mac + p_reg, label="On-Chip Buffers (IA, W, OA)", color="#48d1cc")
    ax.bar(x, p_dram, width, bottom=p_mac + p_reg + p_sram, label="DRAM / SharedBus", color="#e76f51")
    
    ax.set_title("Power Breakdown across Hardware Modules (mW)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Workload Layer & Block Tree Count (BT)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Power Consumption (mW)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(title="Hardware Component", fontsize=9, loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "mma_dse_power_breakdown.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✔ نمودار تفکیک توان اجزا ذخیره شد: {out_path}")

def plot_end_to_end_risk_mlp(df_opt):
    """رسم کارایی کل مدل RiskMLP (مجموع ۳ لایه) برای کانفیگ‌های مختلف"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # تجمیع مقادیر در سطح کل شبکه عصبی (مجموع لایه‌های ۱ و ۲ و ۳)
    grouped = df_opt.groupby(["bt_count", "buf_size_B"]).agg({
        "latency_us": "sum",
        "energy_uJ": "sum",
        "power_mW": "mean",
        "tops_per_w": "mean"
    }).reset_index()
    
    bt_counts = sorted(grouped["bt_count"].unique())
    ref_buf = 256 if 256 in grouped["buf_size_B"].values else grouped["buf_size_B"].iloc[0]
    sub = grouped[grouped["buf_size_B"] == ref_buf].sort_values("bt_count")
    
    # نمودار ۱: تاخیر کل استنتاج و نرخ فریم بر ثانیه (FPS)
    latencies = sub["latency_us"].values
    fps = [1e6 / lat if lat > 0 else 0 for lat in latencies]
    
    ax1_twin = ax1.twinx()
    p1 = ax1.bar([str(bt) for bt in sub["bt_count"]], latencies, color="#1d3557", width=0.45, label="End-to-End Latency (μs)")
    p2 = ax1_twin.plot([str(bt) for bt in sub["bt_count"]], fps, color="#e63946", marker='o', linewidth=2.5, label="Throughput (FPS)")
    
    ax1.set_title("RiskMLP End-to-End Latency & Inferences/Sec", fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Number of Block Trees (BT Count)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Inference Latency (μs)", fontsize=11, fontweight='bold', color="#1d3557")
    ax1_twin.set_ylabel("Inference Throughput (FPS)", fontsize=11, fontweight='bold', color="#e63946")
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # نمودار ۲: مصرف کل انرژی به ازای هر فریم
    ax2.plot([str(bt) for bt in sub["bt_count"]], sub["energy_uJ"], color="#2a9d8f", marker='s', linewidth=2.5, markersize=8)
    ax2.set_title("Total Energy per Inference Frame (μJ)", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel("Number of Block Trees (BT Count)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Energy per Frame (μJ)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "mma_risk_mlp_execution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✔ نمودار استنتاج بلادرنگ RiskMLP ذخیره شد: {out_path}")

def main():
    opt_csv = RESULTS_DIR / "mma_dse_optimal_summary.csv"
    cands_csv = RESULTS_DIR / "mma_dse_all_evaluated_candidates.csv"
    
    if not opt_csv.exists():
        print(f"❌ فایل خلاصه نتایج {opt_csv} یافت نشد. لطفاً ابتدا run_mma_dse.py را اجرا کنید.")
        sys.exit(1)
        
    df_opt = pd.read_csv(opt_csv)
    df_cands = pd.read_csv(cands_csv) if cands_csv.exists() else pd.DataFrame()
    
    print("=" * 80)
    print("📊 شروع ترسیم نمودارهای تحلیلی و مقایسه‌ای فضای طراحی")
    print(f"🔹 تعداد رکوردهای بهینه: {len(df_opt)}")
    print(f"🔹 تعداد رکوردهای کاندید (ساب‌اپتیمال): {len(df_cands):,}")
    print("=" * 80)
    
    plot_pareto(df_opt)
    plot_search_space(df_cands, df_opt)
    plot_power_breakdown(df_opt)
    plot_end_to_end_risk_mlp(df_opt)
    
    print("\n✨ تمام ۴ نمودار با کیفیت بالا در مسیر results/ تولید شدند.")

if __name__ == "__main__":
    main()
