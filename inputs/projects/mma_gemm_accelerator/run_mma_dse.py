#!/usr/bin/env python3
# ==============================================================================
# موتور کاوش فضای طراحی (DSE) و تخمین توان/انرژی/تاخیر برای شتاب‌دهنده MMA
# Design Space Exploration & Power/Latency Engine for Matrix Multiplication Accelerator
# ==============================================================================

import os
import sys
import re
import json
import yaml
import shutil
import argparse
import subprocess
import pandas as pd
from pathlib import Path

# تنظیم مسیرها
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
TECH_DIR = REPO_DIR / "TechFile" / "45nm"
RESULTS_DIR = SCRIPT_DIR / "results"

# لایه‌های مدل RiskMLP (از generate_risk_mlp_fused.py)
RISK_MLP_LAYERS = {
    "Layer1_FC1": {"M": 1, "K": 16, "N": 64, "name": "FC1 (16 -> 64)"},
    "Layer2_FC2": {"M": 1, "K": 64, "N": 64, "name": "FC2 (64 -> 64)"},
    "Layer3_FC3": {"M": 1, "K": 64, "N": 3,  "name": "FC3 (64 -> 3)"},
}

def to_posix_path(p):
    """تبدیل مسیر به فرمت استاندارد لینوکس / WSL"""
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        drive = s[0].lower()
        s = f"/mnt/{drive}" + s[2:]
    return s

def generate_config(bt_count, buf_size, layer_name, layer_spec, output_path):
    """تولید فایل پیکربندی یکپارچه بر اساس پارامترهای سخت‌افزاری و لایه"""
    num_macs = bt_count * 4  # هر بلوک‌تری دارای ۴ ضرب‌کننده در بعد K است
    spatial_n = min(bt_count, layer_spec["N"])
    
    config = {
        "mapper": {
            "algorithm": "hybrid",
            "optimization-metrics": ["energy", "delay"],
            "num-threads": 4,
            "timeout": 500,
            "victory-condition": 80,
            "search-size": 1500,
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
                    "entries": max(1, buf_size // 4),  # کلمات ۳۲ بیتی
                    "instances": bt_count,
                    "meshX": bt_count,
                    "word-bits": 32,
                    "read_bandwidth": 2,
                    "write_bandwidth": 2
                },
                {
                    "name": "W_BUF",
                    "entries": buf_size,               # کلمات ۸ بیتی
                    "instances": bt_count,
                    "meshX": bt_count,
                    "word-bits": 8,
                    "read_bandwidth": 4,
                    "write_bandwidth": 2
                },
                {
                    "name": "IA_BUF",
                    "entries": buf_size,               # کلمات ۸ بیتی مشترک
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
                    "read_bandwidth": 8,
                    "write_bandwidth": 8
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

def parse_optimal_stats(stats_path, freq_mhz):
    """استخراج آمارهای بهینه از فایل timeloop-mapper.stats.txt"""
    if not stats_path.exists():
        return None
        
    text = stats_path.read_text(encoding="utf-8", errors="ignore")
    
    # استخراج مقادیر کلیدی
    cycles_match = re.search(r'Cycles:\s*(\d+)', text)
    energy_match = re.search(r'Energy:\s*([0-9.]+)\s*uJ', text)
    util_match = re.search(r'Utilization:\s*([0-9.]+)', text)
    computes_match = re.search(r'Algorithmic Computes\s*=\s*(\d+)', text)
    
    cycles = int(cycles_match.group(1)) if cycles_match else 0
    energy_uj = float(energy_match.group(1)) if energy_match else 0.0
    utilization = float(util_match.group(1)) if util_match else 0.0
    computes = int(computes_match.group(1)) if computes_match else 0
    
    # استخراج سهم اجزای مختلف از انرژی (pJ/compute)
    components = {}
    pj_section = re.search(r'pJ/Algorithmic-Compute\s*\n(.*?)(?=\n\s*\n|\nActual Computes)', text, re.DOTALL)
    if pj_section:
        for line in pj_section.group(1).splitlines():
            line = line.strip()
            if "=" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    c_name = parts[0].strip()
                    try:
                        c_val = float(parts[1].strip())
                        components[c_name] = c_val
                    except ValueError:
                        pass
                
    # محاسبات زمان، توان و کارایی
    clock_period_ns = 1000.0 / freq_mhz
    latency_ns = cycles * clock_period_ns
    latency_us = latency_ns / 1000.0
    
    # توان کل به میلی‌وات: P(W) = E(uJ) / Latency(us) => P(mW) = P(W) * 1000
    power_mw = (energy_uj / latency_us) * 1000.0 if latency_us > 0 else 0.0
    
    # توان تفکیک‌شده اجزا بر حسب میلی‌وات
    mac_pj = components.get("MACs", 0.0)
    acc_pj = components.get("acc_reg", 0.0)
    oa_pj  = components.get("OA_BUF", 0.0)
    w_pj   = components.get("W_BUF", 0.0)
    ia_pj  = components.get("IA_BUF", 0.0)
    dram_pj= components.get("DRAM", 0.0)
    
    # P_comp(mW) = (pJ_comp * computes / latency_us) * 1e-3
    power_mac = (mac_pj * computes / latency_us) * 1e-3 if latency_us > 0 else 0.0
    power_reg = (acc_pj * computes / latency_us) * 1e-3 if latency_us > 0 else 0.0
    power_sram = ((oa_pj + w_pj + ia_pj) * computes / latency_us) * 1e-3 if latency_us > 0 else 0.0
    power_dram = (dram_pj * computes / latency_us) * 1e-3 if latency_us > 0 else 0.0
    
    # کارایی انرژی (TOPS/W)
    total_ops = 2.0 * computes
    tops_per_w = (total_ops / 1e12) / (energy_uj * 1e-6) if energy_uj > 0 else 0.0
    
    return {
        "cycles": cycles,
        "energy_uJ": round(energy_uj, 4),
        "latency_us": round(latency_us, 3),
        "power_mW": round(power_mw, 3),
        "power_mac_mW": round(power_mac, 3),
        "power_reg_mW": round(power_reg, 3),
        "power_sram_mW": round(power_sram, 3),
        "power_dram_mW": round(power_dram, 3),
        "utilization": utilization,
        "computes": computes,
        "tops_per_w": round(tops_per_w, 4),
        "components_pj": components
    }

def parse_all_suboptimal_candidates(log_path, bt_count, buf_size, layer_name, computes, freq_mhz):
    """استخراج تمام نقاط کاندید ارزیابی شده (اعم از Optimal و Suboptimal) از لاگ تیم‌لوپ"""
    candidates = []
    if not log_path.exists():
        return candidates
        
    clock_period_ns = 1000.0 / freq_mhz
    num_macs = bt_count * 4
    
    pattern = re.compile(r'\[\s*\d+\]\s+Utilization\s*=\s*([0-9.]+)\s*\|\s*pJ/Algorithmic-Compute\s*=\s*([0-9.]+)\s*\|\s*pJ/Compute\s*=\s*([0-9.]+)\s*\|\s*(.+)')
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            m = pattern.search(line)
            if m:
                util = float(m.group(1))
                pj_algo = float(m.group(2))
                pj_comp = float(m.group(3))
                loop_nest = m.group(4).strip()
                
                # محاسبه انرژی کل (uJ)
                energy_uj = (pj_algo * computes) / 1e6
                # تخمین چرخه بر اساس بهره‌وری موازی
                est_cycles = max(1, int(computes / (max(0.01, util) * num_macs)))
                est_latency_us = (est_cycles * clock_period_ns) / 1000.0
                est_power_mw = (energy_uj / est_latency_us) * 1000.0 if est_latency_us > 0 else 0.0
                edp = energy_uj * est_latency_us
                
                candidates.append({
                    "layer": layer_name,
                    "bt_count": bt_count,
                    "num_macs": num_macs,
                    "buf_size_B": buf_size,
                    "candidate_idx": idx,
                    "utilization": util,
                    "pj_per_compute": pj_comp,
                    "energy_uJ": round(energy_uj, 4),
                    "est_cycles": est_cycles,
                    "est_latency_us": round(est_latency_us, 3),
                    "est_power_mW": round(est_power_mw, 3),
                    "edp_uJ_us": round(edp, 4),
                    "loop_nest": loop_nest
                })
                
    return candidates

def run_dse(bt_list, buf_list, layers_dict, freq_mhz, quick_mode=False):
    """اجرای حلقه کامل کاوش فضای طراحی (DSE)"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_run_dir = RESULTS_DIR / "temp_run"
    temp_run_dir.mkdir(parents=True, exist_ok=True)
    
    optimal_summary = []
    all_evaluated_candidates = []
    
    total_runs = len(bt_list) * len(buf_list) * len(layers_dict)
    current_run = 0
    
    print("=" * 80)
    print(f"🚀 شروع کاوش فضای طراحی (DSE) شتاب‌دهنده MMA")
    print(f"🔹 تعداد بلوک‌تری‌ها (BT Counts): {bt_list}")
    print(f"🔹 سایز بافرهای پینگ‌پنگ (Bytes): {buf_list}")
    print(f"🔹 لایه‌های شبکه RiskMLP: {list(layers_dict.keys())}")
    print(f"🔹 فرکانس کلاک کاری: {freq_mhz} MHz (Tclk = {1000.0/freq_mhz:.2f} ns)")
    print(f"🔹 تکنولوژی ساخت: 45nm Nangate Open Cell")
    print(f"🔹 کل سناریوهای بهینه‌سازی: {total_runs}")
    print("=" * 80)
    
    is_linux = sys.platform.startswith("linux")
    
    for bt in bt_list:
        for buf in buf_list:
            for l_name, l_spec in layers_dict.items():
                current_run += 1
                run_tag = f"BT{bt}_BUF{buf}B_{l_name}"
                cfg_path = temp_run_dir / f"{run_tag}.yaml"
                log_path = temp_run_dir / f"{run_tag}.log"
                
                print(f"[{current_run:2d}/{total_runs:2d}] اجرای سناریو: {run_tag} ...", end="", flush=True)
                
                # ۱. ساخت فایل کانفیگ
                generate_config(bt, buf, l_name, l_spec, cfg_path)
                
                # ۲. اجرای timeloop-mapper
                if is_linux:
                    cmd = f"source {to_posix_path(REPO_DIR)}/env/setup-env.bash && cd {to_posix_path(temp_run_dir)} && timeloop-mapper {to_posix_path(cfg_path)} > {to_posix_path(log_path)} 2>&1"
                    res = subprocess.run(["bash", "-c", cmd], capture_output=True)
                else:
                    wsl_cfg = to_posix_path(cfg_path)
                    wsl_log = to_posix_path(log_path)
                    wsl_dir = to_posix_path(temp_run_dir)
                    cmd = f"source {to_posix_path(REPO_DIR)}/env/setup-env.bash && cd {wsl_dir} && timeloop-mapper {wsl_cfg} > {wsl_log} 2>&1"
                    res = subprocess.run(["wsl", "-e", "bash", "-c", cmd], capture_output=True)
                
                stats_path = temp_run_dir / "timeloop-mapper.stats.txt"
                opt_data = parse_optimal_stats(stats_path, freq_mhz)
                
                if opt_data:
                    record = {
                        "scenario": run_tag,
                        "layer": l_name,
                        "bt_count": bt,
                        "num_macs": bt * 4,
                        "buf_size_B": buf,
                        "freq_MHz": freq_mhz,
                        **opt_data
                    }
                    record.pop("components_pj", None)
                    optimal_summary.append(record)
                    print(f" ✔ موفق | تاخیر: {opt_data['latency_us']:.2f} us | توان: {opt_data['power_mW']:.2f} mW | انرژی: {opt_data['energy_uJ']:.4f} uJ")
                else:
                    print(" ❌ خطا در استخراج آمار بهینه!")
                    
                # ۳. استخراج تمام نقاط کاندید (Suboptimal + Optimal)
                cand_list = parse_all_suboptimal_candidates(log_path, bt, buf, l_name, l_spec["K"] * l_spec["N"], freq_mhz)
                all_evaluated_candidates.extend(cand_list)
                
                # پاک‌سازی فایل‌های موقت پوشه اجرا جهت جلوگیری از تداخل نام‌ها
                for tmp_f in ["timeloop-mapper.stats.txt", "timeloop-mapper.map.txt", "timeloop-mapper.map+stats.xml"]:
                    p = temp_run_dir / tmp_f
                    if p.exists():
                        p.unlink()

    # ۴. ذخیره خروجی‌ها در فایل‌های CSV و JSON
    df_opt = pd.DataFrame(optimal_summary)
    opt_csv = RESULTS_DIR / "mma_dse_optimal_summary.csv"
    df_opt.to_csv(opt_csv, index=False, encoding="utf-8")
    
    opt_json = RESULTS_DIR / "mma_dse_optimal_summary.json"
    with open(opt_json, "w", encoding="utf-8") as f:
        json.dump(optimal_summary, f, indent=2, ensure_ascii=False)
        
    df_cands = pd.DataFrame(all_evaluated_candidates)
    cands_csv = RESULTS_DIR / "mma_dse_all_evaluated_candidates.csv"
    df_cands.to_csv(cands_csv, index=False, encoding="utf-8")
    
    print("\n" + "=" * 80)
    print("✨ کاوش فضای طراحی با موفقیت به پایان رسید!")
    print(f"📊 تعداد سناریوهای بهینه ثبت شده: {len(df_opt)}")
    print(f"🔍 تعداد کل نقاط ارزیابی‌شده در فضای جستجو (Optimal + Suboptimal): {len(df_cands):,}")
    print(f"📁 فایل خلاصه نقاط بهینه: {opt_csv}")
    print(f"📁 فایل کلیه کاندیداهای فضا: {cands_csv}")
    print("=" * 80)
    
    shutil.rmtree(temp_run_dir, ignore_errors=True)
    return opt_csv, cands_csv

def main():
    parser = argparse.ArgumentParser(description="DSE Engine for Custom MMA Accelerator on 45nm")
    parser.add_argument("--freq", type=float, default=500.0, help="Clock frequency in MHz (default: 500 MHz)")
    parser.add_argument("--bts", nargs="+", type=int, default=[2, 4, 8, 16, 32], help="List of Block Tree counts (columns)")
    parser.add_argument("--bufs", nargs="+", type=int, default=[128, 256, 512, 1024], help="List of buffer sizes in bytes")
    parser.add_argument("--quick", action="store_true", help="Run minimal quick sweep for testing")
    args = parser.parse_args()
    
    bts = [4, 8] if args.quick else args.bts
    bufs = [256, 512] if args.quick else args.bufs
    
    run_dse(bts, bufs, RISK_MLP_LAYERS, args.freq, args.quick)

if __name__ == "__main__":
    main()
