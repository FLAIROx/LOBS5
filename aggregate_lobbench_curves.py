"""Aggregate LOBbench sweep results and plot Muon vs AdamW curves.

Reads score pkl files from /projects/s5e/lob_pipeline/results_{muon,adamw}-s*/
and produces:
  - muon_vs_adamw_curves.csv  (step, optimizer, ks_mean, l1_mean, wass_mean)
  - muon_vs_adamw_lobbench_curves.png / .pdf

Must run with numpy 2.x environment:
  PYTHONPATH="/projects/s5e/lob_pipeline/pip_packages_extra:/projects/s5e/lob_pipeline/pip_packages"
  /projects/s5e/quant/miniforge3/envs/lobs5/bin/python aggregate_lobbench_curves.py
"""

import glob
import gzip
import os
import pickle
import re
import sys

import numpy as np

# ── Collect results ──────────────────────────────────────────────────────────

RESULTS_BASE = "/lus/lfs1aip2/projects/s5e/lob_pipeline"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

patterns = [
    os.path.join(RESULTS_BASE, "results_muon-s*/scores/scores_uncond_GOOG_*.pkl"),
    os.path.join(RESULTS_BASE, "results_adamw-s*/scores/scores_uncond_GOOG_*.pkl"),
]

rows = []
for pattern in patterns:
    for pkl_path in sorted(glob.glob(pattern)):
        # Skip shard fragments
        if "_shard" in pkl_path:
            continue

        # Extract optimizer and step from results directory name
        results_dir = pkl_path.split("/results_")[1].split("/")[0]
        match = re.match(r"(muon|adamw)-s(\d+)", results_dir)
        if not match:
            print(f"SKIP: can't parse {results_dir}")
            continue

        optimizer = match.group(1)
        step = int(match.group(2))

        # Load scores
        try:
            with gzip.open(pkl_path, "rb") as f:
                scores, _ = pickle.load(f)
        except Exception as e:
            print(f"ERROR loading {pkl_path}: {e}")
            continue

        # Compute mean across all features for each metric
        metrics = {}
        for metric in ["ks", "l1", "wasserstein"]:
            vals = []
            for feature in scores:
                if metric in scores[feature]:
                    val = scores[feature][metric]
                    # val is (point_estimate, ci_array, bootstrap_array)
                    vals.append(float(val[0]))
            if vals:
                metrics[metric] = np.mean(vals)

        if metrics:
            rows.append({
                "optimizer": optimizer,
                "step": step,
                "ks_mean": metrics.get("ks", np.nan),
                "l1_mean": metrics.get("l1", np.nan),
                "wass_mean": metrics.get("wasserstein", np.nan),
                "n_features": len(vals),
                "source": pkl_path,
            })
            print(f"  {optimizer:5s} step {step:6d}: KS={metrics.get('ks',0):.4f}  L1={metrics.get('l1',0):.4f}  Wass={metrics.get('wasserstein',0):.4f}")

if not rows:
    print("ERROR: No results found. Check that sweep jobs have completed.")
    sys.exit(1)

# ── Save CSV ─────────────────────────────────────────────────────────────────

csv_path = os.path.join(OUTPUT_DIR, "muon_vs_adamw_curves.csv")
with open(csv_path, "w") as f:
    f.write("optimizer,step,ks_mean,l1_mean,wass_mean,n_features\n")
    for r in sorted(rows, key=lambda x: (x["optimizer"], x["step"])):
        f.write(f"{r['optimizer']},{r['step']},{r['ks_mean']:.6f},{r['l1_mean']:.6f},{r['wass_mean']:.6f},{r['n_features']}\n")
print(f"\nCSV saved: {csv_path}")

# ── Plot ─────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

muon = sorted([r for r in rows if r["optimizer"] == "muon"], key=lambda x: x["step"])
adamw = sorted([r for r in rows if r["optimizer"] == "adamw"], key=lambda x: x["step"])

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
metrics_info = [
    ("ks_mean", "KS Distance (mean)", "KS"),
    ("l1_mean", "L1 Distance (mean)", "L1"),
    ("wass_mean", "Wasserstein Distance (mean)", "Wasserstein"),
]

for ax, (key, ylabel, title) in zip(axes, metrics_info):
    if muon:
        ax.plot([r["step"] for r in muon], [r[key] for r in muon],
                "o-", color="#2196F3", label="Muon", markersize=5, linewidth=1.5)
    if adamw:
        ax.plot([r["step"] for r in adamw], [r[key] for r in adamw],
                "s-", color="#FF9800", label="AdamW", markersize=5, linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

fig.suptitle("LOBbench Metrics: Muon vs AdamW (360M, GOOG Jan 2026, 1024 seq)", fontsize=13)
plt.tight_layout()

for ext in ["png", "pdf"]:
    path = os.path.join(OUTPUT_DIR, f"muon_vs_adamw_lobbench_curves.{ext}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {path}")

plt.close()
print("\nDone.")
