#!/usr/bin/env python3
"""
Generates publication-grade latency CDFs, stage breakdowns, and closed-loop
tracking RMSE plots from Aegis-RT CSV flight telemetry logs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    cols: Dict[str, List[float]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v, dtype=np.float64) for k, v in cols.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Pairs of Label=path/to/trace.csv")
    parser.add_argument("--out", type=Path, default=Path("results/aegis_benchmark_report.png"))
    args = parser.parse_args()

    runs: Dict[str, Dict[str, np.ndarray]] = {}
    for item in args.inputs:
        if "=" in item:
            label, p = item.split("=", 1)
        else:
            p = item
            label = Path(item).stem
        runs[label] = load_csv(Path(p))

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=160)

    # Panel 1: Stage Latency Breakdown (p50 & p99 markers)
    ax0 = axes[0]
    labels = list(runs.keys())
    x = np.arange(len(labels))
    pre_p50 = [np.percentile(runs[k]["preprocess_us"], 50) / 1000.0 for k in labels]
    inf_p50 = [np.percentile(runs[k]["inference_us"], 50) / 1000.0 for k in labels]
    e2e_p99 = [np.percentile(runs[k]["total_e2e_us"], 99) / 1000.0 for k in labels]

    ax0.bar(x, pre_p50, width=0.48, label="Preprocess p50 (ms)", color="#3b82f6")
    ax0.bar(x, inf_p50, width=0.48, bottom=pre_p50, label="Inference p50 (ms)", color="#10b981")
    ax0.scatter(x, e2e_p99, color="#ef4444", zorder=5, s=55, label="Total E2E p99 (ms)")
    ax0.axhline(16.667, color="#dc2626", linestyle="--", linewidth=1.2, label="60 Hz Deadline (16.7 ms)")
    ax0.set_ylim(0.0, 20.5)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=15, ha="right")
    ax0.set_ylabel("Latency (ms)")
    ax0.set_title("Perception Pipeline Latency Breakdown")
    ax0.legend(loc="upper right", fontsize=8.5)

    # Panel 2: End-to-End Tail Latency CDF
    ax1 = axes[1]
    for label, data in runs.items():
        lat_ms = np.sort(data["total_e2e_us"] / 1000.0)
        cdf = np.linspace(0.0, 1.0, len(lat_ms))
        ax1.plot(lat_ms, cdf, linewidth=2.0, label=f"{label} (p99={np.percentile(lat_ms, 99):.2f}ms)")
    ax1.axvline(16.667, color="#dc2626", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("End-to-End Frame Latency (ms)")
    ax1.set_ylabel("Cumulative Probability (CDF)")
    ax1.set_title("Real-Time Determinism & Tail Jitter CDF")
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel 3: Closed-Loop 3D Target State Estimation Error over Time
    ax2 = axes[2]
    for label, data in runs.items():
        rmse = np.sqrt(np.mean(data["tracking_error_m"] ** 2))
        ax2.plot(data["sim_time_s"], data["tracking_error_m"], linewidth=1.6, alpha=0.9,
                 label=f"{label} (RMSE={rmse:.3f}m)")
    ax2.set_xlabel("Flight Time (s)")
    ax2.set_ylabel("3D Target Tracking Error (m)")
    ax2.set_title("Closed-Loop 6-DOF Pursuit Tracking Accuracy")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("Aegis-RT: Deterministic Edge Perception & Closed-Loop Quadrotor Autonomy",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[Plot] Saved benchmark figure -> {args.out}")


if __name__ == "__main__":
    main()
