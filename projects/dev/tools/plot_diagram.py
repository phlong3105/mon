#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

__all__ = []

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates

px = 1.0 / plt.rcParams["figure.dpi"]  # Conversion factor


# ==============================================================================
# region PLOT
# ==============================================================================

def plot_01(metrics: dict[str, float], scale: float = 4.0, font_size: int = 5):
    values = list(metrics.values())
    colors = [
        "#607D8B",
        "#90A4AE",
        "#CFD8DC",
        # "#1565C0",
        # "#2196F3",
        # "#64B5F6",
    ]

    # 1. Setup the figure
    fig, ax_bottom = plt.subplots(figsize=(33 * px * scale, 25 * px * scale))
    y_pos = np.arange(len(metrics))

    # 2. Bottom axis (For PSNR)
    ax_bottom.set_xlim(0, 25)
    ax_bottom.tick_params(axis="x", labelsize=font_size - 2)
    # ax_bottom.set_xlabel("PSNR Score", fontsize=font_size)

    # Set up the Left Vertical Axis (Y-axis)
    ax_bottom.set_yticks(y_pos)
    ax_bottom.set_yticklabels(metrics, fontsize=font_size)
    ax_bottom.invert_yaxis()  # Invert so PSNR is at the top, LPIPS at the bottom

    # 3. Top axis (For SSIM and LPIPS)
    ax_top = ax_bottom.twiny()
    ax_top.set_xlim(0.0, 1.0)
    ax_top.tick_params(axis="x", labelsize=font_size - 2)
    # ax_top.set_xlabel("SSIM / LPIPS Score", fontsize=font_size)

    # 4. Bars
    bar_height = 0.7
    # PSNR is plotted on the bottom axis scale (Darkest)
    bar_psnr  = ax_bottom.barh(y_pos[0], values[0], color=colors[0], edgecolor=colors[0], height=bar_height)
    # SSIM and LPIPS are plotted on the top axis scale (Lighter shades)
    bar_ssim  = ax_top.barh(y_pos[1], values[1], color=colors[1], edgecolor=colors[1], height=bar_height)
    bar_lpips = ax_top.barh(y_pos[2], values[2], color=colors[2], edgecolor=colors[2], height=bar_height)

    # 5. Add text labels to bars
    ax_bottom.bar_label(bar_psnr, fmt="%.2f", padding=1, color="#000000", fontsize=font_size)
    ax_top.bar_label(bar_ssim, fmt="%.3f", padding=1, color="#000000", fontsize=font_size)
    ax_top.bar_label(bar_lpips, fmt="%.3f", padding=1, color="#000000", fontsize=font_size)

    # 6. Styling
    # ax_bottom.grid(axis="x", linestyle="--", alpha=0.3)
    # ax_top.grid(axis="x", linestyle="--", alpha=0.3)

    # Remove harsh outer borders for an academic look
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    plt.tight_layout()
    ax_top.get_xaxis().set_visible(False)
    ax_bottom.get_xaxis().set_visible(False)

    # Save the plot
    plt.savefig("plot_01.png", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.show()


def plot_02():
    # 1. Define Data
    resolutions = ["128", "256", "512", "480p", "720p", "1080p", "2K", "4K", "8K"]
    vrams = {
        "CLODE":      [0.0614, 0.2316, 0.9239, 1.0803, 3.2331, 7.2712, 12.9241, -1.0, -1.0],
        "CoLIE":      [0.0391, 0.3125, 2.5000, 4.8828, -1.0, -1.0, -1.0, -1.0, -1.0],
        "PairLIE":    [0.0086, 0.0329, 0.1925, 0.2272, 0.6759, 1.5175, 2.6956, 6.0631, -1.0],
        "RetinexNet": [0.0516, 0.2053, 0.8198, 0.9630, 2.8804, 6.4725, 11.5022, -1.0, -1.0],
        "SCI++":      [0.0005, 0.0022, 0.0088, 0.0103, 0.0309, 0.0703, 0.1236, 0.2781, 1.1133],
        "Zero-DCE":   [0.0173, 0.0685, 0.2736, 0.3245, 0.9625, 2.1682, 3.8454, 8.6529, -1.0],
        "ZERO-IG":    [0.0121, 0.0483, 0.1935, 0.2295, 0.6827, 1.5299, 2.7193, 6.1209, -1.0],
        "SLICE":      [0.1209, 0.1758, 0.1821, 0.1823, 0.3526, 0.7781, 1.3732, 3.0710, 12.2508],
    }

    # VRAM in GB (Now on Left Axis, Lower is better)
    vram_slice = [1.5, 2.2, 3.5]
    vram_cnn = [4.0, 9.5, 24.5] # Hits OOM at 4K
    vram_patch = [2.5, 4.5, 8.0]

    # Composite Scores (Now on Right Axis, Higher is better)
    comp_slice = [1.45, 1.42, 1.38]
    comp_cnn = [1.38, 1.25, 0.95]
    comp_patch = [1.30, 1.15, 0.85]

    # 2. Setup Figure
    fig, ax1 = plt.subplots(figsize=(9, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'SLICE': '#5cb85c', 'CNN': '#d9534f', 'Patch': '#f0ad4e'}

    # 3. Plot VRAM (Left Axis - Solid Lines)
    ax1.set_xlabel('Inference Resolution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('VRAM Usage (GB) ↓', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, 28)

    l1 = ax1.plot(resolutions, vram_slice, color=colors['SLICE'], linestyle='-', marker='D', linewidth=3, markersize=8, label='SLICE (VRAM)')
    l2 = ax1.plot(resolutions, vram_cnn, color=colors['CNN'], linestyle='-', marker='s', linewidth=2.5, markersize=8, label='Heavy CNN (VRAM)')
    l3 = ax1.plot(resolutions, vram_patch, color=colors['Patch'], linestyle='-', marker='^', linewidth=2.5, markersize=8, label='Patch-Based (VRAM)')

    # Add OOM threshold explicitly to the left axis
    ax1.axhline(y=24.0, color='red', linestyle=':', linewidth=2, alpha=0.8)
    ax1.annotate('24GB Hardware Limit (OOM)', xy=(1, 24.5), color='red', fontsize=10, fontweight='bold', ha='center')

    # 4. Plot Composite Score (Right Axis - Dashed Lines)
    ax2 = ax1.twinx()
    ax2.set_ylabel('NTIRE Composite Score ↑', fontsize=12, fontweight='bold', color='#444444')
    ax2.tick_params(axis='y', labelcolor='#444444')
    ax2.set_ylim(0.5, 1.6)

    l4 = ax2.plot(resolutions, comp_slice, color=colors['SLICE'], linestyle='--', marker='o', linewidth=2.5, markersize=7, markerfacecolor='white', label='SLICE (Composite)')
    l5 = ax2.plot(resolutions, comp_cnn, color=colors['CNN'], linestyle='--', marker='o', linewidth=2.5, markersize=7, markerfacecolor='white', label='Heavy CNN (Composite)')
    l6 = ax2.plot(resolutions, comp_patch, color=colors['Patch'], linestyle='--', marker='o', linewidth=2.5, markersize=7, markerfacecolor='white', label='Patch-Based (Composite)')

    # 5. Combine Legends
    lines = l1 + l2 + l3 + l4 + l5 + l6
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5), frameon=True, shadow=True)

    plt.title('Memory Scaling vs. Composite Quality Across Resolutions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vram_left_composite_right.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

if __name__ == "__main__":
    # plot_01(metrics={"PSNR": 19.0478, "SSIM": 0.6982, "LPIPS": 0.5716})  # Zero-DCE
    # plot_01(metrics={"PSNR": 20.7204, "SSIM": 0.7147, "LPIPS": 0.5348})  # CLODE
    # plot_01(metrics={"PSNR": 17.5335, "SSIM": 0.8501, "LPIPS": 0.1795})  # CoLIE
    # plot_01(metrics={"PSNR": 21.2499, "SSIM": 0.8940, "LPIPS": 0.1573})  # SLICE

    plot_02()

# endregion
