#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

__all__ = []

import matplotlib.pyplot as plt
import numpy as np

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
    # 1. Define the x-axis resolutions
    resolutions = np.array([256, 512, 1024, 2048, 4096])

    # 2. Define the data for each method
    # Format: 'Method Name': (PSNR_array, Params_in_Millions, Marker, Color)
    methods_data = {
        'CoLIE': (np.array([24.5, 24.2, 23.8, 23.1, 22.5]), 12.5, 'o', '#9CBBA4'),
        'CLODE': (np.array([26.1, 25.8, 25.4, 24.9, 24.1]), 18.2, '^', '#A3B899'),
        # Highlight the main method in bold orange with a star marker
        # 'CALIE': (np.array([26.5, 27.2, 28.1, 28.8, 29.5]), 24.5, '*', '#ECA172')
    }

    # Initialize the plot with a specific size
    fig, ax = plt.subplots(figsize=(8, 6))

    # 3. Plot each method
    for name, (psnr, params, marker, color) in methods_data.items():
        # Scale the parameter count to control the physical area of the scatter points.
        # Adjust this multiplier (e.g., 15) to make the overall dots larger or smaller.
        point_size = params * 15

        # Draw the connecting line between resolutions
        ax.plot(resolutions, psnr, color=color, linewidth=2.5, alpha=0.7, zorder=1)

        # Draw the scatter points
        ax.scatter(
            resolutions, psnr,
            s=point_size,
            marker=marker,
            color=color,
            label=f"{name} ({params}M)",
            zorder=2
        )

        # Add the text label slightly above the final 4K point
        ax.text(
            resolutions[-1], psnr[-1] + 0.3, name,
            color=color,
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='bottom'
        )

    # 4. Apply the clean, academic aesthetics from your reference image
    # Hide the top and right bounding box lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Color the left and bottom axes gray
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')

    # Add the light dashed grid background
    ax.grid(True, linestyle='--', color='#EAEAEA', linewidth=1, zorder=0)

    # 5. Format the X-axis for image resolutions
    ax.set_xscale('log', base=2)
    ax.set_xticks(resolutions)
    ax.set_xticklabels(['256', '512', '1K', '2K', '4K'], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Set the axis labels
    ax.set_xlabel("Image Resolution", fontsize=18, labelpad=12)
    ax.set_ylabel("PSNR Performance (dB)", fontsize=18, labelpad=12)

    # Set the limits slightly wider than the data so markers aren't cut off
    ax.set_xlim(200, 5000)
    ax.set_ylim(21, 31)

    # Add a clean legend
    ax.legend(loc='lower left', frameon=False, fontsize=12, labelspacing=1.2)

    plt.tight_layout()
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
