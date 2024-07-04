#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import LinearLocator
import PIL.Image as Image


# Function to adjust the brightness of a color
def dim_color(color, dim_factor=0.5):
    c = mcolors.colorConverter.to_rgb(color)
    return dim_factor * c[0], dim_factor * c[1], dim_factor * c[2]


# General plot settings
plt.rcParams.update({"font.size": 9})

# Palette
palette = sns.color_palette("Set1", 8)
color1  = palette[0]
color2  = palette[1]


# Make a plot area
fig1,  (ax1)  = plt.subplots(1, 1, figsize=(5, 5))
fig2,  (ax2)  = plt.subplots(1, 1, figsize=(5, 5))
fig3,  (ax3)  = plt.subplots(1, 1, figsize=(5, 5))
fig4,  (ax4)  = plt.subplots(1, 1, figsize=(5, 5))
'''
fig5,  (ax5)  = plt.subplots(1, 1, figsize=(5, 5))
fig6,  (ax6)  = plt.subplots(1, 1, figsize=(5, 5))
fig7,  (ax7)  = plt.subplots(1, 1, figsize=(5, 5))
fig8,  (ax8)  = plt.subplots(1, 1, figsize=(5, 5))
fig9,  (ax9)  = plt.subplots(1, 1, figsize=(5, 5))
fig10, (ax10) = plt.subplots(1, 1, figsize=(5, 5))
fig11, (ax11) = plt.subplots(1, 1, figsize=(5, 5))
fig12, (ax12) = plt.subplots(1, 1, figsize=(5, 5))
'''


# Graph 01
columns1 = ["gamma", "PSNR", "SSIM"]
data1    = pd.DataFrame.from_records(
    [
        (0.0, 17.643, 0.753),
        (0.2, 17.188, 0.750),
        (0.4, 17.140, 0.750),
        (0.6, 17.267, 0.753),
        (0.8, 17.076, 0.749),
        (1.0, 17.261, 0.753),
        (1.2, 17.431, 0.756),
        (1.4, 17.587, 0.759),
        (1.6, 17.668, 0.759),
        (1.8, 17.800, 0.760),
        (2.0, 18.186, 0.766),
        (2.2, 18.158, 0.765),
        (2.4, 18.272, 0.766),
        (2.6, 18.309, 0.768),
        (2.8, 18.184, 0.767),
        (3.0, 17.859, 0.763),
    ],
    columns=columns1
)

data1_pd = pd.melt(data1, id_vars=columns1[0], var_name="metric", value_name="value_numbers")
mask     = data1_pd.metric.isin(["SSIM"])
scale    = int(data1_pd[~mask].value_numbers.mean() / data1_pd[mask].value_numbers.mean())
data1_pd.loc[mask, "value_numbers"] = data1_pd.loc[mask, "value_numbers"] * scale

bar_plot = sns.barplot(x=columns1[0], y="value_numbers", hue="metric", data=data1_pd, ax=ax1, palette="Pastel1")
ax1.yaxis.set_major_locator(LinearLocator(numticks=4))
ax1.set_ylim(ymin=17., ymax=18.4)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_yticklabels(np.round(ax1.get_yticks(), 2))
ax1.yaxis.set_tick_params(rotation=90)
ax1.set_xlabel("$\gamma$ (for $\mathcal{L}_{bri}$)")
ax1.set_ylabel("PSNR$_{c}$", color=color1)
ax1r = ax1.twinx()
ax1r.yaxis.set_major_locator(LinearLocator(numticks=4))
ax1r.set_ylim(ax1.get_ylim())
# ax1r.set_ylim(ymin=0.74, ymax=0.78)
ax1r.tick_params(axis="y", labelcolor=color2)
ax1r.set_yticklabels(np.round(ax1.get_yticks() / scale, 2))
ax1r.yaxis.set_tick_params(rotation=90)
ax1r.set_ylabel("SSIM$_{c}$", color=color2)
ax1.get_legend().remove()

for bar, label in zip(bar_plot.patches, bar_plot.get_xticklabels() * 2):
    if label.get_text() == "2.6":
        print(bar.get_x())
        if bar.get_x() <= 12.600000000000001:
            bar.set_color(color1)
        else:
            bar.set_color(color2)
        # bar.set_facecolor(color1)
        # label.set_color("red")
        label.set_fontweight('bold')


# Graph 02
columns2 = ["radius", "epsilon", "PSNR", "SSIM"]
data2    = pd.DataFrame.from_records(
    [
        ("$r$=1", "$\epsilon$=1e-5", 18.151, 0.685),
        ("$r$=1", "$\epsilon$=1e-4", 18.151, 0.685),
        ("$r$=1", "$\epsilon$=1e-3", 18.151, 0.685),
        ("$r$=1", "$\epsilon$=1e-2", 18.151, 0.685),
        ("$r$=1", "$\epsilon$=1e-1", 17.951, 0.675),
        ("$r$=3", "$\epsilon$=1e-5", 18.246, 0.732),
        ("$r$=3", "$\epsilon$=1e-4", 18.323, 0.776),
        ("$r$=3", "$\epsilon$=1e-3", 18.245, 0.758),
        ("$r$=3", "$\epsilon$=1e-2", 18.151, 0.745),
        ("$r$=3", "$\epsilon$=1e-1", 17.950, 0.739),
        ("$r$=5", "$\epsilon$=1e-5", 18.265, 0.739),
        ("$r$=5", "$\epsilon$=1e-4", 18.309, 0.768),
        ("$r$=5", "$\epsilon$=1e-3", 18.170, 0.743),
        ("$r$=5", "$\epsilon$=1e-2", 17.789, 0.710),
        ("$r$=5", "$\epsilon$=1e-1", 17.895, 0.702),
        ("$r$=7", "$\epsilon$=1e-5", 18.272, 0.739),
        ("$r$=7", "$\epsilon$=1e-4", 18.320, 0.774),
        ("$r$=7", "$\epsilon$=1e-3", 18.091, 0.726),
        ("$r$=7", "$\epsilon$=1e-2", 17.786, 0.679),
        ("$r$=7", "$\epsilon$=1e-1", 17.687, 0.667),
        # (9, "1e-5", 18.275, 0.737),
        # (9, "1e-4", 18.312, 0.771),
        # (9, "1e-3", 17.900, 0.700),
        # (9, "1e-2", 17.900, 0.700),
        # (9, "1e-1", 17.900, 0.700),
    ],
    columns=columns2
)
data2_pd = (data2.pivot(index="radius", columns="epsilon", values="PSNR"))
data3_pd = (data2.pivot(index="radius", columns="epsilon", values="SSIM"))

sns.heatmap(data2_pd, annot=True, annot_kws={"size": 9.5}, fmt=".3f", linewidths=.5, ax=ax2, cbar=True, square=True, cmap="Reds")
ax2.add_patch(Rectangle((3, 1), 1, 1, fill=False, edgecolor="lime", lw=2))
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_title("PSNR$_{c}$")
ax2.tick_params(rotation=45)

for label in ax2.get_yticklabels():
    if label.get_text() == "$r$=3":
        label.set_weight("bold")
for label in ax2.get_xticklabels():
    if label.get_text() == "$\epsilon$=1e-4":
        label.set_weight("bold")

sns.heatmap(data3_pd, annot=True, annot_kws={"size": 9.5}, fmt=".3f", linewidths=.5, ax=ax3, cbar=True, square=True, cmap="Blues")
ax3.add_patch(Rectangle((3, 1), 1, 1, fill=False, edgecolor="lime", lw=2))
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_title("SSIM$_{c}$")
# ax3.yaxis.tick_right()
ax3.tick_params(rotation=45)

for label in ax3.get_yticklabels():
    if label.get_text() == "$r$=3":
        label.set_weight("bold")
for label in ax3.get_xticklabels():
    if label.get_text() == "$\epsilon$=1e-4":
        label.set_weight("bold")


# Graph 04
custom_colors  = ['#00008b', '#0000ff', '#00ffff', '#ffff00', '#ffa500', '#ff0000']
custom_palette = mcolors.LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

image_b     = Image.open("10_c_1_b.png")
image_b     = image_b.convert("L")  # 'L' mode is for grayscale
image_array = np.array(image_b) / 255
sns.heatmap(image_array, cmap=custom_palette, cbar=True, ax=ax4)
ax4.set_xlabel("")
ax4.set_ylabel("")
ax4.set_xticks([])
ax4.set_yticks([])


'''
# Graph 05
image_g     = Image.open("10_c_1_g.png")
image_g     = image_g.convert("L")  # 'L' mode is for grayscale
image_array = np.array(image_g) / 255
sns.heatmap(image_array, cmap=custom_palette, cbar=True, ax=ax5)
ax5.set_xlabel("")
ax5.set_ylabel("")
ax5.set_xticks([])
ax5.set_yticks([])


# Graph 06
image_r     = Image.open("10_c_1_r.png")
image_r     = image_r.convert("L")  # 'L' mode is for grayscale
image_array = np.array(image_r) / 255
sns.heatmap(image_array, cmap=custom_palette, cbar=True, ax=ax6)
ax6.set_xlabel("")
ax6.set_ylabel("")
ax6.set_xticks([])
ax6.set_yticks([])


# Graph 07
image     = Image.open("10_c_1.png")
image     = image.convert("L")  # 'L' mode is for grayscale
image_array = np.array(image) / 255
sns.heatmap(image_array, cmap=custom_palette, cbar=True, ax=ax7)
ax7.set_xlabel("")
ax7.set_ylabel("")
ax7.set_xticks([])
ax7.set_yticks([])


# Graph 08
image_      = Image.open("10_c_2.png")
image_      = image_.convert("L")  # 'L' mode is for grayscale
image_array = np.array(image_) / 255
sns.heatmap(image_array, cmap="viridis", cbar=True, ax=ax8)
ax8.set_xlabel("")
ax8.set_ylabel("")
ax8.set_xticks([])
ax8.set_yticks([])


# Graph 09
image_array = (np.array(image_) / 255) * (np.array(image_b) / 255)
sns.heatmap(image_array, cmap=custom_palette, cbar=True, ax=ax9)
ax9.set_xlabel("")
ax9.set_ylabel("")
ax9.set_xticks([])
ax9.set_yticks([])


# Graph 10
image_array = (np.array(image_) / 255) * (np.array(image_g) / 255)
sns.heatmap(image_array, cmap=custom_palette, cbar=True, ax=ax10)
ax10.set_xlabel("")
ax10.set_ylabel("")
ax10.set_xticks([])
ax10.set_yticks([])


# Graph 11
image_array = (np.array(image_) / 255) * (np.array(image_r) / 255)
sns.heatmap(image_array, cmap=custom_palette, cbar=True, ax=ax11)
ax11.set_xlabel("")
ax11.set_ylabel("")
ax11.set_xticks([])
ax11.set_yticks([])


# Graph 12
image_array = (np.array(image_) / 255) * (np.array(image) / 255)
sns.heatmap(image_array, cmap=custom_palette, cbar=True, ax=ax12)
ax12.set_xlabel("")
ax12.set_ylabel("")
ax12.set_xticks([])
ax12.set_yticks([])
'''


# Show
fig1.tight_layout()
# fig1.colorbar(ax2.collections[0], ax=ax2, location="left",  use_gridspec=False, pad=0.1)
# fig1.colorbar(ax3.collections[0], ax=ax3, location="right", use_gridspec=False, pad=0.1)
plt.show()
