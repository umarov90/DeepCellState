import math

import numpy as np
import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.colors as mcolors
import os
import pandas as pd

plt.rcParams['hatch.linewidth'] = 2
# plt.style.use('seaborn')
data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)

sns.set(font_scale=1.3, style='white')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
df = pd.read_csv("tf_analysis_data.csv")  # TF


def binomial_statistic(overlap_num):
    K = 50
    num_important_genes = 50
    p_vals = []
    for i in overlap_num:
        pc = num_important_genes / 978
        p_val = 0
        if pc * K < i:
            for k in range(int(i), K + 1):
                p_val += (pc ** k) * ((1 - pc) ** (K - k))
        else:
            for k in range(0, int(i) + 1):
                p_val += (pc ** k) * ((1 - pc) ** (K - k))
        p_vals.append(p_val)
    return p_vals


col_vals_avg = binomial_statistic(df['Average_num'])
col_vals_mcf7 = binomial_statistic(df['MCF7_num'])

max_p = max(col_vals_avg + col_vals_mcf7)
min_p = min(col_vals_avg + col_vals_mcf7)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


colormap = truncate_colormap(sns.color_palette("gist_heat_r", as_cmap=True), 0.2, 0.8)
# colormap = sns.color_palette("gist_heat_r", as_cmap=True)
scalarmappaple = cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=min_p, vmax=max_p), cmap=colormap)

df2 = df.drop(['Average_num', 'MCF7_num'], 1)
df_long = pd.melt(df2, "TF", var_name="Targets", value_name="Fold Enrichment")

newwidth = 0.3
sns.barplot(x="TF", hue="Targets", y="Fold Enrichment", data=df_long, ax=ax)
ax.get_legend().remove()
# , palette=cm.gist_heat(col_vals_avg)
for i, bar in enumerate(ax.patches):
    if i < 13:  # Average
        bar.set_color(scalarmappaple.to_rgba(col_vals_avg[i]))
        # bar.set_edgecolor("white")
        bar.set_hatch("\\\\")
    else:  # MCF7
        index = i - 13
        bar.set_color(scalarmappaple.to_rgba(col_vals_mcf7[index]))
        bar.set_hatch("xx")
    bar.set_edgecolor("black")
    bar.set_linewidth("2")

custom_lines = [mpatches.Patch(hatch="\\\\", edgecolor="black", facecolor="#fed8b1"),
                mpatches.Patch(hatch="xx", edgecolor="black", facecolor="#fed8b1")]
plt.legend(handles=custom_lines, title='', loc='upper right', labels=['Average', 'MCF-7'])
cb = fig.colorbar(scalarmappaple, ax=ax)
cb.ax.invert_yaxis()
cb.ax.set_ylabel('P-value', labelpad=10)
plt.title("TF target genes overlap", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("figures/bar_tf.png")

