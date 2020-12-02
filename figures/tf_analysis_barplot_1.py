import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.colors as mcolors
import os
import pandas as pd

plt.rcParams['hatch.linewidth'] = 2
matplotlib.use("agg")
os.chdir(open("../data_dir").read())
sns.set(font_scale=1.2, style='white')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 4))
df = pd.read_csv("figures_data/tf_analysis_data.csv")  # TF


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


col_vals_mcf7 = binomial_statistic(df['MCF7_num'])

max_p = max(col_vals_mcf7)
min_p = min(col_vals_mcf7)


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

df2 = df.drop(['MCF7_num'], 1)
df_long = pd.melt(df2, "TF", var_name="Targets", value_name="Fold Enrichment")

newwidth = 0.3
sns.barplot(x="TF", hue="Targets", y="Fold Enrichment", data=df_long, ax=ax)
ax.get_legend().remove()
# , palette=cm.gist_heat(col_vals_avg)
for i, bar in enumerate(ax.patches):
    if i < len(df["TF"]):  # Average
        bar.set_color(scalarmappaple.to_rgba(col_vals_mcf7[i]))
        # bar.set_edgecolor("white")
        bar.set_hatch("\\\\")
    bar.set_edgecolor("black")
    bar.set_linewidth("2")


cb = fig.colorbar(scalarmappaple, ax=ax)
cb.ax.invert_yaxis()
cb.ax.set_ylabel('P-value', labelpad=10)
plt.title("TF target genes overlap", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("figures/bar_tf_1.svg")

