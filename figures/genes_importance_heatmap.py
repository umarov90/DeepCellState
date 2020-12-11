import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
matplotlib.use("Agg")


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


os.chdir(open("../data_dir").read().strip())
matplotlib.use("agg")
sns.set(font_scale=1.3, style='white')
df = pd.read_csv("figures_data/clustermap.csv", header=0, index_col=0)
special_genes = ["SPP1"]
mcf7_genes = ["SRC", "TGFBR2", "NET1", "FGFR2", "TBX2"]
pc3_genes = ["FKBP4", "BMP4"]
special_genes.extend(mcf7_genes)
special_genes.extend(pc3_genes)
a = df[df.columns[:50]]
# a.columns = [c if c in special_genes else "         " for c in a.columns]
y0 = .45


colormap = truncate_colormap(sns.color_palette("rocket", as_cmap=True), 0.0, 0.8)
cm = sns.clustermap(a, figsize=(8, 3), row_cluster=False, cmap=colormap,
                    tree_kws=dict(linewidths=0), cbar_pos=(0.05, y0, .03, .4), cbar_kws={},
                    rasterized=True, xticklabels=2)
cm.ax_heatmap.set_yticklabels(('MCF-7','PC-3'), rotation=0, fontsize="18", va="center")
for axis in ['top','bottom','left','right']:
    cm.cax.spines[axis].set_visible(True)
    cm.cax.spines[axis].set_color('black')
cm.cax.set_frame_on(True)
pos = cm.ax_heatmap.get_position()
pos.p0[0] = pos.p0[0] - 0.03
cm.ax_heatmap.set_position(pos)
cm.cax.set_xlabel('Score')
plt.setp(cm.ax_heatmap.get_xticklabels())#, rotation=45, ha="right", rotation_mode="anchor")
cm.ax_heatmap.tick_params(left=False, bottom=True)

for axis in [cm.cax.xaxis, cm.cax.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)
    axis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))

cm.fig.suptitle('Gene importance per decoder', fontsize=18)
plt.savefig("figures/heat.svg")

