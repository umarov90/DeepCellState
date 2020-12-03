import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import LogNorm
matplotlib.use("Agg")


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


os.chdir(open("../data_dir").read())
matplotlib.use("agg")
sns.set(font_scale=1.3, style='white')
df = pd.read_csv("figures_data/clustermap.csv", header=0, index_col=0)
top20 = False
if top20:
    a = df[df.columns[-20:]]
    y0 = .42
else:
    a = df[df.columns[-120:-20]]
    y0 = .45


colormap = truncate_colormap(sns.color_palette("rocket", as_cmap=True), 0.1, 0.9)
cm = sns.clustermap(a, figsize=(8, 3), row_cluster=False, cmap=colormap,
                    tree_kws=dict(linewidths=0), cbar_pos=(0.05, y0, .03, .4), cbar_kws={},
                    rasterized=True)
cm.ax_heatmap.set_yticklabels(('MCF-7','PC-3'), rotation=0, fontsize="18", va="center")
for axis in ['top','bottom','left','right']:
    cm.cax.spines[axis].set_visible(True)
    cm.cax.spines[axis].set_color('black')
cm.cax.set_frame_on(True)
pos = cm.ax_heatmap.get_position()
pos.p0[0] = pos.p0[0] - 0.03
cm.ax_heatmap.set_position(pos)
cm.cax.set_xlabel('Score')
if top20:
    cm.fig.suptitle('Gene importance per decoder, top 20 genes', fontsize=18)
    plt.savefig("figures/heat20.svg")
else:
    cm.fig.suptitle('Gene importance per decoder, top 120 to 20 genes', fontsize=18)
    plt.savefig("figures/heat100.svg")
