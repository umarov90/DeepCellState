import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
matplotlib.use("Agg")

os.chdir(open("../data_dir").read())
matplotlib.use("agg")
sns.set(font_scale=1.3, style='white')

df = pd.read_csv("figures_data/clustermap.csv", header=0, index_col=0)
a = df[df.columns[0:200]]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


colormap = truncate_colormap(sns.color_palette("gist_heat_r", as_cmap=True), 0.2, 0.8)
sns.clustermap(a, figsize=(12, 5))
# sns.heatmap(df)
plt.tight_layout()
plt.savefig("figures/heat.png")