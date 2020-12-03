import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")

os.chdir(open("../data_dir").read())
sns.set(font_scale=1.3, style='white')
df = pd.read_pickle("figures_data/latent.p")
f, ax = plt.subplots(figsize=(6, 4))

cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=1, as_cmap=True)
hm = sns.heatmap(df, annot=False, fmt=".2f",
           linewidths=0, cmap=cmap, vmin=-1, vmax=1, square=True)
cbar_ax = f.axes[-1]
for axis in ['top','bottom','left','right']:
    cbar_ax.spines[axis].set_visible(True)
    cbar_ax.spines[axis].set_color('black')
cbar_ax.set_frame_on(True)
ax.collections[0].colorbar.set_label("PCC")
ax.tick_params(left=False, bottom=False)
plt.yticks(plt.yticks()[0], labels=[])
plt.xticks(plt.xticks()[0], labels=[])
plt.title("Latent vector correlation matrix", loc='center', fontsize=18)
plt.xlabel('MCF-7 profiles', fontsize=18)
plt.ylabel('PC-3 profiles', fontsize=18)
plt.savefig("figures/latent_pcc.pdf")
plt.tight_layout()
plt.close(None)