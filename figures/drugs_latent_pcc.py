import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")
sns.set(font_scale=1.3, style='white')
wdir = "2cell_10fold/"
data_folder = "/home/user/data/DeepFake/" + wdir
os.chdir(data_folder)
df = pd.read_pickle("latent.p")
# mask
mask = np.triu(np.ones_like(df, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(6, 5.5))

# color map
# cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=1, as_cmap=True)
# plot heatmap
sns.heatmap(df, annot=False, fmt=".2f",
           linewidths=0, cmap=cmap, vmin=-1, vmax=1,
           cbar_kws={"shrink": .8}, square=True)
ax.tick_params(left=False, bottom=False)
# ticks
# yticks = [i.upper() for i in df.index]
# xticks = [i.upper() for i in df.columns]
plt.yticks(plt.yticks()[0], labels=[])
plt.xticks(plt.xticks()[0], labels=[])
plt.title("Latent vector correlation matrix", loc='center', fontsize=18)
plt.xlabel('MCF-7 profiles', fontsize=18)
plt.ylabel('PC-3 profiles', fontsize=18)
# Draw the heatmap with the mask and correct aspect ratio
plt.savefig("an1.png")
plt.tight_layout()
plt.close(None)