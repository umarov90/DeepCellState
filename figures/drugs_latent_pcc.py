import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")

wdir = "2cell_10fold/"
data_folder = "/home/user/data/DeepFake/" + wdir
os.chdir(data_folder)
df = pd.read_pickle("latent.p")
# mask
mask = np.triu(np.ones_like(df, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 8))

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
plt.yticks(plt.yticks()[0], labels=[], rotation=0)
plt.xticks(plt.xticks()[0], labels=[])
# title
# title = 'CORRELATION MATRIX'
# plt.title(title, loc='left', fontsize=18)

# Draw the heatmap with the mask and correct aspect ratio
plt.savefig("an1.png")
plt.close(None)