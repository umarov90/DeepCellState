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
f, ax = plt.subplots(figsize=(6, 5.5))

cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=1, as_cmap=True)
sns.heatmap(df, annot=False, fmt=".2f",
           linewidths=0, cmap=cmap, vmin=-1, vmax=1,
           cbar_kws={"shrink": .8}, square=True)
ax.tick_params(left=False, bottom=False)
plt.yticks(plt.yticks()[0], labels=[])
plt.xticks(plt.xticks()[0], labels=[])
plt.title("Latent vector correlation matrix", loc='center', fontsize=18)
plt.xlabel('MCF-7 profiles', fontsize=18)
plt.ylabel('PC-3 profiles', fontsize=18)
plt.savefig("figures/latent_pcc.png")
plt.tight_layout()
plt.close(None)