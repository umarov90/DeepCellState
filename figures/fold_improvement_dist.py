import numpy as np
import matplotlib
import os
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pandas as pd

data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)
sns.set(font_scale=1.3, style='white')
fig, axs = plt.subplots(1,1,figsize=(6,4))

ax = axs

#for n, ax in enumerate(axs):
n = 0
df = pd.read_csv("figures_data/all_results.csv", sep=",", skipinitialspace=True)
df["fold_improvement"] = df['DeepCellState'] / df['Baseline']
ax = sns.displot(df, x="fold_improvement", element="step", hue="Cell type", binwidth=0.25)
ax.set(xlim=(0, 7))
ax.set(xlabel='Fold improvement')
# ax.set(ylabel='Average PCC')
# plt.ylabel('Average PCC', fontsize=18)
# sns.stripplot(data=df, ax=ax, jitter=0.05, palette="Set2", dodge=True, linewidth=1, size=5, alpha=0.2)
# # Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.subplots_adjust(hspace=0.3)
plt.title("MCF-7 and PC-3 response prediction", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("figures/dist.png")
