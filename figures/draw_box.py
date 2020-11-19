import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pandas as pd

sns.set(font_scale=1.3, style='white')
fig, axs = plt.subplots(1,1,figsize=(6,4))
# plt.style.use('default')
# types = ["MoA validation", "Unseen cell type", "Multiple cell types", "shRNA for LoF"]
types = ["10-fold validation"]
# axs = axs.flat
ax = axs

#for n, ax in enumerate(axs):
n = 0
df = pd.read_csv("../all_results.csv", sep=",")
df = df[(df['Validation'] == types[n])]
# df_long = pd.melt(df, "Validation", var_name="Methods", value_name="PCC")
# plt.style.use('seaborn-whitegrid')
# sns.set(style="whitegrid")
sns.violinplot(data=df, ax=ax, palette="Set2") # x="Validation", hue="Methods", y="PCC", data=df_long, , fliersize=0
ax.set(ylabel='Average PCC')
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
plt.savefig("vio.svg")
