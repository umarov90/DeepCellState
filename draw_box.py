import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# plt.style.use('seaborn')


fig, ax = plt.subplots(figsize=(17, 4))

df = pd.read_csv("all_results.csv", sep=",")

df_long = pd.melt(df, "Validation", var_name="Methods", value_name="PCC")
# plt.style.use('seaborn-whitegrid')
# sns.set(style="whitegrid")
sns.boxplot(x="Validation", hue="Methods", y="PCC", data=df_long, ax=ax, palette="Set2", fliersize=0)
sns.stripplot(x="Validation", hue="Methods", y="PCC", data=df_long, ax=ax,
              jitter=0.1, palette="Set2", dodge=True, linewidth=0, size=3, alpha=0.1, marker="D")
# # Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.savefig("box.png")
