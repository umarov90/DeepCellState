import numpy as np
import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# plt.style.use('seaborn')
data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)
sns.set(font_scale=1.3, style='white')
fig, ax = plt.subplots(figsize=(6, 6))

df = pd.read_csv("figures_data/latent_num_an.csv")

sns.lineplot(data=df, x="Latent nodes", y="PCC")
# # Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#handles, labels = ax.get_legend_handles_labels()
# labels[1] = "Baseline"
# handles[1] = Line2D([0], [0], color='brown', linewidth=3, linestyle='dashed')
# When creating the legend, only use the first two elements
# to effectively remove the last two.
#l = plt.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc=1.5, borderaxespad=0.)
plt.tight_layout()
plt.savefig("figures/latent_num.pdf")
