import math

import numpy as np
import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import matplotlib.colors as mcolors
import os
import pandas as pd

# plt.style.use('seaborn')
data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)

sns.set(font_scale=1.3, style='white')
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
plt.subplots_adjust(hspace=0.3)
df = pd.read_csv("tf_analysis_data.csv") #TF

K = 50
num_important_genes = 50
col_vals_avg = []
for i in df['Average_num']:
    pc = num_important_genes / 978
    p_val = 0
    for k in range(int(i), K+1):
        p_val += (pc ** k) * ((1 - pc) ** (K - k))
    col_vals_avg.append(-math.log(p_val))

col_vals_mcf7 = []
for i in df['MCF7_num']:
    pc = num_important_genes / 978
    p_val = 0
    for k in range(int(i), K+1):
        p_val += (pc ** k) * ((1 - pc) ** (K - k))
    col_vals_mcf7.append(-math.log(p_val))

max_p = max(col_vals_avg + col_vals_mcf7)
min_p = min(col_vals_avg + col_vals_mcf7)
normalize = mcolors.Normalize(vmin=min_p, vmax=max_p)
colormap = cm.gist_heat
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)

df2 = df.drop(['Average_num', 'MCF7_num'], 1)
df_long = pd.melt(df2, "TF", var_name="Targets", value_name="Fold Enrichment")

col_vals_avg = [(float(i)-min_p)/(max_p-min_p) for i in col_vals_avg]
sns.barplot(x="TF", hue="Targets", y="Fold Enrichment", data=df_long, ax=axs[0]) #, palette=cm.gist_heat(col_vals_avg)
# axs[0].set(ylabel='Fold Enrichment', xlabel="")
# axs[0].set_title('Average')
# axs[0].set(ylim=(0, 3))

# col_vals_mcf7 = [(float(i)-min_p)/(max_p-min_p) for i in col_vals_mcf7]
# sns.barplot(x=df.index, y=df['MCF7_fold'], ax=axs[1], palette=cm.gist_heat(col_vals_mcf7))
# axs[1].set(ylabel='Fold Enrichment', xlabel="")
# axs[1].set_title('MCF7')
# axs[1].set(ylim=(0, 3))

fig.colorbar(scalarmappaple, ax=axs.ravel().tolist())
plt.savefig("bar_tf.pdf")
