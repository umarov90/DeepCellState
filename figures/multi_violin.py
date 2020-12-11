import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pandas as pd
import os
from scipy.stats import ttest_ind
from matplotlib.ticker import FormatStrFormatter
matplotlib.use("agg")

types = ["MoA validation", "Multiple cell types", "Unseen cell type", "shRNA for LoF"]

os.chdir(open("../data_dir").read().strip())
sns.set(font_scale=1.3, style='ticks')
fig, axs = plt.subplots(2,2,figsize=(12,8))
axs = axs.flat

df = pd.read_csv("figures_data/all_results_supp.csv", sep=",")
for n, ax in enumerate(axs):
    ax.text(-0.1, 1.05, string.ascii_lowercase[n], transform=ax.transAxes, size=20, weight='bold')
    df2 = df[(df['Validation'] == types[n])]
    df2 = df2[['Baseline', 'DeepCellState']]
    sns.violinplot(data=df2, ax=ax, palette="Set2")
    ax.set(ylabel='PCC')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_ticks_position('none')
    if n == 1:
        df_2cell = pd.read_csv("figures_data/2cell_all_results.tsv", sep="\t")
        t, p = ttest_ind(df2["DeepCellState"].to_list(), df_2cell["DeepCellState"].to_list())
    else:
        t, p = ttest_ind(df2["DeepCellState"].to_list(), df2["Baseline"].to_list())
    print(types[n] + ": " + str(p))
    print(str(np.mean(df2["Baseline"].to_numpy())) + " " + str(np.mean(df2["DeepCellState"].to_numpy())))
    print()

plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig("figures/multi.png")
