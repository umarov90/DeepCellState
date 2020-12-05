import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pandas as pd
import os
from scipy.stats import ttest_ind
from matplotlib.ticker import FormatStrFormatter
matplotlib.use("agg")

types = ["MoA validation", "Unseen cell type", "Multiple cell types", "shRNA for LoF"]

os.chdir(open("../data_dir").read())
sns.set(font_scale=1.3, style='white')
fig, axs = plt.subplots(2,2,figsize=(12,8))
axs = axs.flat

df = pd.read_csv("figures_data/supp_results.csv", sep=",")
for n, ax in enumerate(axs):
    ax.text(-0.1, 1.05, string.ascii_lowercase[n], transform=ax.transAxes, size=20, weight='bold')
    df2 = df[(df['Validation'] == types[n])]
    sns.violinplot(data=df2, ax=ax, palette="Set2")
    ax.set(ylabel='Average PCC')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    t, p = ttest_ind(df2["DeepCellState"].to_list(), df2["Baseline"].to_list())
    print(types[n] + ": " + str(p))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig("figures/multi.pdf")
