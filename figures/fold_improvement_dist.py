import numpy as np
import matplotlib
import os
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pandas as pd

os.chdir(open("../data_dir").read())
sns.set(font_scale=1.3, style='white')
fig, axs = plt.subplots(1,1,figsize=(6,4))

n = 0
df = pd.read_csv("figures_data/2cell_all_results.tsv", sep="\t")
df["fold_improvement"] = df['DeepCellState'] / df['Baseline']
ax = sns.displot(df, x="fold_improvement", element="step", hue="Cell type", binwidth=0.25)
ax.set(xlim=(0, 7))
ax.set(xlabel='Fold improvement')

plt.title("MCF-7 and PC-3 response prediction", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("figures/dist.svg")
