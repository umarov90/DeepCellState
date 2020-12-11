import matplotlib
import os
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.chdir(open("../data_dir").read().strip())
sns.set(font_scale=1.3, style='ticks')
fig, axs = plt.subplots(1,1,figsize=(6,3))

df = pd.read_csv("figures_data/2cell_all_results.tsv", sep="\t")
df = df.replace("MCF7", "MCF-7", regex=True)
df = df.replace("PC3", "PC-3", regex=True)
df["fold_improvement"] = df['DeepCellState'] / df['Baseline']
ax = sns.histplot(df, x="fold_improvement", element="step", hue="Cell type", binwidth=0.25)
ax.set(xlim=(0, 7))
ax.set(xlabel='Fold change')

plt.title("DeepCellState improvement over baseline", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("figures/dist4.svg")
