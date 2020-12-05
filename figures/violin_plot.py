import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import ttest_ind

os.chdir(open("../data_dir").read())
sns.set(font_scale=1.3, style='white')
fig, axs = plt.subplots(1,1,figsize=(6,4))

n = 0
df = pd.read_csv("figures_data/2cell_all_results.tsv", sep="\t")
df = df[['Baseline', 'DeepCellState']]
sns.violinplot(data=df, ax=axs, palette="Set2") # x="Validation", hue="Methods", y="PCC", data=df_long, , fliersize=0
t, p = ttest_ind(df["DeepCellState"].to_list()[:100], df["Baseline"].to_list()[:100])
print("p value: " + str(p))
axs.set(ylabel='Average PCC')
axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.title("MCF-7 and PC-3 response prediction", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("figures/violin.svg")
