import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

os.chdir(open("../data_dir").read().strip())
sns.set(font_scale=1.3, style='ticks')
fig, axs = plt.subplots(1,1,figsize=(8,5))

n = 0
df = pd.read_csv("figures_data/for_violin_plot_our.csv", sep=",")
# df = pd.read_csv("figures_data/for_violin_plot_their.csv", sep=",")
sns.violinplot(data=df, ax=axs)
axs.set(ylabel='PCC')
axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.xticks(rotation=30)
plt.xticks(fontsize=12)
plt.title("Comparison with DNPP and FaLRTC", loc='center', fontsize=18)
plt.tight_layout()
axs.xaxis.set_ticks_position('none')
plt.savefig("figures/S4_Fig.svg")
# plt.savefig("figures/S5_Fig.svg")
