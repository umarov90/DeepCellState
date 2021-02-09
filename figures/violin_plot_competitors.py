import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import ttest_ind

os.chdir(open("../data_dir").read().strip())
sns.set(font_scale=1.3, style='ticks')
fig, axs = plt.subplots(1,1,figsize=(16,4))

n = 0
df = pd.read_csv("data/competitors/all_results.csv", sep=",")
newpal = sns.color_palette("Set2")
a = newpal[1]
b = newpal[4]
newpal[1] = b
newpal[4] = a
sns.violinplot(data=df, ax=axs, palette=newpal)
axs.set(ylabel='PCC')
axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.title("Comparison with alternative approaches", loc='center', fontsize=18)
plt.tight_layout()
axs.xaxis.set_ticks_position('none')
plt.savefig("figures/violin_competitors.svg")
