import os
import pickle
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(open("../data_dir").read().strip())
matplotlib.use("agg")
sns.set(font_scale=1.3, style='ticks')
fig, axs = plt.subplots(1,1,figsize=(8,4))

df = pd.read_csv("figures_data/cancer_drugs.tsv", sep="\t")

sns.boxplot(data=df, palette="Set2", ax=axs)
sns.stripplot(data=df, jitter=0.1, dodge=True, linewidth=2, size=8, palette="Set2", ax=axs)
axs.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
axs.set(ylabel='Average PCC')
plt.title("Anticancer agents response prediction", loc='center', fontsize=18)
plt.tight_layout()
axs.xaxis.set_ticks_position('none')
plt.savefig("figures/ext_cancer.svg")


