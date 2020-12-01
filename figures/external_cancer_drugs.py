import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(open("../data_dir").read())
matplotlib.use("agg")
sns.set(font_scale=1.3, style='white')
fig, axs = plt.subplots(1,1,figsize=(7,4))

df = pd.read_csv("figures_data/cancer_drugs.tsv", sep="\t")

sns.boxplot(data=df, palette="Set2", ax=axs)
sns.stripplot(data=df, jitter=0.1, dodge=True, linewidth=2, size=8, palette="Set2", ax=axs)
axs.set(ylabel='Average PCC')
plt.title("Cancer drugs response prediction", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("figures/ext_cancer.png")
plt.close(None)


