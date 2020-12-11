import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(open("../data_dir").read().strip())
matplotlib.use("agg")
sns.set(font_scale=1.3, style='ticks')
fig, axs = plt.subplots(1,1,figsize=(8,4))

df = pd.read_csv("figures_data/4statins.tsv", sep="\t")
df_long = pd.melt(df, "Statin", var_name="Methods", value_name="PCC")
sns.barplot(x="Statin", hue="Methods", y="PCC", data=df_long, palette="Set2", ax=axs)
axs.legend_.set_title(None)
axs.set(ylabel='Average PCC')
axs.set(xlabel='')
plt.title("Statins response prediction", loc='center', fontsize=18)
plt.tight_layout()
axs.xaxis.set_ticks_position('none')
plt.savefig("figures/ext_statins.svg")


