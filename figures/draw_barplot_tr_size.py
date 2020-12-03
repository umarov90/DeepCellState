import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

os.chdir(open("../data_dir").read())
matplotlib.use("agg")
sns.set(font_scale=1.3, style='white')
fig, axs = plt.subplots(1,1,figsize=(10,4))

df = pd.read_csv("figures_data/tr_size.tsv", sep="\t")

df_long = pd.melt(df, "Training set size", var_name="Methods", value_name="PCC")
sns.barplot(x="Training set size", hue="Methods", y="PCC", data=df_long,
    palette="Set2", saturation=0.5, linewidth=1,
            edgecolor="0.2", capsize=.15, errwidth=2, ax=axs, hue_order=["Baseline", "DeepCellState"])
axs.legend_.set_title(None)
plt.title("Average PCC per training set size", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("figures/bar.svg")
