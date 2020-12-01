import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")

os.chdir(open("../data_dir").read())
matplotlib.use("agg")
sns.set(font_scale=1.3, style='white')

df = pd.read_csv("figures_data/clustermap.csv", header=0, index_col=0)
a = df[df.columns[0:200]]
#plt.imshow(a, cmap='hot', interpolation='nearest')
sns.clustermap(a, figsize=(12, 5))
# sns.heatmap(df)
plt.tight_layout()
plt.savefig("figures/heat.png")