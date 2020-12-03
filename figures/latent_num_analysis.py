import numpy as np
import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

plt.style.use('seaborn')
data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)
sns.set(font_scale=1.3)
fig, ax = plt.subplots(figsize=(6, 4))

df = pd.read_csv("figures_data/latent_num_an.csv")

sns.lineplot(data=df, x="Latent nodes", y="PCC")
plt.title("Average PCC per latent node number", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("figures/latent_num.pdf")
