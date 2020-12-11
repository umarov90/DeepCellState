import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib.ticker as ticker

plt.style.use('seaborn')
os.chdir(open("../data_dir").read().strip())
sns.set(font_scale=1.3)
fig, ax = plt.subplots(figsize=(6, 4))

df = pd.read_csv("figures_data/latent_num_an.csv")

ax = sns.lineplot(data=df, x="Latent nodes", y="PCC")
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

plt.title("Average PCC per latent node number", loc='center', fontsize=18)
plt.tight_layout()
fig.subplots_adjust(right=0.87)
plt.savefig("figures/latent_num1.svg")
