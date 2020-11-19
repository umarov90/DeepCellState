import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")
sns.set(font_scale=1.3, style='white')
fig, axs = plt.subplots(1,1,figsize=(6,4))
data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)
bdata = pickle.load(open("bdata.p", "rb"))
ddata = pickle.load(open("ddata.p", "rb"))
cdata = pickle.load(open("cdata.p", "rb"))
pert_ids = pickle.load(open("pert_ids.p", "rb"))
df = pd.DataFrame(list(zip(bdata, ddata, cdata)),
                  columns=['Baseline', 'DeepCellState', "DeepCellState*"], index=pert_ids)
#f, ax = plt.subplots(1, 1)
#df_bar = df.reset_index().melt(id_vars=["index"])
sns.boxplot(data=df, palette="Set2", ax=axs)
sns.stripplot(data=df, jitter=0.1, dodge=True, linewidth=2, size=8, palette="Set2", ax=axs)
axs.set(ylabel='Average PCC')
plt.title("Cancer drugs response prediction", loc='center', fontsize=18)
plt.tight_layout()
plt.savefig("ext_cancer.png")
plt.close(None)


