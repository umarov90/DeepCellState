import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")

data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)
#a = np.genfromtxt("cells_genes_heat.csv", delimiter=",")
df = pd.read_csv("clustermap.csv", header=0, index_col=0)
a = df[df.columns[0:20]]
#plt.imshow(a, cmap='hot', interpolation='nearest')
ax = sns.clustermap(a)
plt.savefig("heat.png")