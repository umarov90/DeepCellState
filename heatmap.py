import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

data_folder = "/home/user/data/DeepFake/sub_all/"
os.chdir(data_folder)
#a = np.genfromtxt("cells_genes_heat.csv", delimiter=",")
a = pd.read_csv("clustermap.csv", header=0, index_col=0)
#plt.imshow(a, cmap='hot', interpolation='nearest')
ax = sns.clustermap(a)
plt.savefig("heat.png")