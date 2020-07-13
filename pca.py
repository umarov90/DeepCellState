import os
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")

data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)
#antibiotics = np.loadtxt("families/antibiotics", delimiter=",")
#cholinergic = np.loadtxt("families/cholinergic", delimiter=",")[:, 0:2]
#modulator = np.loadtxt("families/5-HT modulator", delimiter=",")
#adrenergic = np.loadtxt("families/adrenergic", delimiter=",")
#antipsychotic = np.loadtxt("families/antipsychotic", delimiter=",")[:, 0:2]
#histaminergic = np.loadtxt("families/histaminergic", delimiter=",")[:, 0:2]
sh = np.loadtxt("sh", delimiter=",")
shc = np.loadtxt("sh.cgs", delimiter=",")
cp = np.loadtxt("trt_cp", delimiter=",")
lig = np.loadtxt("lig", delimiter=",")
oe = np.loadtxt("oe", delimiter=",")

colors = []
colors.extend(["sh"]*len(sh))
colors.extend(["sh.cgs"]*len(shc))
colors.extend(["cp"]*len(cp))
# colors.extend(["cholinergic"]*len(cholinergic))

colors.extend(["oe"]*len(oe))
colors.extend(["lig"]*len(lig))
# colors.extend(["adrenergic"]*len(adrenergic))
# colors.extend(["antipsychotic"]*len(antipsychotic))
# colors.extend(["histaminergic"]*len(histaminergic))

pca = PCA(n_components=2)
data = np.vstack([sh, shc, cp, oe, lig])
A = pca.fit_transform(data)

sns.scatterplot(x=A[:, 0], y=A[:, 1], hue=colors, alpha=0.1)
plt.savefig("trt_2.png")
plt.close(None)