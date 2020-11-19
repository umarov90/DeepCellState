import os
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
matplotlib.use("Agg")

data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)
#utils1.draw_vectors(encoder.predict(data.val_data), "latent_vectors/" + str(e) + "_1")
antibiotics = np.loadtxt("families/antibiotics", delimiter=",")
cholinergic = np.loadtxt("families/cholinergic", delimiter=",")
modulator = np.loadtxt("families/5-HT modulator", delimiter=",")
adrenergic = np.loadtxt("families/adrenergic", delimiter=",")
#antipsychotic = np.loadtxt("families/antipsychotic", delimiter=",")[:, 0:2]
#histaminergic = np.loadtxt("families/histaminergic", delimiter=",")[:, 0:2]
# sh = np.loadtxt("sh", delimiter=",")
# shc = np.loadtxt("sh.cgs", delimiter=",")
# cp = np.loadtxt("trt_cp", delimiter=",")
# lig = np.loadtxt("lig", delimiter=",")
# oe = np.loadtxt("oe", delimiter=",")

colors = []
colors.extend(["cholinergic"]*len(antibiotics))
colors.extend(["antibiotics"]*len(cholinergic))
colors.extend(["5-HT modulator"]*len(modulator))
colors.extend(["adrenergic"]*len(adrenergic))
# colors.extend(["sh"]*len(sh))
# colors.extend(["sh.cgs"]*len(shc))
# colors.extend(["cp"]*len(cp))
# colors.extend(["oe"]*len(oe))
# colors.extend(["lig"]*len(lig))
# colors.extend(["adrenergic"]*len(adrenergic))
# colors.extend(["antipsychotic"]*len(antipsychotic))
# colors.extend(["histaminergic"]*len(histaminergic))

data = np.vstack([antibiotics, cholinergic, modulator, adrenergic])
# pca = PCA(n_components=2)
# A = pca.fit_transform(data)
A = TSNE(n_components=2, perplexity=5, early_exaggeration=2).fit_transform(data)

sns.scatterplot(x=A[:, 0], y=A[:, 1], hue=colors)
plt.savefig("drugs.png")
plt.close(None)