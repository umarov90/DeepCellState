import math
import os
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
matplotlib.use("Agg")
import matplotlib as mpl
data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)

original_input = np.loadtxt("input_profiles", delimiter=",")
good = np.loadtxt("families/1", delimiter=",")
perts = np.loadtxt("perts.csv", delimiter=",", dtype=np.str)
# bad = np.loadtxt("bad.np", delimiter=",")
col = good[:, -1]
colors = []
colors.extend(["good"]*len(good))
# colors.extend(["bad"]*len(bad))
#
# pca = PCA(n_components=2)
# data = np.vstack([good, bad])
# good = pca.fit_transform(good[:, :-1])
#

chosen_perts = []
good = TSNE(n_components=2, random_state=0, perplexity=50).fit_transform(good[:, :-1]) # good[:, :-1]
center = [-3, -60]
for i, p in enumerate(good):
    dist = math.hypot(center[0] - p[0], center[1] - p[1])
    if dist < 15:
        chosen_perts.append(perts[i])
        # col[i] = -1
np.savetxt("cluster_perts.csv", np.asarray(chosen_perts), delimiter=",", fmt='%s')
#
# sns.scatterplot(x=A1[:, 0], y=A1[:, 1], hue=colors)
# plt.savefig("good_vs_bad_PCA.png")
# plt.close(None)
#

ax = sns.scatterplot(x=good[:, 0], y=good[:, 1], c=col, cmap=plt.cm.coolwarm, size=1) #, c=good[:, -1], cmap=plt.cm.coolwarm, size=1, alpha=0.5

ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.grid(b=True, which='major', color='black', linewidth=1.0)
ax.grid(b=True, which='minor', color='black', linewidth=0.5)

plt.savefig("tsne.svg")