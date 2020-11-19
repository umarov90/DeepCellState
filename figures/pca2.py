import math
import os
from sklearn.decomposition import PCA
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
matplotlib.use("Agg")
import matplotlib as mpl
data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)
sns.set(font_scale=1.3, style='white')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), constrained_layout=True)
# fig.subplots_adjust(top=0.9)
original_input = np.loadtxt("input_profiles", delimiter=",")
good = np.loadtxt("good.np", delimiter=",")
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
# original_input = pca.fit_transform(original_input)
#


def pcc(a, b):
    v = stats.pearsonr(a.flatten(), b.flatten())[0]
    return 1 - v

chosen_perts = []
good = TSNE(n_components=2, random_state=0, perplexity=5, metric=pcc).fit_transform(good[:, :-1])
original_input = TSNE(n_components=2, random_state=0, perplexity=5, metric=pcc).fit_transform(original_input)
center = [-3, -60]
special_perts = ["tacalcitol", "nefazodone", "XL-888", "valrubicin", "alvespimycin"]
other = []
index_list = {}
for i, p in enumerate(good):
    # dist = math.hypot(center[0] - p[0], center[1] - p[1])
    # if dist < 15:
    #     chosen_perts.append(perts[i])

    if perts[i] in special_perts:
        index_list.setdefault(special_perts.index(perts[i]), []).append(i)
    else:
        other.append(i)

np.savetxt("cluster_perts.csv", np.asarray(chosen_perts), delimiter=",", fmt='%s')
#
# sns.scatterplot(x=A1[:, 0], y=A1[:, 1], hue=colors)
# plt.savefig("good_vs_bad_PCA.png")
# plt.close(None)

sns.scatterplot(x=original_input[:, 0][other], y=original_input[:, 1][other], s=20, alpha=0.2, ax=axes[0], label='_nolegend_')
for key in index_list.keys():
    sns.scatterplot(x=original_input[:, 0][index_list[key]], y=original_input[:, 1][index_list[key]], s=80, alpha=0.8, ax=axes[0])
axes[0].set_title("Input space")

sns.scatterplot(x=good[:, 0][other], y=good[:, 1][other], s=20, alpha=0.2, ax=axes[1], label='_nolegend_')
for key in index_list.keys():
    sns.scatterplot(x=good[:, 0][index_list[key]], y=good[:, 1][index_list[key]], s=80, alpha=0.8, ax=axes[1])
axes[1].set_title("Latent space")
# Put the legend out of the figure
custom = [Line2D([], [], marker='.', color='b', linestyle='None'),
          Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='r', linestyle='None')]

special_perts = ["Tacalcitol", "Nefazodone", "XL-888", "Valrubicin", "Alvespimycin"]
plt.legend(special_perts, bbox_to_anchor=(1.05, 1), loc='upper left', title="Drugs",)


# ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# ax.grid(b=True, which='major', color='black', linewidth=1.0)
# ax.grid(b=True, which='minor', color='black', linewidth=0.5)
fig.suptitle('t-SNE visualization of MCF-7 and PC-3 profiles', fontsize=18)
# plt.tight_layout()
plt.savefig("tsne.png")