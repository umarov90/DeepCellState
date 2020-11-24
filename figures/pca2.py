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
original_input = np.loadtxt("for_tsne_figure_profiles", delimiter=",")
latent_vectors = np.loadtxt("for_tsne_figure_vectors", delimiter=",")
perts = np.loadtxt("for_tsne_figure_perts_info", delimiter=",", dtype=np.str)
# bad = np.loadtxt("bad.np", delimiter=",")
col = latent_vectors[:, -1]
colors = []
colors.extend(["good"] * len(latent_vectors))


# colors.extend(["bad"]*len(bad))
#
# pca = PCA(n_components=2)
# data = np.vstack([good, bad])
# latent_vectors = pca.fit_transform(latent_vectors[:, :-1])
# original_input = pca.fit_transform(original_input)
#


def pcc(a, b):
    v = stats.pearsonr(a.flatten(), b.flatten())[0]
    return 1 - v


chosen_perts = []
latent_vectors = TSNE(n_components=2, random_state=0, perplexity=5, metric=pcc).fit_transform(latent_vectors[:, :-1])
original_input = TSNE(n_components=2, random_state=0, perplexity=5, metric=pcc).fit_transform(original_input)
center = [-3, -60]
special_perts = ["tacalcitol", "nefazodone", "eperezolid", "valrubicin", "alvespimycin"]
other = []
index_list = {}
for i, p in enumerate(latent_vectors):
    # dist = math.hypot(center[0] - p[0], center[1] - p[1])
    # if dist < 15:
    #     chosen_perts.append(perts[i])

    if perts[i][0] in special_perts:
        if perts[i][1] == "MCF7":
            index_list.setdefault(special_perts.index(perts[i][0]), [0, 0])[0] = i
        else:
            index_list.setdefault(special_perts.index(perts[i][0]), [0, 0])[1] = i
    else:
        other.append(i)

np.savetxt("cluster_perts.csv", np.asarray(chosen_perts), delimiter=",", fmt='%s')
#
# sns.scatterplot(x=A1[:, 0], y=A1[:, 1], hue=colors)
# plt.savefig("good_vs_bad_PCA.png")
# plt.close(None)
colors = sns.color_palette()
sns.scatterplot(x=original_input[:, 0][other], y=original_input[:, 1][other], s=20, alpha=0.2, ax=axes[0],
                label='_nolegend_')
for i, key in enumerate(index_list.keys()):
    sns.scatterplot(x=[original_input[:, 0][index_list[key]][0]], y=[original_input[:, 1][index_list[key]][0]],
                    s=100, alpha=1, ax=axes[0], marker=4, color=colors[i])
    sns.scatterplot(x=[original_input[:, 0][index_list[key]][1]], y=[original_input[:, 1][index_list[key]][1]],
                    s=80, alpha=1, ax=axes[0], marker=5, color=colors[i])
axes[0].set_title("Input space")

sns.scatterplot(x=latent_vectors[:, 0][other], y=latent_vectors[:, 1][other], s=20, alpha=0.2, ax=axes[1],
                label='_nolegend_')
for i, key in enumerate(index_list.keys()):
    sns.scatterplot(x=[latent_vectors[:, 0][index_list[key]][0]], y=[latent_vectors[:, 1][index_list[key]][0]],
                    s=100, alpha=1, ax=axes[1], marker=4, color=colors[i])
    sns.scatterplot(x=[latent_vectors[:, 0][index_list[key]][1]], y=[latent_vectors[:, 1][index_list[key]][1]],
                    s=80, alpha=1, ax=axes[1], marker=5, color=colors[i])
axes[1].set_title("Latent space")
# Put the legend out of the figure
custom = [Line2D([], [], marker='.', color='b', linestyle='None'),
          Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='r', linestyle='None'),
          Line2D([], [], marker='.', color='r', linestyle='None')]

special_perts = ["Tacalcitol", "Nefazodone", "Eperezolid", "Valrubicin", "Alvespimycin"]
legend_elements = [Line2D([0], [0], color='w', markerfacecolor=colors[i],
                          label=special_perts[i], marker='.', markersize=20)
                   for i in range(len(special_perts))]

legend_elements.append(Line2D([0], [0], linewidth=0, color='gray',
                              label="MCF-7", marker=4, markersize=10))
legend_elements.append(Line2D([0], [0], linewidth=0, color='gray',
                              label="PC-3", marker=5, markersize=8))

plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')  # , title="Drugs"

# ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# ax.grid(b=True, which='major', color='black', linewidth=1.0)
# ax.grid(b=True, which='minor', color='black', linewidth=0.5)
fig.suptitle('t-SNE visualization of MCF-7 and PC-3 profiles', fontsize=18)
# plt.tight_layout()
plt.savefig("tsne.png")
