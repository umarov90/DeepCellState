import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use("Agg")

def draw_vectors(vectors, output):
    n = 10
    data = []
    names = []
    for i in range(n):
        data.append(vectors[i])
        names.append(str(i+1))
    input_size = len(vectors[0])
    all_data = np.asarray(data)
    vmin = np.min(all_data)
    vmax = np.max(all_data)
    fig, axes = plt.subplots(nrows=len(data), ncols=1, figsize=(14, 4))
    fig.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=0.4, hspace=1.4)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    for j, ax in enumerate(axes.flatten()):
        if (j == 0):
            hm = sns.heatmap(data[j].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                             cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
        else:
            hm = sns.heatmap(data[j].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                             cbar=False, vmin=vmin, vmax=vmax)
        # ax.set_xticklabels(xlabels)
        ax.set_ylabel(names[j], rotation=45)
        ax.tick_params(axis='x', rotation=0)
        ax.get_yaxis().set_label_coords(-0.08, -0.5)
        for label in hm.get_xticklabels():
            if np.int(label.get_text()) % 50 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

        for label in hm.get_yticklabels():
            label.set_visible(False)
        # ax.set_title(names[i], x=-1.05)
    plt.savefig(output)
    plt.close(None)