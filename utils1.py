import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use("Agg")


def draw_vectors(vectors, output, names=None):
    if names is None:
        names = []
        for i in range(len(vectors)):
            names.append(str(i))
    input_size = len(vectors[0])
    # all_data = np.asarray(data)
    # vmin = np.min(all_data)
    # vmax = np.max(all_data)
    vmin = -1
    vmax = 1
    fig, axes = plt.subplots(nrows=len(vectors), ncols=1, figsize=(14, 4))
    fig.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=0.4, hspace=1.4)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    for j, ax in enumerate(axes.flatten()):
        if (j == 0):
            hm = sns.heatmap(vectors[j].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                             cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
        else:
            hm = sns.heatmap(vectors[j].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                             cbar=False, vmin=vmin, vmax=vmax)
        # ax.set_xticklabels(xlabels)
        ax.set_ylabel(names[j], rotation=90)
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


def draw_profiles(test_profile, decoded, closest_profile, input_size, output_file):
    img_data = [closest_profile.flatten(), decoded.flatten(), test_profile.flatten()]
    all_data = np.asarray(img_data)
    vmin = np.min(all_data)
    vmax = np.max(all_data)
    names = ["Baseline", "DeepCellState", "Ground truth"]
    fig, axes = plt.subplots(nrows=len(img_data), ncols=1, figsize=(14, 4))
    fig.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=0.4, hspace=1.4)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=1, as_cmap=True)
    for j, ax in enumerate(axes.flatten()):
        if (j == 0):
            hm = sns.heatmap(img_data[j].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                             cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
        else:
            hm = sns.heatmap(img_data[j].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                             cbar=False, vmin=vmin, vmax=vmax)
        # ax.set_xticklabels(xlabels)
        ax.set_ylabel(names[j], rotation=45)
        ax.tick_params(axis='x', rotation=0)
        ax.get_yaxis().set_label_coords(-0.05, 0.3)
        for label in hm.get_xticklabels():
            if np.int(label.get_text()) % 50 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

        for label in hm.get_yticklabels():
            label.set_visible(False)
        # ax.set_title(names[i], x=-1.05)
    plt.savefig(output_file)
    plt.close(None)


def draw_batch_profiles(profiles, input_size, output_file):
    img_data = []
    names = []
    for i, p in enumerate(profiles):
        img_data.append(p.flatten())
        names.append(str(i))
    names[-1] = "Mean"
    all_data = np.asarray(img_data)
    vmin = np.min(all_data)
    vmax = np.max(all_data)
    fig, axes = plt.subplots(nrows=len(img_data), ncols=1, figsize=(14, 4))
    fig.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=0.4, hspace=1.4)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=1, as_cmap=True)
    for j, ax in enumerate(axes.flatten()):
        if (j == 0):
            hm = sns.heatmap(img_data[j].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                             cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
        else:
            hm = sns.heatmap(img_data[j].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                             cbar=False, vmin=vmin, vmax=vmax)
        # ax.set_xticklabels(xlabels)
        ax.set_ylabel(names[j], rotation=45)
        ax.tick_params(axis='x', rotation=0)
        ax.get_yaxis().set_label_coords(-0.05, 0.3)
        for label in hm.get_xticklabels():
            if np.int(label.get_text()) % 50 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

        for label in hm.get_yticklabels():
            label.set_visible(False)
        # ax.set_title(names[i], x=-1.05)
    plt.savefig(output_file)
    plt.close(None)

def draw_one_profiles(profiles, input_size, output_file):
    img_data = []
    names = []
    for i, p in enumerate(profiles):
        img_data.append(p.flatten())
        names.append(str(i))
    names[-1] = "Mean"
    all_data = np.asarray(img_data)
    vmin = np.min(all_data)
    vmax = np.max(all_data)
    fig, ax = plt.subplots(nrows=len(img_data), ncols=1, figsize=(14, 4))
    fig.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=0.4, hspace=1.4)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=1, as_cmap=True)

    hm = sns.heatmap(img_data[0].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                         cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
    # ax.set_xticklabels(xlabels)
    ax.set_ylabel(names[0], rotation=45)
    ax.tick_params(axis='x', rotation=0)
    ax.get_yaxis().set_label_coords(-0.05, 0.3)
    for label in hm.get_xticklabels():
        if np.int(label.get_text()) % 50 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)

    for label in hm.get_yticklabels():
        label.set_visible(False)
        # ax.set_title(names[i], x=-1.05)
    plt.savefig(output_file)
    plt.close(None)


def draw_scatter_profiles(test_profile, decoded, closest_profile, output_file):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
    # fig.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=0.4, hspace=1.4)
    sns.scatterplot(x=decoded.flatten(), y=test_profile.flatten(), ax=axes[0])
    axes[0].set_title("DeepCellState")
    sns.scatterplot(x=closest_profile.flatten(), y=test_profile.flatten(), ax=axes[1])
    axes[1].set_title("Baseline")
    plt.savefig(output_file)
    plt.close(None)


def draw_dist(matrix, output_file):
    sns.distplot(matrix.flatten())
    plt.savefig(output_file)
    plt.close(None)