import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use("Agg")
sns.set(font_scale=1.3, style='ticks')


def draw_profiles(test_profile, decoded, closest_profile, pname, input_size, output_file):
    img_data = [closest_profile.flatten(), decoded.flatten(), test_profile.flatten()]
    all_data = np.asarray(img_data)
    maxv = max(abs(np.min(all_data)), abs(np.max(all_data)))
    vmin = -maxv
    vmax = +maxv
    names = ["Baseline", "DeepCellState", "Ground truth"]
    fig, axes = plt.subplots(nrows=len(img_data), ncols=1, figsize=(10, 4))
    cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=1, as_cmap=True)
    for j, ax in enumerate(axes.flatten()):
        hm = sns.heatmap(img_data[j].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                             cbar=False, vmin=vmin, vmax=vmax, xticklabels=100)
        ax.set_ylabel(names[j], rotation=45)
        ax.tick_params(axis='x', rotation=0)
        ax.get_yaxis().set_label_coords(-0.1, 0.3)
        ax.yaxis.set_ticks_position('none')
        for label in hm.get_yticklabels():
            label.set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(right=0.84, top=0.8, wspace=0.35)
    cax = fig.add_axes([.90, .2, .02, .6])
    cax.set_xlabel('Expression')
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(points, cax=cax)
    fig.suptitle("MCF-7 " + pname + " response prediction using PC-3 response", fontsize=18)
    plt.savefig(output_file)
    plt.close(None)


def fix(s):
    return "".join([x if x.isalnum() else "_" for x in s])


def draw_scatter_profiles(test_profile, decoded, closest_profile, pname, output_file):
    cmap = sns.color_palette("dark:salmon", as_cmap=True)
    col1 = np.abs(closest_profile - test_profile).flatten()
    col2 = np.abs(decoded - test_profile).flatten()
    vmin = 0
    vmax = max(np.max(col1), np.max(col2))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    sns.scatterplot(x=closest_profile.flatten(), y=test_profile.flatten(), ax=axes[0], c=col1,
                    cmap=cmap, vmin=vmin, vmax=vmax, s=20)
    axes[0].set_title("Baseline")
    axes[0].set(xlabel='Predicted', ylabel='Observed')
    sns.scatterplot(x=decoded.flatten(), y=test_profile.flatten(), ax=axes[1], c=col2,
                    cmap=cmap, vmin=vmin, vmax=vmax, s=20)
    axes[1].set_title("DeepCellState")
    axes[1].set(xlabel='Predicted', ylabel='Observed')
    plt.tight_layout()
    fig.subplots_adjust(right=0.84, top=0.8, wspace=0.35)
    cax = fig.add_axes([.90, .2, .02, .6])
    cax.set_xlabel('Error')
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(points, cax=cax)
    fig.suptitle("Predicted MCF-7 " + pname + " response gene values", fontsize=18)
    plt.savefig(output_file)
    plt.close(None)



