import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
matplotlib.use("Agg")
sns.set(font_scale=1.3, style='white')

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
    maxv = max(abs(np.min(all_data)), abs(np.max(all_data)))
    vmin = -maxv # np.min(all_data)
    vmax = +maxv # np.max(all_data)
    names = ["Baseline", "DeepCellState", "Ground truth"]
    fig, axes = plt.subplots(nrows=len(img_data), ncols=1, figsize=(12, 4))
    fig.subplots_adjust(left=0.2, bottom=None, right=0.85, top=0.8, wspace=0.4, hspace=1.4)
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
        ax.get_yaxis().set_label_coords(-0.1, 0.3)
        for label in hm.get_xticklabels():
            if np.int(label.get_text()) % 50 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

        for label in hm.get_yticklabels():
            label.set_visible(False)
        # ax.set_title(names[i], x=-1.05)
    fig.suptitle('MCF-7 alvespimycin response prediction using PC-3 response', fontsize=18)
    plt.savefig(output_file)
    plt.close(None)


def fix(s):
    return "".join([x if x.isalnum() else "_" for x in s])


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
    fig.subplots_adjust(left=None, bottom=None, right=0.85, top=0.9, wspace=0.4, hspace=1.4)
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
    cmap = sns.color_palette("dark:salmon", as_cmap=True)
    # cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=1, as_cmap=True)
    col1 = np.abs(closest_profile - test_profile).flatten()
    col2 = np.abs(decoded - test_profile).flatten()
    vmin = 0
    vmax = max(np.max(col1), np.max(col2))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    # sns.kdeplot(x=closest_profile.flatten(), y=test_profile.flatten(), fill=True, ax=axes[0], levels=5)
    sns.scatterplot(x=closest_profile.flatten(), y=test_profile.flatten(), ax=axes[0], c=col1,
                    cmap=cmap, vmin=vmin, vmax=vmax, s=20)
    axes[0].set_title("Baseline")
    # axes[0].set(xlabel='Baseline gene value', ylabel='True gene value')
    # sns.kdeplot(x=decoded.flatten(), y=test_profile.flatten(), fill=True, ax=axes[1], levels=5)
    sns.scatterplot(x=decoded.flatten(), y=test_profile.flatten(), ax=axes[1], c=col2,
                    cmap=cmap, vmin=vmin, vmax=vmax, s=20)

    # plt.ylim(-1, 1)
    # plt.xlim(-1, 1)

    axes[1].set_title("DeepCellState")
    # axes[0].set(xlabel='Predicted gene value', ylabel='True gene value')
    # axes[0].text(-1, 1.1, letter, transform=ax.transAxes, size=20, weight='bold')
    # fig.suptitle(letter, size=20, weight='bold', horizontalalignment='left', x=0.1, y=.95)

    # Make space for the colorbar
    fig.subplots_adjust(right=.92, top=0.8)

    # Define a new Axes where the colorbar will go
    cax = fig.add_axes([.94, .25, .02, .6])

    # Get a mappable object with the same colormap as the data
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)

    # Draw the colorbar
    fig.colorbar(points, cax=cax)
    fig.suptitle('MCF-7 alvespimycin response prediction using PC-3 response', fontsize=18)
    plt.savefig(output_file)
    plt.close(None)


def draw_dist(matrix, output_file):
    sns.distplot(matrix.flatten())
    plt.savefig(output_file)
    plt.close(None)



