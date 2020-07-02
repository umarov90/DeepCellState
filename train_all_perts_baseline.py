import os
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from math import sqrt
from numpy import zeros
import sys
import re
import math
from scipy import stats
import pickle
from pathlib import Path
import time
from random import randint
import random
from random import uniform
from random import shuffle
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
import gc
import pandas as pd
from tensorflow.python.keras import backend as K
import numpy as np
# import the necessary packages
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import Conv1DTranspose
from tensorflow.python.keras.layers import GlobalMaxPooling1D
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import ReLU
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow import keras
import numpy as np
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config1 = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# config2 = tf.config.experimental.set_memory_growth(physical_devices[1], True)

tf.random.set_seed(0)
np.random.seed(0)

# parameters
BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 2e-5
input_size = 978
nb_epoch = 50
batch_size = 128
latent_dim = 64
vmin = -1


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


def build(input_size, filters=(64, 128)):
    input_shape = (input_size)
    inputs = Input(shape=input_shape)
    x = inputs
    # loop over the number of filters
    for f in filters:
        x = Dense(f)(x)  #
        x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    latent = Dense(input_size)(x)
    return Model(inputs, latent, name="encoder")


def test_loss(prediction, ground_truth):
    prediction = np.squeeze(np.asarray(prediction))
    ground_truth = np.squeeze(np.asarray(ground_truth))
    sub = np.subtract(prediction, ground_truth)
    square = np.square(sub)
    mean = square.mean()
    return mean


def parse_data(file):
    df = pd.read_csv(file, sep="\t")
    df.reset_index(drop=True, inplace=True)
    try:
        df = df.drop('Unnamed: 0', 1)
    except Exception as e:
        pass
    # df = df.drop("distil_id", 1)
    print(df.shape)
    #df = df.groupby("pert_id").filter(lambda x: len(x) > 100)
    df = df.groupby("cell_id").filter(lambda x: len(x) > 12040)
    print(df.shape)
    df = df.groupby(['cell_id', 'pert_id', 'pert_idose', 'pert_itime'], as_index=False).mean()
    print(df.shape)
    cell_ids = df["cell_id"].values
    pert_ids = df["pert_id"].values
    all_pert_ids = set(pert_ids)
    pert_idose = df["pert_idose"].values
    pert_itime = df["pert_itime"].values
    perts = np.stack([cell_ids, pert_ids, pert_idose, pert_itime]).transpose()
    df = df.drop(['cell_id', 'pert_id', 'pert_idose', 'pert_itime'], 1)
    data = df.values
    data = data / max(np.max(data), abs(np.min(data)))
    return data, perts, all_pert_ids


def split_data(data, meta, all_pert_ids):
    cell_types = set([meta[i][0] for i, x in enumerate(meta)])
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(meta)
    split = int(0.9 * len(all_pert_ids))
    all_pert_ids_list = list(all_pert_ids)
    shuffle(all_pert_ids_list)
    train_perts = all_pert_ids_list[:split]
    test_perts = all_pert_ids_list[split:]
    train_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in train_perts])
    test_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in test_perts])
    train_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in train_perts])
    test_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in test_perts])
    return train_data, test_data, train_meta, test_meta, cell_types, train_perts, test_perts


def get_duration(param):
    vals = param.split(" ")
    duration = float(vals[0])
    if vals[1] == "m":
        duration = duration * 60
    elif vals[1] == "h":
        duration = duration * 60 * 60
    return duration


def get_profile(data, meta_data, test_pert):
    meta_data = meta_data[test_pert[1]]
    if test_pert[2] == "-666" or test_pert[3] == "-666" or test_pert[2] == -666 or test_pert[3] == -666:
        return -1, None, None, None
    pert_list = [p[1] for p in meta_data if p[0][2] == test_pert[2] and p[0][3] == test_pert[3] and p[0][0] != test_pert[0]]
    if len(pert_list) > 0:
        random_best = randint(0, len(pert_list) - 1)
        median_profile = np.mean(np.asarray(data[pert_list]), axis=0, keepdims=True)
        return pert_list[random_best], np.asarray([data[pert_list[random_best]]]), median_profile, data[pert_list]
    else:
        return -1, None, None, None


data_folder = "/home/user/data/DeepFake/sub_base/"
os.chdir(data_folder)
shutil.rmtree('models')
os.makedirs('models')
shutil.rmtree('arrays')
os.makedirs('arrays')
# data_sh, meta_sh, all_pert_ids_sh = parse_data("lincs_trt_cp_phase_1.tsv")
# meta_dictionary_pert_test_sh = {}
# for pert_id in all_pert_ids_sh:
#     meta_dictionary_pert_test_sh[pert_id] = [[p, i] for i, p in enumerate(meta_sh) if p[1] == pert_id]
#
# pickle.dump(data_sh, open("arrays/data_sh", "wb"))
# pickle.dump(meta_sh, open("arrays/meta_sh", "wb"))
# pickle.dump(all_pert_ids_sh, open("arrays/all_pert_ids_sh", "wb"))
# pickle.dump(meta_dictionary_pert_test_sh, open("arrays/meta_dictionary_pert_test_sh", "wb"))

# data
if Path("arrays/train_data").is_file():
    print("Loading existing data")
    train_data = pickle.load(open("arrays/train_data", "rb"))
    test_data = pickle.load(open("arrays/test_data", "rb"))
    train_meta = pickle.load(open("arrays/train_meta", "rb"))
    test_meta = pickle.load(open("arrays/test_meta", "rb"))
    cell_types = pickle.load(open("arrays/cell_types", "rb"))
    meta_dictionary_pert = pickle.load(open("arrays/meta_dictionary_pert", "rb"))
    meta_dictionary_pert_test = pickle.load(open("arrays/meta_dictionary_pert_test", "rb"))
    data_dictionary_cell = pickle.load(open("arrays/data_dictionary_cell", "rb"))
    all_pert_ids = pickle.load(open("arrays/all_pert_ids", "rb"))
    train_perts = pickle.load(open("arrays/train_perts", "rb"))
    test_perts = pickle.load(open("arrays/test_perts", "rb"))
else:
    print("Parsing data")
    data, meta, all_pert_ids = parse_data("../lincs_trt_cp_phase_2.tsv")
    train_data, test_data, train_meta, test_meta, cell_types, train_perts, test_perts = split_data(data, meta, all_pert_ids)
    meta_dictionary_pert = {}
    for pert_id in train_perts:
        meta_dictionary_pert[pert_id] = [[p, i] for i, p in enumerate(train_meta) if p[1] == pert_id]
    meta_dictionary_pert_test = {}
    for pert_id in test_perts:
        meta_dictionary_pert_test[pert_id] = [[p, i] for i, p in enumerate(test_meta) if p[1] == pert_id]
    data_dictionary_cell = {}
    for cell in cell_types:
        data_dictionary_cell[cell] = [train_data[i] for i, p in enumerate(train_meta) if p[0] == cell]
    pickle.dump(train_data, open("arrays/train_data", "wb"))
    pickle.dump(test_data, open("arrays/test_data", "wb"))
    pickle.dump(train_meta, open("arrays/train_meta", "wb"))
    pickle.dump(test_meta, open("arrays/test_meta", "wb"))
    pickle.dump(cell_types, open("arrays/cell_types", "wb"))
    pickle.dump(meta_dictionary_pert, open("arrays/meta_dictionary_pert", "wb"))
    pickle.dump(meta_dictionary_pert_test, open("arrays/meta_dictionary_pert_test", "wb"))
    pickle.dump(data_dictionary_cell, open("arrays/data_dictionary_cell", "wb"))
    pickle.dump(all_pert_ids, open("arrays/all_pert_ids", "wb"))
    pickle.dump(train_perts, open("arrays/train_perts", "wb"))
    pickle.dump(test_perts, open("arrays/test_perts", "wb"))



# data_sh = pickle.load(open("arrays/data_sh", "rb"))
# meta_sh = pickle.load(open("arrays/meta_sh", "rb"))
# all_pert_ids_sh = pickle.load(open("arrays/all_pert_ids_sh", "rb"))
# meta_dictionary_pert_test_sh = pickle.load(open("arrays/meta_dictionary_pert_test_sh", "rb"))
print("----------------------------------------------")
print(train_data.shape)
print(test_data.shape)
print("----------------------------------------------")
cell_decoders = {}
if os.path.isdir("./models/main_model"):
    print("Loading model")
    for cell in cell_types:
        cell_decoders[cell] = keras.models.load_model("./models/cell")
else:
    print("Building autoencoder ")
    for cell in cell_types:
        autoencoder = build(input_size)
        autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-4))
        autoencoder.save("./models/" + cell)
        cell_decoders[cell] = autoencoder

for cell in cell_types:
    print(cell + " =========================================")
    cell_data = np.asarray([[train_data[i], train_meta[i]] for i, p in enumerate(train_meta) if p[0] == cell])
    input_profiles = []
    output_profiles = []
    for i in range(len(cell_data)):
        index, profile, median_profile, all_profiles = get_profile(train_data, meta_dictionary_pert, cell_data[i][1])
        if median_profile is not None:
            for p in all_profiles:
                input_profiles.append(p)
                output_profiles.append(cell_data[i][0])
    if len(input_profiles) == 0:
        continue
    input_profiles = np.asarray(input_profiles)
    output_profiles = np.asarray(output_profiles)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder = cell_decoders[cell]
    autoencoder.fit(input_profiles, output_profiles, epochs=100, batch_size=batch_size,
                    validation_split=0.1, callbacks=[callback])
    autoencoder.save("./models/" + cell)

results = {}
skipped = 0
img_count = 0
test_num = len(test_data)  # len(test_data)
our_data = []
baseline_data = []
gt_data = []
seen_perts = []
print("Total test objects: " + str(test_num))
for i in range(test_num):
    if i % 100 == 0:
        print(str(i) + " - ", end="", flush=True)
    test_meta_object = test_meta[i]
    closest, closest_profile, median_profile, all_profiles = get_profile(test_data, meta_dictionary_pert_test, test_meta_object)
    if closest_profile is None:
        skipped = skipped + 1
        continue
    if test_meta_object[1] in seen_perts:
        skipped = skipped + 1
        continue
    seen_perts.append(test_meta_object[1])

    test_profile = np.asarray([test_data[i]])

    autoencoder = cell_decoders[test_meta[i][0]]
    decoded1 = autoencoder.predict(closest_profile)
    results["Our performance is: "] = results.get("Our performance is: ", 0) + test_loss(decoded1, test_profile)

    results["Our correlation is: "] = results.get("Our correlation is: ", 0) + \
                                      stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]


    decoded1 = autoencoder.predict(median_profile)
    our_data.append(decoded1.flatten())
    gt_data.append(test_profile.flatten())
    results["Our performance is (median profile): "] = results.get("Our performance is (median profile): ",
                                                                   0) + test_loss(decoded1, test_profile)

    results["Our correlation: "] = results.get("Our correlation: ", 0) + \
                                   stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]

    zero_vector = zeros(decoded1.shape)
    # zero_vector.fill(0.5)
    results["zero vector loss is: "] = results.get("zero vector loss is: ", 0) + test_loss(zero_vector, test_profile)

    results["closest profile: "] = results.get("closest profile: ", 0) + test_loss(closest_profile, test_profile)
    results["closest profile correlation is: "] = results.get("closest profile correlation is: ", 0) + \
                                                  stats.pearsonr(closest_profile.flatten(), test_profile.flatten())[0]

    results["closest profile (median profile): "] = results.get("closest profile (median profile): ", 0) + test_loss(
        median_profile, test_profile)
    results["Baseline correlation: "] = results.get("Baseline correlation: ", 0) + \
                                        stats.pearsonr(median_profile.flatten(), test_profile.flatten())[0]
    baseline_data.append(median_profile.flatten())

    decoded3 = autoencoder.predict(test_profile)
    results["main autoencoder with test object as input (should be very good): "] = results.get(
        "main autoencoder with test object as input (should be very good): ", 0) + test_loss(decoded3, test_profile)
    results["cheating correlation: "] = results.get("cheating correlation: ", 0) + \
                                        stats.pearsonr(decoded3.flatten(), test_profile.flatten())[0]

    if img_count < 10:
        img_count = img_count + 1
        data = [decoded1, closest_profile, decoded3]
        names = ["ground truth", "our method", "closest profile", "cheating"]
        fig, axes = plt.subplots(nrows=len(data) + 1, ncols=1, figsize=(14, 4))
        fig.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=0.4, hspace=1.4)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
        for j, ax in enumerate(axes.flatten()):
            if (j == 0):
                hm = sns.heatmap(test_profile.reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                                 cbar_ax=cbar_ax, vmin=vmin, vmax=1)
            else:
                hm = sns.heatmap(data[j - 1].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                                 cbar=False, vmin=vmin, vmax=1)
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
        plt.savefig("profiles/profile" + str(i) + ".png")
        plt.close(None)
print(" Done")
for key, value in results.items():
    print(key + str(value / (test_num - skipped)))

gt_data = np.asarray(gt_data).transpose()
our_data = np.asarray(our_data).transpose()
baseline_data = np.asarray(baseline_data).transpose()

corr_gene_our = []
corr_gene_base = []
for i in range(input_size):
    corr_gene_our.append(stats.pearsonr(gt_data[i], our_data[i])[0])
    corr_gene_base.append(stats.pearsonr(gt_data[i], baseline_data[i])[0])

print("skipped " + str(skipped))
print("Improvement: " + str(results["zero vector loss is: "] / results["Our performance is: "]))
# corr_gene_our = np.asarray(corr_gene_our)
# corr_gene_base = np.asarray(corr_gene_base)
# np.savetxt("corr_gene_our.csv", corr_gene_our, delimiter=",")
# np.savetxt("corr_gene_base.csv", corr_gene_base, delimiter=",")
# print("gene corr: " + str(stats.pearsonr(corr_gene_our, corr_gene_base)[0]))
# print(np.mean(corr_gene_our))
# print(np.std(corr_gene_our))
# results = {}
# skipped = 0
# test_num = 10000
# for i in range(test_num):
#     if i % 100 == 0:
#         print(str(i) + " - ", end="", flush=True)
#     test_meta_object = meta_sh[i]
#     closest, closest_profile, median_profile = get_profile(data_sh, meta_dictionary_pert_test_sh, test_meta_object)
#
#     if closest_profile is None:
#         skipped = skipped + 1
#         continue
#
#     test_profile = np.asarray([data_sh[i]])
#
#     median_profile = median_profile / max(np.max(median_profile), abs(np.min(median_profile)))
#
#     weights = cell_decoders[meta_sh[i][0]]
#     autoencoder.get_layer("decoder").set_weights(weights)
#     decoded1 = autoencoder.predict(median_profile)
#     results["Our correlation: "] = results.get("Our correlation: ", 0) + \
#                                    stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]
#     results["Baseline correlation: "] = results.get("Baseline correlation: ", 0) + \
#                                     stats.pearsonr(median_profile.flatten(), test_profile.flatten())[0]
#
# print(" Done")
# for key, value in results.items():
#     print(key + str(value / (test_num - skipped)))
#
# print("skipped " + str(skipped))