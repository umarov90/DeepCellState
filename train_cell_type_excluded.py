import os
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
import utils1

tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config1 = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# config2 = tf.config.experimental.set_memory_growth(physical_devices[1], True)

#tf.random.set_seed(0)
#np.random.seed(0)

# parameters
wdir = "sub4/"
test_fold = "1"
input_size = 978
nb_epoch = 100
batch_size = 128
latent_dim = 128


def build(input_size, channels, latent_dim):
    layer_units = [512, 256]
    input_shape = (input_size, channels)
    drop_rate = 0.8
    inputs = Input(shape=input_shape)
    x = inputs
    x = Dropout(0.4, input_shape=(None, 978, 1))(x)
    for f in layer_units:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(drop_rate, input_shape=(None, input_size, layer_units[1]))(x)
    shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, kernel_regularizer=regularizers.l2(1e-5),
                   activity_regularizer=regularizers.l1(1e-5))(x)
    #, kernel_regularizer=regularizers.l2(1e-5),
    #               activity_regularizer=regularizers.l1(1e-5)
    encoder = Model(inputs, latent, name="encoder")
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(shape[1] * shape[2])(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)
    for f in layer_units[::-1]:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(drop_rate, input_shape=(None, input_size, layer_units[0]))(x)
    x = Dense(1)(x)
    outputs = Activation("tanh")(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    return autoencoder


def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction-ground_truth)**2))


def parse_data(file):
    print("Parsing data at " + file)
    df = pd.read_csv(file, sep="\t")
    df.reset_index(drop=True, inplace=True)
    print("Total: " + str(df.shape))
    #df = df[(df['cell_id'] == "MCF7") | (df['cell_id'] == "PC3")]
    df = df[(df['cell_id'] == "MCF7") | (df['cell_id'] == "PC3") |
            (df['cell_id'] == "VCAP") | (df['cell_id'] == "A549") | (df['cell_id'] == "A375")]
    df = df[(df['pert_type'] == "trt_cp")]
    # df = df[(df['pert_type'] == "trt_cp") | (df['pert_type'] == "trt_sh") |
    #         (df['pert_type'] == "trt_sh.cgs") | (df['pert_type'] == "trt_sh.css") |
    #         (df['pert_type'] == "trt_oe") | (df['pert_type'] == "trt_lig")]
    # df = df[(df['pert_type'] == "trt_cp")]
    print("Cell filtering: " + str(df.shape))
    df = df.groupby(['cell_id', 'pert_id']).filter(lambda x: len(x) > 1)
    print("Pert filtering: " + str(df.shape))
    # df = df.drop_duplicates(['cell_id', 'pert_id', 'pert_idose', 'pert_itime', 'pert_type'])
    # df = df.groupby(['cell_id', 'pert_id'], as_index=False).mean()
    df = df.groupby(['cell_id', 'pert_id', 'pert_type'], as_index=False).mean()  # , 'pert_type'
    # print("Merging: " + str(df.shape))
    # df.pert_type.value_counts().to_csv("trt_count_final.tsv", sep='\t')
    cell_ids = df["cell_id"].values
    pert_ids = df["pert_id"].values
    all_pert_ids = set(pert_ids)
    # pert_idose = df["pert_idose"].values
    # pert_itime = df["pert_itime"].values
    pert_type = df["pert_type"].values
    perts = np.stack([cell_ids, pert_ids, pert_type]).transpose()
    df = df.drop(['cell_id', 'pert_id', 'pert_type', 'Unnamed: 0'], 1)
    data = df.values
    # data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # for i in range(len(data)):
    data = data / max(np.max(data), abs(np.min(data)))
    # data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data))  - 1
    data = np.expand_dims(data, axis=-1)
    return data, perts, all_pert_ids


def split_data(data, meta, all_pert_ids):
    print(test_fold)
    cell_types = set([meta[i][0] for i, x in enumerate(meta)])
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(meta)
    all_pert_ids_list = list(all_pert_ids)
    shuffle(all_pert_ids_list)

    test_perts = np.loadtxt("folds/" + str(test_fold), dtype='str')
    z = list(all_pert_ids - set(test_perts))
    shuffle(z)
    train_perts = z[:int(0.95 * len(z))]
    val_perts = z[int(0.95 * len(z)):]

    train_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in train_perts and m[0] != "A375"])
    test_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in test_perts])
    val_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in val_perts])
    train_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in train_perts and m[0] != "A375"])
    test_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in test_perts])
    val_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in val_perts])

    return train_data, train_meta, test_data, test_meta, val_data, val_meta, cell_types, train_perts, val_perts, test_perts


def get_profile(data, meta_data, test_pert, train_data=False):
    if train_data:
        pert_list = [p[1] for p in meta_data if
                 p[0][0] != test_pert[0]]  # and p[0][2] == test_pert[2] and p[0][3] == test_pert[3]
    else:
        pert_list = [p[1] for p in meta_data if
                 p[0][0] != test_pert[0] and p[0][0] == "A375"]
    if len(pert_list) > 0:
        random_best = randint(0, len(pert_list) - 1)
        mean_profile = np.mean(np.asarray(data[pert_list]), axis=0, keepdims=True)
        return random_best, np.asarray([data[pert_list[random_best]]]), mean_profile, data[pert_list]
    else:
        return -1, None, None, None


data_folder = "/home/user/data/DeepFake/" + wdir
os.chdir(data_folder)
shutil.rmtree('models')
os.makedirs('models')
shutil.rmtree('arrays')
os.makedirs('arrays')

# data
if Path("arrays/train_data").is_file():
    print("Loading existing data")
    train_data = pickle.load(open("arrays/train_data", "rb"))
    test_data = pickle.load(open("arrays/test_data", "rb"))
    train_meta = pickle.load(open("arrays/train_meta", "rb"))
    test_meta = pickle.load(open("arrays/test_meta", "rb"))
    cell_types = pickle.load(open("arrays/cell_types", "rb"))
    all_pert_ids = pickle.load(open("arrays/all_pert_ids", "rb"))
    val_data = pickle.load(open("arrays/val_data", "rb"))
    val_meta = pickle.load(open("arrays/val_meta", "rb"))
    meta_dictionary_pert = pickle.load(open("arrays/meta_dictionary_pert", "rb"))
    meta_dictionary_pert_test = pickle.load(open("arrays/meta_dictionary_pert_test", "rb"))
    meta_dictionary_pert_val = pickle.load(open("arrays/meta_dictionary_pert_val", "rb"))
else:
    print("Parsing data")
    data, meta, all_pert_ids = parse_data("../LINCS/lincs_phase_1_2.tsv")
    train_data, train_meta, test_data, test_meta, val_data, \
    val_meta, cell_types, train_perts, val_perts, test_perts = split_data(data, meta, all_pert_ids)
    meta_dictionary_pert = {}
    for pert_id in train_perts:
        meta_dictionary_pert[pert_id] = [[p, i] for i, p in enumerate(train_meta) if p[1] == pert_id]
    meta_dictionary_pert_test = {}
    for pert_id in test_perts:
        meta_dictionary_pert_test[pert_id] = [[p, i] for i, p in enumerate(test_meta) if p[1] == pert_id]
    meta_dictionary_pert_val = {}
    for pert_id in val_perts:
        meta_dictionary_pert_val[pert_id] = [[p, i] for i, p in enumerate(val_meta) if p[1] == pert_id]
    pickle.dump(meta_dictionary_pert, open("arrays/meta_dictionary_pert", "wb"))
    pickle.dump(meta_dictionary_pert_test, open("arrays/meta_dictionary_pert_test", "wb"))
    pickle.dump(meta_dictionary_pert_val, open("arrays/meta_dictionary_pert_val", "wb"))
    pickle.dump(train_data, open("arrays/train_data", "wb"))
    pickle.dump(test_data, open("arrays/test_data", "wb"))
    pickle.dump(train_meta, open("arrays/train_meta", "wb"))
    pickle.dump(test_meta, open("arrays/test_meta", "wb"))
    pickle.dump(cell_types, open("arrays/cell_types", "wb"))
    pickle.dump(all_pert_ids, open("arrays/all_pert_ids", "wb"))
    pickle.dump(val_data, open("arrays/val_data", "wb"))
    pickle.dump(val_meta, open("arrays/val_meta", "wb"))

# data_sh = pickle.load(open("arrays/data_sh", "rb"))
# meta_sh = pickle.load(open("arrays/meta_sh", "rb"))

print("----------------------------------------------")
print(train_data.shape)
print(test_data.shape)
print("----------------------------------------------")
cell_decoders = {}
if os.path.isdir("./models/main_model"):
    print("Loading model")
    autoencoder = keras.models.load_model("./models/main_model")
    for cell in cell_types:
        cell_decoders[cell] = pickle.load(open("./models/" + cell + "_decoder_weights", "rb"))
else:
    print("Building autoencoder ")
    autoencoder = build(input_size, 1, latent_dim)
    autoencoder.save("./models/main_model")

train_perf = []
val_perf = []
should_train = True
if should_train:
    del autoencoder
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    count = 0
    e = 0
    while e < nb_epoch:
        print("Total epoch " + str(e) + " ------------------------------------------------------")
        autoencoder = keras.models.load_model("./models/main_model")
        encoder = autoencoder.get_layer("encoder")
        encoder.trainable = True
        encoder.compile(loss="mse", optimizer=Adam(lr=1e-4))
        autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-4))

        if e == 0:
            print("Main autoencoder" + " =========================================")
            #autoencoder = keras.models.load_model("default_autoencoder")
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            autoencoder.fit(train_data, train_data, epochs=200, batch_size=batch_size, validation_split=0.1,
                           callbacks=[callback])  # , validation_split=0.1, callbacks=[callback]
            autoencoder.save("default_autoencoder")
            for cell in cell_types:
                decoder = autoencoder.get_layer("decoder")
                cell_decoders[cell] = decoder.get_weights().copy()
                pickle.dump(cell_decoders[cell], open("./models/" + cell + "_decoder_weights", "wb"))
                del decoder

        utils1.draw_vectors(encoder.predict(val_data), "latent_vectors/" + str(e) + "_1")
        print("Training decoders again")
        decoder = autoencoder.get_layer("decoder")
        encoder_weights_copy = encoder.get_weights().copy()
        original_main_decoder_weights = autoencoder.get_layer("decoder").get_weights()
        cl = list(cell_types)
        random.shuffle(cl)
        if e == nb_epoch - 1:
            print("training with frozen encoder")
            encoder.trainable = False
            decoder.trainable = True
            autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-4))
        for cell in cl:
            if cell == "A375":
                continue
            print(cell + " =========================================")
            cell_data = np.asarray([[train_data[i], train_meta[i]] for i, p in enumerate(train_meta) if p[0] == cell])
            input_profiles = []
            output_profiles = []
            for i in range(len(cell_data)):
                #input_profiles.append(cell_data[i][0])
                #output_profiles.append(cell_data[i][0])
                closest, profile, median_profile, all_profiles = get_profile(train_data,
                                                                             meta_dictionary_pert[cell_data[i][1][1]],
                                                                             cell_data[i][1], train_data=True)
                if median_profile is not None:
                    for p in all_profiles:
                        input_profiles.append(p)
                        output_profiles.append(cell_data[i][0])

            input_profiles = np.asarray(input_profiles)
            output_profiles = np.asarray(output_profiles)
            autoencoder.get_layer("decoder").set_weights(cell_decoders[cell])
            if e == nb_epoch - 1:
                cell_data_val = np.asarray([[val_data[i], val_meta[i]] for i, p in enumerate(val_meta) if p[0] == cell])
                input_profiles_val = []
                output_profiles_val = []
                for i in range(len(cell_data_val)):
                    closest, profile, median_profile, all_profiles = get_profile(val_data,
                                                                                 meta_dictionary_pert_val[
                                                                                     cell_data_val[i][1][1]],
                                                                                 cell_data_val[i][1])
                    if median_profile is not None:
                        for p in all_profiles:
                            input_profiles_val.append(p)
                            output_profiles_val.append(cell_data_val[i][0])
                input_profiles_val = np.asarray(input_profiles_val)
                output_profiles_val = np.asarray(output_profiles_val)
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
                autoencoder.fit(input_profiles, output_profiles, epochs=2000, batch_size=batch_size,
                                validation_data=(input_profiles_val, output_profiles_val), callbacks=[callback])
            else:
                autoencoder.fit(input_profiles, output_profiles, epochs=2, batch_size=batch_size)

            cell_decoders[cell] = autoencoder.get_layer("decoder").get_weights()
            gc.collect()
        autoencoder.get_layer("decoder").set_weights(original_main_decoder_weights)
        a = (encoder_weights_copy[0] == autoencoder.get_layer("encoder").get_weights()[0]).all()
        a = a and (encoder_weights_copy[1] == autoencoder.get_layer("encoder").get_weights()[1]).all()
        print(a)
        print("---------------------------------------------------------------\n")
        autoencoder.save("./models/main_model")
        for cell in cell_types:
            pickle.dump(cell_decoders[cell], open("./models/" + cell + "_decoder_weights", "wb"))

        # train_cor_sum = 0.0
        # train_count = 0
        # train_rmse_sum = 0.0
        # seen_perts = []
        # for i in range(len(train_data)):
        #     train_meta_object = train_meta[i]
        #     if train_meta_object[1] in seen_perts:
        #         continue
        #     closest, closest_profile, median_profile, all_profiles = get_profile(train_data,
        #                                                                          meta_dictionary_pert[
        #                                                                              train_meta_object[1]],
        #                                                                          train_meta_object, train_data=True)
        #     if closest_profile is None:
        #         continue
        #     seen_perts.append(train_meta_object[1])
        #     train_count = train_count + 1
        #     weights = cell_decoders[train_meta_object[0]]
        #     autoencoder.get_layer("decoder").set_weights(weights)
        #     decoded1 = autoencoder.predict(closest_profile)
        #     train_cor_sum = train_cor_sum + stats.pearsonr(decoded1.flatten(), train_data[i].flatten())[0]
        #     train_rmse_sum = train_rmse_sum + test_loss(decoded1, train_data[i])
        # train_cor = train_cor_sum / train_count
        # train_perf.append(train_rmse_sum / train_count)
        # print("Training pcc: " + str(train_cor))
        # print("Evaluated:" + str(train_count))

        val_cor_sum = 0.0
        val_rmse_sum = 0.0
        val_count = 0
        seen_perts = []
        for i in range(len(val_data)):
            val_meta_object = val_meta[i]
            if val_meta_object[1] in seen_perts:
                continue
            closest, closest_profile, median_profile, all_profiles = get_profile(val_data,
                                                                                 meta_dictionary_pert_val[
                                                                                     val_meta_object[1]],
                                                                                 val_meta_object)
            if closest_profile is None:
                continue
            seen_perts.append(val_meta_object[1])
            val_count = val_count + 1
            weights = cell_decoders[val_meta_object[0]]
            autoencoder.get_layer("decoder").set_weights(weights)
            decoded1 = autoencoder.predict(closest_profile)
            val_cor_sum = val_cor_sum + stats.pearsonr(decoded1.flatten(), val_data[i].flatten())[0]
            val_rmse_sum = val_rmse_sum + test_loss(decoded1, train_data[i])
        val_cor = val_cor_sum / val_count
        val_rmse = val_rmse_sum / val_count
        val_perf.append(val_rmse)
        print("Validation pcc: " + str(val_cor))
        print("Evaluated:" + str(val_count))
        if e == 0:
            best_val_cor = val_rmse
        else:
            if val_cor < best_val_cor:
                count = count + 1
            else:
                best_val_cor = val_cor
                count = 0
                autoencoder.save("./best/main_model")
                for cell in cell_types:
                    pickle.dump(cell_decoders[cell], open("./best/" + cell + "_decoder_weights", "wb"))

        if count > 2:
            e = nb_epoch - 2
            count = 0
            autoencoder = keras.models.load_model("./best/main_model")
            for cell in cell_types:
                cell_decoders[cell] = pickle.load(open("./best/" + cell + "_decoder_weights", "rb"))

        del autoencoder
        del encoder
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("---------------------------------------------------------------\n")
        e = e + 1

autoencoder = keras.models.load_model("./models/main_model")
encoder = autoencoder.get_layer("encoder")
decoder = autoencoder.get_layer("decoder")


results = {}
skipped = 0
img_count = 0
test_num = len(test_data)  # len(test_data)
our_data = []
baseline_data = []
gt_data = []
seen_perts = []
print("Total test objects: " + str(test_num))
worse_base = 0
all_results = []
closest_cor = 0

test_trt = "trt_cp"
for i in range(test_num):
    if i % 100 == 0:
        print(str(i) + " - ", end="", flush=True)
    test_meta_object = test_meta[i]
    if test_meta_object[2] != test_trt:
        continue
    closest, closest_profile, median_profile, all_profiles = get_profile(test_data,
                                                                         meta_dictionary_pert_test[test_meta_object[1]],
                                                                         test_meta_object)
    if closest_profile is None:
        skipped = skipped + 1
        continue
    if test_meta_object[1] in seen_perts:
        skipped = skipped + 1
        continue
    seen_perts.append(test_meta_object[1])

    test_profile = np.asarray([test_data[i]])

    # closest_cor = closest_cor + find_closest_corr(train_data, closest_profile, test_profile)

    weights = cell_decoders[test_meta[i][0]]
    autoencoder.get_layer("decoder").set_weights(weights)
    decoded1 = autoencoder.predict(closest_profile)

    results["count"] = results.get("count", 0) + 1
    results["Our performance is: "] = results.get("Our performance is: ", 0) + test_loss(decoded1, test_profile)

    results["Our correlation is: "] = results.get("Our correlation is: ", 0) + \
                                      stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]

    results["Our spearman is: "] = results.get("Our spearman is: ", 0) + \
                                      stats.spearmanr(decoded1.flatten(), test_profile.flatten())[0]

    decoded1 = autoencoder.predict(median_profile)
    our_data.append(decoded1.flatten())
    gt_data.append(test_profile.flatten())
    results["Our performance is (median profile): "] = results.get("Our performance is (median profile): ",
                                                                   0) + test_loss(decoded1, test_profile)

    results["Our correlation: "] = results.get("Our correlation: ", 0) + \
                                   stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]

    all_results.append(str(stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]) + ", " +
                       str(stats.pearsonr(closest_profile.flatten(), test_profile.flatten())[0]) + ", "
                       + test_meta_object[0] + ", " + test_meta_object[1])

    zero_vector = zeros(decoded1.shape)
    # zero_vector.fill(0.5)
    results["zero vector loss is: "] = results.get("zero vector loss is: ", 0) + test_loss(zero_vector, test_profile)

    results["closest profile: "] = results.get("closest profile: ", 0) + test_loss(closest_profile, test_profile)
    results["closest profile correlation is: "] = results.get("closest profile correlation is: ", 0) + \
                                                  stats.pearsonr(closest_profile.flatten(), test_profile.flatten())[0]

    results["closest profile spearman is: "] = results.get("closest profile spearman is: ", 0) + \
                                      stats.spearmanr(closest_profile.flatten(), test_profile.flatten())[0]

    results["closest profile (median profile): "] = results.get("closest profile (median profile): ", 0) + test_loss(
        median_profile, test_profile)
    results["Baseline correlation: "] = results.get("Baseline correlation: ", 0) + \
                                        stats.pearsonr(median_profile.flatten(), test_profile.flatten())[0]

    baseline_data.append(median_profile.flatten())

    autoencoder.get_layer("decoder").set_weights(weights)
    decoded3 = autoencoder.predict(test_profile)
    results["main autoencoder with test object as input (should be very good): "] = results.get(
        "main autoencoder with test object as input (should be very good): ", 0) + test_loss(decoded3, test_profile)
    results["cheating correlation: "] = results.get("cheating correlation: ", 0) + \
                                        stats.pearsonr(decoded3.flatten(), test_profile.flatten())[0]

    if img_count < 10:
        img_count = img_count + 1
        data = [decoded1, closest_profile, decoded3]
        all_data = np.asarray(data)
        vmin = np.min(all_data)
        vmax = np.max(all_data)
        names = ["ground truth", "our method", "closest profile", "cheating"]
        fig, axes = plt.subplots(nrows=len(data) + 1, ncols=1, figsize=(14, 4))
        fig.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=0.4, hspace=1.4)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
        for j, ax in enumerate(axes.flatten()):
            if (j == 0):
                hm = sns.heatmap(test_profile.reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                                 cbar_ax=cbar_ax, vmin=vmin, vmax=vmax)
            else:
                hm = sns.heatmap(data[j - 1].reshape(1, input_size), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
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
        plt.savefig("profiles/profile" + str(i) + ".png")
        plt.close(None)
print(" Done")

for key, value in results.items():
    if key == "count":
        continue
    print(key + str(value / results["count"]))
print("-----------------------------------------------------------------------------------------------------------")
print(str(results["Our performance is: "] / results["count"]) + "\t" + str(results["Our correlation is: "] / results["count"])
            + "\t" + str(results["Our spearman is: "]/ results["count"]) + "\t" + str(results["closest profile: "] / results["count"])
      + "\t" + str(results["closest profile correlation is: "] / results["count"]) + "\t" + str(results["closest profile spearman is: "] / results["count"]))
print("-----------------------------------------------------------------------------------------------------------")
# print("closest train corr:" + str(closest_cor / (test_num - skipped)))

with open("all_results_per_drug.csv", 'w+') as f:
    f.write('\n'.join(all_results))
print("skipped " + str(skipped))