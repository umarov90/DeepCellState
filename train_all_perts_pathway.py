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
input_size = 251
nb_epoch = 50
batch_size = 128
latent_dim = 32
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


def build(input_size, channels, latent_dim, filters=(64, 128), encoder=None):
    if encoder is None:
        input_shape = (input_size, channels)
        # define the input to the encoder
        inputs = Input(shape=input_shape)
        x = inputs
        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            x = Conv1D(f, 1, strides=1, padding="same", use_bias=True)(x)  #
            x = LeakyReLU(alpha=0.2)(x)
            # x = Activation('tanh')(x)
            #x = BatchNormalization()(x)
        # flatten the network and then construct our latent vector
        volume_size = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latent_dim)(x)
        # , kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        #                bias_regularizer=regularizers.l2(1e-4),
        #                activity_regularizer=regularizers.l2(1e-5)
        # latent = BatchNormalization()(latent)
        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")
    else:
        # for l in encoder.layers:
        #    l.trainable = False
        # encoder.trainable = False
        inputs = encoder.input
        volume_size = encoder.layers[-3].input_shape
    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    latentInputs = Input(shape=(latent_dim,))
    x = Dense(np.prod(volume_size[1:]))(latentInputs)
    x = Reshape((volume_size[1], volume_size[2]))(x)
    # x = Dropout(0.5)(x)
    # loop over our number of filters again, but this time in
    # reverse order
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv1DTranspose(f, 1, strides=1, padding="same", use_bias=True)(
            x)  # kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY
        x = LeakyReLU(alpha=0.2)(x)
        # x = Activation('tanh')(x)
        #x = BatchNormalization()(x)

    # apply a single CONV_TRANSPOSE layer used to recover the
    # original depth of the image
    ###########################################################################################################
    # Maybe her BN as well
    ###########################################################################################################
    x = Conv1DTranspose(channels, 1, padding="same")(x)
    # outputs = LeakyReLU(alpha=0.2)(x)
    # outputs = x
    outputs = Activation('tanh')(x)
    # outputs = BatchNormalization()(outputs)
    # build the decoder model
    decoder = Model(latentInputs, outputs, name="decoder")
    # our autoencoder is the encoder + decoder
    autoencoder = Model(inputs, decoder(encoder(inputs)),
                        name="autoencoder")
    # return a 3-tuple of the encoder, decoder, and autoencoder
    return autoencoder


def test_loss(prediction, ground_truth):
    prediction = np.squeeze(np.asarray(prediction))
    ground_truth = np.squeeze(np.asarray(ground_truth))
    sub = np.subtract(prediction, ground_truth)
    square = np.square(sub)
    mean = square.mean()
    return mean


def parse_data(file):
    df = pd.read_csv(file)
    print(df.shape)
    df.reset_index(drop=True, inplace=True)
    df = df.drop(df.columns[0], axis=1)
    df.drop(df.tail(1).index,inplace=True)
    cell_ids = []
    pert_ids = []
    for col in df.columns:
        vals = col.split("@")
        pert_ids.append(vals[0])
        cell_ids.append(vals[1])
    all_pert_ids = set(pert_ids)
    perts = np.stack([cell_ids, pert_ids]).transpose()
    data = df.values.transpose()
    data = data / max(np.max(data), abs(np.min(data)))
    data = np.expand_dims(data, axis=-1)
    return data, perts, all_pert_ids


def split_data(data, meta, all_pert_ids):
    cell_types = set([meta[i][0] for i, x in enumerate(meta)])
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(meta)
    split = int(0.9 * len(all_pert_ids))
    all_pert_ids_list = list(all_pert_ids)
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
    pert_list = [p[1] for p in meta_data if p[0][0] != test_pert[0]]
    if len(pert_list) > 0:
        random_best = randint(0, len(pert_list) - 1)
        median_profile = np.mean(np.asarray(data[pert_list]), axis=0, keepdims=True)
        return pert_list[random_best], np.asarray([data[pert_list[random_best]]]), median_profile
    else:
        return -1, None, None


data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)
sub = "sub1/"
shutil.rmtree(sub + 'models')
os.makedirs(sub + 'models')

# data
if Path(sub + "arrays/train_data").is_file():
    print("Loading existing data")
    train_data = pickle.load(open(sub + "arrays/train_data", "rb"))
    test_data = pickle.load(open(sub + "arrays/test_data", "rb"))
    train_meta = pickle.load(open(sub + "arrays/train_meta", "rb"))
    test_meta = pickle.load(open(sub + "arrays/test_meta", "rb"))
    cell_types = pickle.load(open(sub + "arrays/cell_types", "rb"))
    meta_dictionary_pert = pickle.load(open(sub + "arrays/meta_dictionary_pert", "rb"))
    meta_dictionary_pert_test = pickle.load(open(sub + "arrays/meta_dictionary_pert_test", "rb"))
    data_dictionary_cell = pickle.load(open(sub + "arrays/data_dictionary_cell", "rb"))
    all_pert_ids = pickle.load(open(sub + "arrays/all_pert_ids", "rb"))
    train_perts = pickle.load(open(sub + "arrays/train_perts", "rb"))
    test_perts = pickle.load(open(sub + "arrays/test_perts", "rb"))
else:
    print("Parsing data")
    data, meta, all_pert_ids = parse_data("path.csv")
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
    pickle.dump(train_data, open(sub + "arrays/train_data", "wb"))
    pickle.dump(test_data, open(sub + "arrays/test_data", "wb"))
    pickle.dump(train_meta, open(sub + "arrays/train_meta", "wb"))
    pickle.dump(test_meta, open(sub + "arrays/test_meta", "wb"))
    pickle.dump(cell_types, open(sub + "arrays/cell_types", "wb"))
    pickle.dump(meta_dictionary_pert, open(sub + "arrays/meta_dictionary_pert", "wb"))
    pickle.dump(meta_dictionary_pert_test, open(sub + "arrays/meta_dictionary_pert_test", "wb"))
    pickle.dump(data_dictionary_cell, open(sub + "arrays/data_dictionary_cell", "wb"))
    pickle.dump(all_pert_ids, open(sub + "arrays/all_pert_ids", "wb"))
    pickle.dump(train_perts, open(sub + "arrays/train_perts", "wb"))
    pickle.dump(test_perts, open(sub + "arrays/test_perts", "wb"))


print("----------------------------------------------")
print(train_data.shape)
print(test_data.shape)
print("----------------------------------------------")
cell_decoders = {}
if os.path.isdir(sub + "models/main_model"):
    print("Loading model")
    autoencoder = keras.models.load_model(sub + "models/main_model",
                                          custom_objects={'correlation_coefficient_loss': correlation_coefficient_loss})
    for cell in cell_types:
        cell_decoders[cell] = pickle.load(open(sub + "models/" + cell + "_decoder_weights", "rb"))
else:
    print("Building autoencoder ")
    autoencoder = build(input_size, 1, latent_dim)
    autoencoder.save(sub + "models/main_model")

should_train = True
if should_train:
    del autoencoder
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    for e in range(nb_epoch):
        print("Total epoch " + str(e) + " ------------------------------------------------------")
        autoencoder = keras.models.load_model(sub + "models/main_model", custom_objects={
            'correlation_coefficient_loss': correlation_coefficient_loss})
        encoder = autoencoder.get_layer("encoder")
        for layer in encoder.layers:
            layer.trainable = True
        encoder.trainable = True
        autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-4))
        encoder.compile(loss="mse", optimizer=Adam(lr=1e-4))

        if e == 0:
            print("Main autoencoder" + " =========================================")
            p_train = []
            for pert_id in train_perts:
                pert_list = [p[1] for p in meta_dictionary_pert[pert_id]]
                median_profile = np.mean(np.asarray(train_data[pert_list]), axis=0)
                p_train.append(median_profile)
            p_train = np.asarray(p_train)
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            autoencoder.fit(p_train, p_train, epochs=40, batch_size=batch_size, validation_split=0.1, callbacks=[callback]) #, validation_split=0.1, callbacks=[callback]
            for cell in cell_types:
                decoder = autoencoder.get_layer("decoder")
                cell_decoders[cell] = decoder.get_weights().copy()
                pickle.dump(cell_decoders[cell], open(sub + "models/" + cell + "_decoder_weights", "wb"))
                del decoder
            for layer in encoder.layers:
                layer.trainable = True
            encoder.trainable = True
            autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-7))
            encoder.compile(loss="mse", optimizer=Adam(lr=1e-7))

        # original_main_decoder_weights = autoencoder.get_layer("decoder").get_weights()
        # for cell in cell_types:
        #     print(cell + " =========================================")
        #     c_train = np.asarray(data_dictionary_cell[cell])
        #     autoencoder.get_layer("decoder").set_weights(cell_decoders[cell])
        #     autoencoder.fit(c_train, c_train, epochs=sub_epochs, batch_size=batch_size)
        #     cell_decoders[cell] = autoencoder.get_layer("decoder").get_weights()
        #     gc.collect()
        # autoencoder.get_layer("decoder").set_weights(original_main_decoder_weights)

        # print("latent encoder")
        # for k in range(2):
        #     print("finding closest perturbations")
        #     closest_perturbations = []
        #     closest_indexes = []
        #     for i in range(len(train_data)):
        #         if i % 1000 == 0:
        #             print(str(i) + " - ", end="", flush=True)
        #         closest, profile, median_profile = get_profile(train_data, meta_dictionary_pert, train_meta[i])
        #         if closest != -1:
        #             closest_perturbations.append(train_data[i])
        #         closest_indexes.append(closest)
        #     closest_perturbations = np.asarray(closest_perturbations)
        #
        #     latent_vectors = encoder.predict(train_data)
        #     closest_profile_latent_vectors = []
        #     for i in range(len(train_data)):
        #         if closest_indexes[i] != -1:
        #             closest_profile_latent_vectors.append(latent_vectors[closest_indexes[i]])
        #     closest_profile_latent_vectors = np.asarray(closest_profile_latent_vectors)
        #     encoder.fit(closest_perturbations, closest_profile_latent_vectors, epochs=2, batch_size=batch_size)

        latent_vectors = encoder.predict(train_data[:5])
        data = [latent_vectors[0], latent_vectors[1], latent_vectors[2], latent_vectors[3],
                latent_vectors[4]]
        names = ["1", "2", "3", "4", "5"]
        fig, axes = plt.subplots(nrows=len(data), ncols=1, figsize=(14, 4))
        fig.subplots_adjust(left=None, bottom=None, right=0.85, top=None, wspace=0.4, hspace=1.4)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
        for j, ax in enumerate(axes.flatten()):
            if (j == 0):
                hm = sns.heatmap(data[j].reshape(1, latent_dim), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
                                 cbar_ax=cbar_ax, vmin=vmin, vmax=1)
            else:
                hm = sns.heatmap(data[j].reshape(1, latent_dim), linewidth=0.0, rasterized=True, cmap=cmap, ax=ax,
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
        plt.savefig(sub + "latent_vectors/latent_vector" + str(e) + ".png")
        plt.close(None)

        print("Training decoders again")
        if e == nb_epoch - 1:
            for layer in encoder.layers:
                layer.trainable = False
            encoder.trainable = False
            autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-6))
            encoder.compile(loss="mse", optimizer=Adam(lr=1e-6))

        original_main_decoder_weights = autoencoder.get_layer("decoder").get_weights()
        cl = list(cell_types)
        random.shuffle(cl)
        for cell in cl:
            print(cell + " =========================================")
            cell_data = np.asarray([[train_data[i], train_meta[i]] for i, p in enumerate(train_meta) if p[0] == cell])
            input_profiles = []
            output_profiles = []
            for i in range(len(cell_data)):
                index, profile, median_profile = get_profile(train_data, meta_dictionary_pert, cell_data[i][1])
                if median_profile is not None:
                    input_profiles.append(profile)
                    output_profiles.append(cell_data[i][0])
            if len(input_profiles) == 0:
                continue
            input_profiles = np.squeeze(np.asarray(input_profiles), axis=1)
            output_profiles = np.asarray(output_profiles)
            autoencoder.get_layer("decoder").set_weights(cell_decoders[cell])
            if e == nb_epoch - 1:
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                autoencoder.fit(input_profiles, output_profiles, epochs=100, batch_size=batch_size,
                                validation_split=0.1, callbacks=[callback])
            else:
                autoencoder.fit(input_profiles, output_profiles, epochs=1, batch_size=batch_size)

            cell_decoders[cell] = autoencoder.get_layer("decoder").get_weights()
            gc.collect()
        autoencoder.get_layer("decoder").set_weights(original_main_decoder_weights)

        print("---------------------------------------------------------------\n")
        autoencoder.save(sub + "models/main_model")
        del autoencoder
        del encoder
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("---------------------------------------------------------------\n")

    autoencoder = keras.models.load_model(sub + "models/main_model",
                                          custom_objects={'correlation_coefficient_loss': correlation_coefficient_loss})
    encoder = autoencoder.get_layer("encoder")
    autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-4))
    encoder.compile(loss="mse", optimizer=Adam(lr=1e-4))

for cell in cell_types:
    weights = cell_decoders[cell]
    c_train = np.asarray(data_dictionary_cell[cell])
    pickle.dump(weights, open(sub + "models/" + cell + "_decoder_weights", "wb"))
    autoencoder.get_layer("decoder").set_weights(weights)
    decoded = autoencoder.predict(c_train)
    print(cell + " loss is: " + str(test_loss(decoded, c_train)))

original_main_decoder_weights = autoencoder.get_layer("decoder").get_weights()

results = {}
skipped = 0
img_count = 0
test_num = len(test_data)  # len(test_data)
for i in range(test_num):
    if i % 100 == 0:
        print(str(i) + " - ", end="", flush=True)
    test_meta_object = test_meta[i]
    closest, closest_profile, median_profile = get_profile(test_data, meta_dictionary_pert_test, test_meta_object)
    if closest_profile is None:
        skipped = skipped + 1
        continue

    test_profile = np.asarray([test_data[i]])

    weights = cell_decoders[test_meta[i][0]]
    autoencoder.get_layer("decoder").set_weights(weights)
    decoded1 = autoencoder.predict(closest_profile)
    results["Our performance is: "] = results.get("Our performance is: ", 0) + test_loss(decoded1, test_profile)

    results["Our correlation is: "] = results.get("Our correlation is: ", 0) + \
                                      stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]

    decoded1 = autoencoder.predict(median_profile)
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

    autoencoder.get_layer("decoder").set_weights(weights)
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
        plt.savefig(sub + "profiles/profile" + str(i) + ".png")
        plt.close(None)
print(" Done")
for key, value in results.items():
    print(key + str(value / (test_num - skipped)))

print("skipped " + str(skipped))
print("Improvement: " + str(results["zero vector loss is: "] / results["Our performance is: "]))
