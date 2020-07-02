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
nb_epoch = 400
batch_size = 32
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


def build(input_size, channels, latent_dim, filters=(128, 256), encoder=None):
    filter_len = 3
    input_shape = (input_size, channels)
    # define the input to the encoder
    inputs = Input(shape=input_shape)
    x = inputs
    # loop over the number of filters
    for f in filters:
        x = Conv1D(f, filter_len, strides=1, padding="same", use_bias=True)(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    # Shape info needed to build Decoder Model
    shape = K.int_shape(x)
    x = Flatten()(x)
    print(x)
    x = Dropout(0.8, input_shape=(None, 500736))(x)
    latent = Dense(latent_dim)(x)
    print(latent)
    latent = Dropout(0.5, input_shape=(None, 64))(latent)
    #x = Dropout(0.5, input_shape=(None, 978, 256))(x)
    # latent = BatchNormalization()(latent)
    # build the encoder model
    encoder = Model(inputs, latent, name="encoder")
    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(shape[1] * shape[2])(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)

    # loop over our number of filters again, but this time in
    # reverse order
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv1DTranspose(f, filter_len, strides=1, padding="same", use_bias=True)(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    # apply a single CONV_TRANSPOSE layer used to recover the
    # original depth of the image
    ###########################################################################################################
    # Maybe her BN as well
    ###########################################################################################################
    x = Conv1DTranspose(1, 1, padding="same")(x)
    # outputs = LeakyReLU(alpha=0.2)(x)
    # outputs = x
    outputs = LeakyReLU(alpha=0.2)(x)
    # outputs = BatchNormalization()(outputs)
    # build the decoder model
    decoder = Model(latent_inputs, outputs, name="decoder")
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
    print("Parsing data at " + file)
    df = pd.read_csv(file, sep="\t")
    df.reset_index(drop=True, inplace=True)
    try:
        df = df.drop('Unnamed: 0', 1)
    except Exception as e:
        pass
    try:
        df = df.drop('distil_id', 1)
    except Exception as e:
        pass
    print("Total: " + str(df.shape))
    df = df[(df['cell_id'] == "MCF7") | (df['cell_id'] == "PC3")]
    # df = df[(df['pert_type'] == "trt_cp") | (df['pert_type'] == "trt_sh") |
    #         (df['pert_type'] == "trt_sh.cgs") | (df['pert_type'] == "trt_sh.css") |
    #         (df['pert_type'] == "trt_oe") | (df['pert_type'] == "trt_lig")]
    # df = df[(df['pert_type'] == "trt_cp")]
    print("Cell filtering: " + str(df.shape))
    df = df.groupby(['cell_id', 'pert_id']).filter(lambda x: len(x) > 1)
    print("Pert filtering: " + str(df.shape))
    # df = df.drop_duplicates(['cell_id', 'pert_id', 'pert_idose', 'pert_itime', 'pert_type'])
    # df = df.groupby(['cell_id', 'pert_id'], as_index=False).mean()
    # df = df.groupby(['cell_id', 'pert_id', 'pert_type'], as_index=False).mean()  # , 'pert_type'
    # print("Merging: " + str(df.shape))
    #df.pert_type.value_counts().to_csv("trt_count_final.tsv", sep='\t')
    cell_ids = df["cell_id"].values
    pert_ids = df["pert_id"].values
    all_pert_ids = set(pert_ids)
    pert_idose = df["pert_idose"].values
    pert_itime = df["pert_itime"].values
    #pert_type = df["pert_type"].values
    perts = np.stack([cell_ids, pert_ids, pert_idose, pert_itime]).transpose()  # , pert_idose, pert_itime, pert_type
    df = df.drop(['cell_id', 'pert_id', 'pert_idose', 'pert_itime'],
                 1)  # , 'pert_idose', 'pert_itime', 'pert_type'
    try:
        df = df.drop('cid', 1)
    except Exception as e:
        pass
    data = df.values
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # for i in range(len(data)):
    #    data[i] = data[i] / max(np.max(data[i]), abs(np.min(data[i])))
    #data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data))  - 1
    data = np.expand_dims(data, axis=-1)
    return data, perts, all_pert_ids


def split_data(data, meta, all_pert_ids):
    cell_types = set([meta[i][0] for i, x in enumerate(meta)])
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(meta)
    all_pert_ids_list = list(all_pert_ids)
    shuffle(all_pert_ids_list)

    train_perts = all_pert_ids_list[:int(0.8 * len(all_pert_ids_list))]
    val_perts = all_pert_ids_list[int(0.8 * len(all_pert_ids_list)):int(0.9 * len(all_pert_ids_list))]
    test_perts = all_pert_ids_list[int(0.9 * len(all_pert_ids_list)):]

    train_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in train_perts])
    test_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in test_perts])
    val_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in val_perts])
    train_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in train_perts])
    test_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in test_perts])
    val_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in val_perts])

    return train_data, train_meta, test_data, test_meta, val_data, val_meta, cell_types, train_perts, val_perts, test_perts


def get_duration(param):
    vals = param.split(" ")
    duration = float(vals[0])
    if vals[1] == "m":
        duration = duration * 60
    elif vals[1] == "h":
        duration = duration * 60 * 60
    return duration


def get_profile(data, meta_data, test_pert, train_mode=False):
    pert_list = [p[1] for p in meta_data if
                 p[0][0] != test_pert[0]]  # and p[0][2] == test_pert[2] and p[0][3] == test_pert[3]
    if len(pert_list) > 0:
        random_best = randint(0, len(pert_list) - 1)
        mean_profile = np.mean(np.asarray(data[pert_list]), axis=0, keepdims=True)
        mean_profile = (mean_profile - np.min(mean_profile)) / (np.max(mean_profile) - np.min(mean_profile))
        return np.asarray([data[pert_list[random_best]]]), mean_profile, data[pert_list]
    else:
        return None, None, None


def get_all_profiles_pert(data, meta_data, test_pert):
    pert_list = [p[1] for p in meta_data if p[0][0] == test_pert[0]]
    if len(pert_list) > 0:
        mean_profile = np.mean(np.asarray(data[pert_list]), axis=0, keepdims=True)
        mean_profile = (mean_profile - np.min(mean_profile)) / (np.max(mean_profile) - np.min(mean_profile))
        return mean_profile
    else:
        return None


def find_closest_corr(train_data, input_profile, test_profile):
    best_corr = -1
    best_ind = -1
    for i, p in enumerate(train_data):
        p_corr = stats.pearsonr(p.flatten(), input_profile.flatten())[0]
        if p_corr > best_corr:
            best_corr = p_corr
            best_ind = i
    # best_corr = stats.pearsonr(train_data[best_ind].flatten(), test_profile.flatten())[0]
    return best_corr


data_folder = "/home/user/data/DeepFake/sub1/"
os.chdir(data_folder)
shutil.rmtree('models')
os.makedirs('models')
shutil.rmtree('arrays')
os.makedirs('arrays')

# data_sh, meta_sh, all_pert_ids_sh = parse_data("lincs_trt_cp_phase_1.tsv")
# rng_state = np.random.get_state()
# np.random.shuffle(data_sh)
# np.random.set_state(rng_state)
# np.random.shuffle(meta_sh)
# pickle.dump(data_sh, open("arrays/data_sh", "wb"))
# pickle.dump(meta_sh, open("arrays/meta_sh", "wb"))

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
    data, meta, all_pert_ids = parse_data("../lincs_trt_cp_phase_2.tsv")
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
    autoencoder = keras.models.load_model("./models/main_model",
                                          custom_objects={'correlation_coefficient_loss': correlation_coefficient_loss})
    for cell in cell_types:
        cell_decoders[cell] = pickle.load(open("./models/" + cell + "_decoder_weights", "rb"))
else:
    print("Building autoencoder ")
    autoencoder = build(input_size, 1, latent_dim)
    autoencoder.save("./models/main_model")

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
        autoencoder = keras.models.load_model("./models/main_model", custom_objects={
            'correlation_coefficient_loss': correlation_coefficient_loss})
        encoder = autoencoder.get_layer("encoder")
        encoder.trainable = True
        autoencoder.compile(loss="mse", optimizer=Adam(lr=4e-5))

        if e == 0:
            print("Main autoencoder" + " =========================================")
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            denoised_data = []
            for p in train_meta:
                denoised_data.append(get_all_profiles_pert(train_data, meta_dictionary_pert[p[1]], p))
            denoised_data = np.asarray(denoised_data)
            denoised_data = np.squeeze(denoised_data, axis=1)
            autoencoder.fit(train_data, denoised_data, epochs=4, batch_size=batch_size, validation_split=0.1,
                            callbacks=[callback])  # , validation_split=0.1, callbacks=[callback]
            for cell in cell_types:
                decoder = autoencoder.get_layer("decoder")
                cell_decoders[cell] = decoder.get_weights().copy()
                pickle.dump(cell_decoders[cell], open("./models/" + cell + "_decoder_weights", "wb"))
                del decoder

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
        plt.savefig("latent_vectors/latent_vector" + str(e) + ".png")
        plt.close(None)

        print("Training decoders again")
        decoder = autoencoder.get_layer("decoder")

        if e == nb_epoch - 1:
            lr = 1e-5
            encoder.trainable = False
            decoder.trainable = True
            autoencoder.compile(loss="mse", optimizer=Adam(lr=lr))

        original_main_decoder_weights = autoencoder.get_layer("decoder").get_weights()
        cl = list(cell_types)
        # random.shuffle(cl)
        for cell in cl:
            print(cell + " =========================================")
            cell_data = np.asarray([[train_data[i], train_meta[i]] for i, p in enumerate(train_meta) if p[0] == cell])
            input_profiles = []
            output_profiles = []
            for i in range(len(cell_data)):
                profile, median_profile, all_profiles = get_profile(train_data,
                                                                    meta_dictionary_pert[cell_data[i][1][1]],
                                                                    cell_data[i][1], train_mode=True)
                denoised_profile = get_all_profiles_pert(train_data,
                                                                    meta_dictionary_pert[cell_data[i][1][1]],
                                                                    cell_data[i][1])
                if median_profile is not None:
                    for p in all_profiles:
                        input_profiles.append(p)
                        output_profiles.append(denoised_profile)


            input_profiles = np.asarray(input_profiles)
            output_profiles = np.asarray(output_profiles)
            output_profiles = np.squeeze(output_profiles, axis=1)
            autoencoder.get_layer("decoder").set_weights(cell_decoders[cell])
            if e == nb_epoch - 1:
                cell_data_val = np.asarray([[val_data[i], val_meta[i]] for i, p in enumerate(val_meta) if p[0] == cell])
                input_profiles_val = []
                output_profiles_val = []
                for i in range(len(cell_data_val)):
                    profile, median_profile, all_profiles = get_profile(val_data,
                                                                        meta_dictionary_pert_val[cell_data_val[i][1][1]],
                                                                        cell_data_val[i][1])
                    denoised_profile = get_all_profiles_pert(val_data, meta_dictionary_pert_val[cell_data_val[i][1][1]],
                                                             cell_data_val[i][1])
                    if median_profile is not None:
                        for p in all_profiles:
                            input_profiles_val.append(p)
                            output_profiles_val.append(denoised_profile)
                input_profiles_val = np.asarray(input_profiles_val)
                output_profiles_val = np.asarray(output_profiles_val)
                output_profiles_val = np.squeeze(output_profiles_val, axis=1)
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                autoencoder.fit(input_profiles, output_profiles, epochs=400, batch_size=batch_size,
                                validation_data=(input_profiles_val, output_profiles_val), callbacks=[callback])
            else:
                autoencoder.fit(input_profiles, output_profiles, epochs=1, batch_size=batch_size)

            cell_decoders[cell] = autoencoder.get_layer("decoder").get_weights()
            gc.collect()
        autoencoder.get_layer("decoder").set_weights(original_main_decoder_weights)

        print("---------------------------------------------------------------\n")
        autoencoder.save("./models/main_model")
        for cell in cell_types:
            pickle.dump(cell_decoders[cell], open("./models/" + cell + "_decoder_weights", "wb"))
        val_cor_sum = 0.0
        val_cor_sum_hard = 0.0
        val_count = 0
        seen_perts = []
        skipped = 0
        for i in range(len(val_data)):
            val_meta_object = val_meta[i]
            if val_meta_object[1] in seen_perts:
                continue
            seen_perts.append(val_meta_object[1])
            val_count = val_count + 1
            closest_profile, median_profile, all_profiles = get_profile(val_data, meta_dictionary_pert_val[val_meta_object[1]],
                                                     val_meta_object)
            if closest_profile is None:
                skipped = skipped + 1
                continue
            denoised_profile = get_all_profiles_pert(val_data, meta_dictionary_pert_val[val_meta_object[1]],
                                                     val_meta_object)
            if denoised_profile is None:
                continue
            weights = cell_decoders[val_meta_object[0]]
            autoencoder.get_layer("decoder").set_weights(weights)
            decoded1 = autoencoder.predict(closest_profile)
            val_cor_sum = val_cor_sum + stats.pearsonr(decoded1.flatten(), denoised_profile.flatten())[0]
            val_cor_sum_hard = val_cor_sum_hard + stats.pearsonr(decoded1.flatten(), val_data[i].flatten())[0]
        val_cor = val_cor_sum / val_count
        print("Validation pcc: " + str(val_cor))
        print("Validation pcc: " + str(val_cor_sum_hard / val_count))
        print("Evaluated:" + str(val_count))
        print("Skipped:" + str(skipped))
        if e == 0:
            best_val_cor = val_cor
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
results_groups = {}
results_groups["trt_cp"] = {}
results_groups["trt_sh"] = {}
results_groups["trt_sh.cgs"] = {}
results_groups["trt_sh.css"] = {}
results_groups["trt_oe"] = {}
results_groups["trt_lig"] = {}
for i in range(test_num):
    if i % 100 == 0:
        print(str(i) + " - ", end="", flush=True)
    test_meta_object = test_meta[i]
    closest_profile, median_profile, all_profiles = get_profile(test_data,
                                                                meta_dictionary_pert_test[test_meta_object[1]],
                                                                test_meta_object)
    if closest_profile is None:
        skipped = skipped + 1
        continue
    if test_meta_object[1] in seen_perts:
        skipped = skipped + 1
        continue
    seen_perts.append(test_meta_object[1])

    test_profile = get_all_profiles_pert(test_data, meta_dictionary_pert_test[test_meta_object[1]],
                                         test_meta_object)

    # closest_cor = closest_cor + find_closest_corr(train_data, closest_profile, test_profile)

    weights = cell_decoders[test_meta[i][0]]
    autoencoder.get_layer("decoder").set_weights(weights)
    decoded1 = autoencoder.predict(closest_profile)
    if test_meta_object[2] not in results_groups.keys():
        skipped = skipped + 1
        continue
    results = results_groups[test_meta_object[4]]
    results["count"] = results.get("count", 0) + 1
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

    all_results.append(str(stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]) + ", " +
                       str(stats.pearsonr(closest_profile.flatten(), test_profile.flatten())[0]) + ", "
                       + test_meta_object[0] + ", " + test_meta_object[1])

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
        plt.savefig("profiles/profile" + str(i) + ".png")
        plt.close(None)
print(" Done")

for key1, value1 in results_groups.items():
    if "count" not in value1.keys():
        continue
    print(key1 + " - " + str(value1["count"]))
    for key, value in value1.items():
        if key == "count":
            continue
        print(key + str(value / value1["count"]))

# print("closest train corr:" + str(closest_cor / (test_num - skipped)))

with open("all_results_per_drug.csv", 'w+') as f:
    f.write('\n'.join(all_results))
print("skipped " + str(skipped))

#
# results = {}
# skipped = 0
# test_num = len(data_sh)
# seen_perts = []
# closest_cor = 0
# for i in range(test_num):
#     if i % 100 == 0:
#         print(str(i) + " - ", end="", flush=True)
#     test_meta_object = meta_sh[i]
#     if test_meta_object[1] in seen_perts:
#         skipped = skipped + 1
#         continue
#
#     closest_profile, median_profile, all_profiles = get_profile(data_sh, meta_sh, test_meta_object)
#
#     if closest_profile is None:
#         skipped = skipped + 1
#         continue
#     seen_perts.append(test_meta_object[1])
#
#     test_profile = np.asarray([data_sh[i]])
#
#     closest_cor = closest_cor + find_closest_corr(train_data, closest_profile, test_profile)
#
#     weights = cell_decoders[meta_sh[i][0]]
#     autoencoder.get_layer("decoder").set_weights(weights)
#     decoded1 = autoencoder.predict(closest_profile)
#
#     results["Our correlation: "] = results.get("Our correlation: ", 0) + \
#                                    stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]
#     results["Baseline correlation: "] = results.get("Baseline correlation: ", 0) + \
#                                     stats.pearsonr(closest_profile.flatten(), test_profile.flatten())[0]
#
# print(" Done")
# for key, value in results.items():
#     print(key + str(value / (test_num - skipped)))
#
# print("closest train corr:" + str(closest_cor / (test_num - skipped)))
#
# print("skipped " + str(skipped))
