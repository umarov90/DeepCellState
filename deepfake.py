import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.optimizers import Adam
from tensorflow import keras
from scipy import stats
import tensorflow as tf
import gc
import pickle
import numpy as np
import random
import shutil

tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config1 = tf.config.experimental.set_memory_growth(physical_devices[0], True)

nb_total_epoch = 100
nb_autoencoder_epoch = 50
nb_frozen_epoch = 100
batch_size = 32
use_existing = False

# 2 cell  dropout 0.5, 0.8, l1 1e-7
# 7 cell and trt_sh  dropout 0.2, 0.8, l1 1e-8
# cell type excluded dropout 0.5, 0.8, l1 1e-7
# ext_val dropout 0.5, 0.9, l1 1e-5
def build(input_size, latent_dim):
    layer_units = [512, 256]
    input_shape = (input_size, 1)
    inputs = Input(shape=input_shape)
    x = inputs
    xd = Dropout(0.5, input_shape=(None, 978, 1))(x)
    x = xd
    for f in layer_units:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, activity_regularizer=regularizers.l1(1e-6))(x)
    encoder = Model(inputs, latent, name="encoder")
    latent_inputs = Input(shape=(latent_dim,))
    xd_input = Input(shape=input_shape)
    x = Dense(shape[1] * shape[2])(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)
    for f in layer_units[::-1]:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(0.9, input_shape=(None, input_size, layer_units[0]))(x)
    z = tf.keras.layers.Concatenate(axis=-1)([x, xd_input])
    x = Dense(1)(z)
    outputs = Activation("tanh")(x)
    decoder = Model([xd_input, latent_inputs], outputs, name="decoder")
    autoencoder = Model(inputs, decoder([xd, encoder(inputs)]), name="autoencoder")
    return autoencoder


def get_best_autoencoder(input_size, latent_dim, data, test_fold, n):
    best_cor = -2
    if not (use_existing and os.path.exists("best_autoencoder_" + test_fold)):
        if not os.path.exists("best_autoencoder_" + test_fold):
            os.makedirs("best_autoencoder_" + test_fold)
        for i in range(n):
            print(
                test_fold + " run number - " + str(i + 1) + " ========================================================")
            autoencoder, cell_decoders, val_cor = get_autoencoder(input_size, latent_dim, data)
            if val_cor > best_cor:
                best_cor = val_cor
                autoencoder.save("best_autoencoder_" + test_fold + "/main_model")
                for cell in data.cell_types:
                    pickle.dump(cell_decoders[cell], open("best_autoencoder_" + test_fold + "/"
                                                          + cell + "_decoder_weights", "wb"))
        print(test_fold + " best validation cor: " + str(best_cor))
    autoencoder = keras.models.load_model("best_autoencoder_" + test_fold + "/main_model")
    cell_decoders = {}
    for cell in data.cell_types:
        cell_decoders[cell] = pickle.load(open("best_autoencoder_" + test_fold + "/" + cell + "_decoder_weights", "rb"))
    return autoencoder, cell_decoders


def get_autoencoder(input_size, latent_dim, data):
    autoencoder = build(input_size, latent_dim)
    autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-4))
    encoder = autoencoder.get_layer("encoder")
    cell_decoders = {}
    count = 0
    e = 0
    if not os.path.exists("best"):
        os.makedirs("best")
    if not os.path.exists("weights"):
        os.makedirs("weights")
    while e < nb_total_epoch:
        print("Total epoch " + str(e) + " ------------------------------------------------------")
        if e > 0:
            autoencoder = keras.models.load_model("./weights/main_model")
            encoder = autoencoder.get_layer("encoder")

        if e == 0:
            print("Main autoencoder")
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            autoencoder.fit(data.train_data, data.train_data, epochs=nb_autoencoder_epoch, batch_size=batch_size,
                            validation_data=[data.val_data, data.val_data],
                            callbacks=[callback])
            for cell in data.cell_types:
                decoder = autoencoder.get_layer("decoder")
                cell_decoders[cell] = decoder.get_weights().copy()
                pickle.dump(cell_decoders[cell], open("./weights/" + cell + "_decoder_weights", "wb"))
                del decoder
        print("Training decoders")
        decoder = autoencoder.get_layer("decoder")
        cl = list(data.cell_types)
        random.shuffle(cl)
        if e == nb_total_epoch - 1:
            print("freezing encoder")
            encoder.trainable = False
            decoder.trainable = True
            autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-5))
        for cell in cl:
            # if cell not in ["MCF7", "PC3"]:
            #     continue
            print(cell)
            cell_data = np.asarray([[data.train_data[i], data.train_meta[i]]
                                    for i, p in enumerate(data.train_meta) if p[0] == cell])
            if len(cell_data) == 0:
                continue
            input_profiles = []
            output_profiles = []
            for i in range(len(cell_data)):
                input_profiles.append(cell_data[i][0])
                output_profiles.append(cell_data[i][0])
                closest, profile, mean_profile, all_profiles = data.get_profile(data.train_data,
                                                                                data.meta_dictionary_pert[
                                                                                    cell_data[i][1][1]],
                                                                                cell_data[i][1], train_data=True)
                if mean_profile is not None:
                    for p in all_profiles:
                        input_profiles.append(p)
                        output_profiles.append(cell_data[i][0])

            input_profiles = np.asarray(input_profiles)
            output_profiles = np.asarray(output_profiles)
            autoencoder.get_layer("decoder").set_weights(cell_decoders[cell])
            if e == nb_total_epoch - 1:
                cell_data_val = np.asarray([[data.val_data[i], data.val_meta[i]]
                                            for i, p in enumerate(data.val_meta) if p[0] == cell])
                input_profiles_val = []
                output_profiles_val = []
                for i in range(len(cell_data_val)):
                    closest, profile, mean_profile, all_profiles = data.get_profile(data.val_data,
                                                                                    data.meta_dictionary_pert_val[
                                                                                        cell_data_val[i][1][1]],
                                                                                    cell_data_val[i][1])
                    if mean_profile is not None:
                        for p in all_profiles:
                            input_profiles_val.append(p)
                            output_profiles_val.append(cell_data_val[i][0])
                input_profiles_val = np.asarray(input_profiles_val)
                output_profiles_val = np.asarray(output_profiles_val)
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                autoencoder.fit(input_profiles, output_profiles, epochs=nb_frozen_epoch, batch_size=batch_size,
                                validation_data=(input_profiles_val, output_profiles_val), callbacks=[callback])
            else:
                autoencoder.fit(input_profiles, output_profiles, epochs=2, batch_size=batch_size)

            cell_decoders[cell] = autoencoder.get_layer("decoder").get_weights()
            gc.collect()
        print("---------------------------------------------------------------\n")
        encoder.trainable = True
        decoder.trainable = True
        autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-4))
        autoencoder.save("weights/main_model")
        for cell in data.cell_types:
            pickle.dump(cell_decoders[cell], open("weights/" + cell + "_decoder_weights", "wb"))

        # train_cor_sum = 0.0
        # train_count = 0
        # seen_perts = []
        # for i in range(len(data.train_data)):
        #     train_meta_object = data.train_meta[i]
        #     if train_meta_object[0] not in ["MCF7", "PC3"]:
        #         continue
        #     # if train_meta_object[1] in seen_perts:
        #     #     continue
        #     closest, closest_profile, mean_profile, all_profiles = data.get_profile(data.train_data,
        #                                                                             data.meta_dictionary_pert[
        #                                                                                 train_meta_object[1]],
        #                                                                             train_meta_object, train_data=True)
        #     if closest_profile is None:
        #         continue
        #     seen_perts.append(train_meta_object[1])
        #     train_count = train_count + 1
        #     weights = cell_decoders[train_meta_object[0]]
        #     autoencoder.get_layer("decoder").set_weights(weights)
        #     decoded1 = autoencoder.predict(closest_profile)
        #     train_cor_sum = train_cor_sum + stats.pearsonr(decoded1.flatten(), data.train_data[i].flatten())[0]
        # train_cor = train_cor_sum / train_count
        # print("Training pcc: " + str(train_cor))
        # print("Evaluated:" + str(train_count))

        val_cor_sum = 0.0
        val_count = 0
        seen_perts = []
        for i in range(len(data.val_data)):
            val_meta_object = data.val_meta[i]
            if val_meta_object[0] not in ["MCF7", "PC3"]:
                continue
            # if val_meta_object[1] in seen_perts:
            #     continue
            closest, closest_profile, mean_profile, all_profiles = data.get_profile(data.val_data,
                                                                                    data.meta_dictionary_pert_val[
                                                                                        val_meta_object[1]],
                                                                                    val_meta_object)
            if closest_profile is None:
                continue
            seen_perts.append(val_meta_object[1])
            val_count = val_count + 1
            weights = cell_decoders[val_meta_object[0]]
            autoencoder.get_layer("decoder").set_weights(weights)

            predictions = []
            for p in all_profiles:
                predictions.append(autoencoder.predict(np.asarray([p])))
            special_decoded = np.mean(np.asarray(predictions), axis=0, keepdims=True)
            val_cor_sum = val_cor_sum + stats.pearsonr(special_decoded.flatten(), data.val_data[i].flatten())[0]
        val_cor = val_cor_sum / val_count
        print("Validation pcc: " + str(val_cor))
        print("Evaluated:" + str(val_count))
        if e == 0:
            best_val_cor = val_cor
        else:
            if val_cor < best_val_cor:
                count = count + 1
            else:
                best_val_cor = val_cor
                count = 0
                autoencoder.save("best/main_model")
                for cell in data.cell_types:
                    pickle.dump(cell_decoders[cell], open("best/" + cell + "_decoder_weights", "wb"))

        if count > 4:
            e = nb_total_epoch - 2
            count = 0
            for cell in data.cell_types:
                cell_decoders[cell] = pickle.load(open("best/" + cell + "_decoder_weights", "rb"))
            shutil.rmtree('weights')
            shutil.move('best', 'weights')

        # Needed to prevent Keras memory leak
        del autoencoder
        del encoder
        del decoder
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("---------------------------------------------------------------\n")
        e = e + 1

    autoencoder = keras.models.load_model("weights/main_model")
    return autoencoder, cell_decoders, val_cor
