import os

from figures import profiles_viz
from competitors.VAE import VAE
from unused.sampling import Sampling

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
from scipy import stats
import tensorflow as tf
import pickle
import numpy as np
import random
import shutil

# tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config1 = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx('float64')

nb_total_epoch = 100
nb_autoencoder_epoch = 40
batch_size = 64
use_existing = False
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def build(input_size, latent_dim):
    layer_units = [256, 128]
    input_shape = (input_size, 1)
    drop_rate = 0.8
    inputs = Input(shape=input_shape)
    x = inputs
    x = Dropout(0.5, input_shape=(None, 978, 1))(x)
    for f in layer_units:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(drop_rate, input_shape=(None, input_size, layer_units[1]))(x)
    shape = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(shape[1] * shape[2])(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)
    for f in layer_units[::-1]:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(drop_rate, input_shape=(None, input_size, layer_units[0]))(x)
    x = Dense(1)(x)
    outputs = x
    # outputs = Activation("tanh")(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    # autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    vae = VAE(encoder, decoder, name="autoencoder")
    return vae


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



autoencoder_optimizer = tf.keras.optimizers.Adam(0.0001)


# @tf.function
def train_step(autoencoder, pert_profiles, target_profiles):
    with tf.GradientTape() as tape:
        z_mean, z_log_var, z = autoencoder.get_layer("encoder")(pert_profiles, training=True)
        reconstruction = autoencoder.get_layer("decoder")(z, training=True)

        reconstruction_loss = tf.reduce_mean(
            tf.math.squared_difference(target_profiles, reconstruction))
        same_pert_loss = tf.reduce_mean(tf.math.squared_difference(z[0], z[1]))
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = 0.05 * same_pert_loss + 0.005 * kl_loss + reconstruction_loss

    gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
    autoencoder_optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

def get_autoencoder(input_size, latent_dim, data):
    learning_rate = 0.00001
    df = pd.read_csv("../data/GSE70138_Broad_LINCS_pert_info.txt", sep="\t")
    autoencoder = build(input_size, latent_dim)
    autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate))
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
        # if e > 0:
        #     autoencoder_saved = keras.models.load_model("./weights/main_model")
        #     autoencoder = build(input_size, latent_dim)
        #     autoencoder.set_weights(autoencoder_saved.get_weights())
        #     autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate))
        #     del autoencoder_saved
        #     encoder = autoencoder.get_layer("encoder")

        if e == 0:
            print("Main autoencoder")
            # autoencoder = keras.models.load_model("default_autoencoder")
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            autoencoder.fit(data.train_data, data.train_data, epochs=nb_autoencoder_epoch, batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[callback])
            autoencoder.save("default_autoencoder")
            for cell in data.cell_types:
                decoder = autoencoder.get_layer("decoder")
                cell_decoders[cell] = decoder.get_weights().copy()
                pickle.dump(cell_decoders[cell], open("./weights/" + cell + "_decoder_weights", "wb"))
                del decoder
        print("Training decoders")
        decoder = autoencoder.get_layer("decoder")
        count_im = 0
        if not os.path.exists("vectors/"):
            os.makedirs("vectors/")
        for pert in data.train_perts:
            pname = profiles_viz.fix(df.query('pert_id=="' + str(pert) + '"')["pert_iname"].tolist()[0])
            cell = random.choice(list(data.cell_types))
            decoder.set_weights(cell_decoders[cell])
            pert_profiles = np.asarray([data.train_data[i]
                                        for i, p in enumerate(data.train_meta) if p[1] == pert])
            cell_names = np.asarray([p[0] for i, p in enumerate(data.train_meta) if p[1] == pert])
            target_profiles = [data.train_data[i]
                               for i, p in enumerate(data.train_meta) if p[1] == pert and p[0] == cell]
            while len(target_profiles) < len(pert_profiles):
                target_profiles.append(target_profiles[0])
            target_profiles = np.asarray(target_profiles)
            if count_im < 5:
                z_mean, z_log_var, z = encoder.predict(pert_profiles)
                profiles_viz.draw_vectors(z, "vectors/" + pname + "_1.png", cell_names)

            train_step(autoencoder, pert_profiles, target_profiles)
            if count_im < 5:
                z_mean, z_log_var, z = encoder.predict(pert_profiles)
                profiles_viz.draw_vectors(z, "vectors/" + pname + "_2.png", cell_names)
            count_im = count_im + 1
            cell_decoders[cell] = decoder.get_weights().copy()

        print("---------------------------------------------------------------\n")

        val_cor_sum = 0.0
        val_count = 0
        seen_perts = []
        for i in range(len(data.val_data)):
            val_meta_object = data.val_meta[i]
            if val_meta_object[1] in seen_perts:
                continue
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
            special_decoded = np.mean(np.asarray(predictions), axis=0)
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
            for cell in data.cell_types:
                cell_decoders[cell] = pickle.load(open("best/" + cell + "_decoder_weights", "rb"))
            shutil.rmtree('weights')
            shutil.move('best', 'weights')
            autoencoder.save("weights/main_model")
            break

        autoencoder.save("weights/main_model")
        for cell in data.cell_types:
            pickle.dump(cell_decoders[cell], open("weights/" + cell + "_decoder_weights", "wb"))

        # Needed to prevent Keras memory leak
        # del autoencoder
        # del encoder
        # gc.collect()
        # K.clear_session()
        # tf.compat.v1.reset_default_graph()
        print("---------------------------------------------------------------\n")
        e = e + 1

    autoencoder = keras.models.load_model("weights/main_model")
    return autoencoder, cell_decoders, val_cor