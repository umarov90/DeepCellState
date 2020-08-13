import os

from DL.VAE import VAE
from DL.sampling import Sampling

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from tensorflow.keras import layers
from tensorflow import keras
from scipy import stats
import tensorflow as tf
import gc
import pickle
import numpy as np
import random
import shutil
import math

# tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config1 = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx('float64')

nb_total_epoch = 100
nb_autoencoder_epoch = 50
nb_frozen_epoch = 200
batch_size = 32
use_existing = False
use_kl = False
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
    #autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
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


def make_discriminator_model(input_size):
    layer_units = [512, 256]
    inputs = Input(shape=(input_size, 1))
    x = inputs
    # x = Dropout(0.5, input_shape=(None, 978, 1))(x)
    for f in layer_units:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    # x = Dropout(0.3, input_shape=(None, input_size, layer_units[1]))(x)
    x = Flatten()(x)
    output = Dense(1)(x)
    model = Model(inputs, output, name="discriminator")
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# @tf.function
def train_step(input_data, output_profiles, generator, discriminator, epoch):
    gan_epochs = 2
    gen_learning_start_epoch = 3
    if epoch > gen_learning_start_epoch:
        gan_epochs = 21
    for d in range(gan_epochs):
        fake_data_old = generator.predict(input_data)
        total = int(math.ceil(float(len(input_data)) / batch_size))
        for i in range(total):
            input_data = input_data[i * batch_size:(i + 1) * batch_size]
            output_data = output_profiles[i * batch_size:(i + 1) * batch_size]
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_data = generator(input_data, training=True)

                real_output = discriminator(output_data, training=True)
                fake_output = discriminator(generated_data, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer = tf.keras.optimizers.Adam(0.00001)  # even smaller??
            discriminator_optimizer = tf.keras.optimizers.Adam(0.0005)

            if epoch > gen_learning_start_epoch and d % 5 == 0 and d != 0:
                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            # old_weights = discriminator.get_weights()[0]
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            # (discriminator.get_weights()[0] == old_weights).all()
        fake_data = generator.predict(input_data)
        r = 0
        f_new = 0
        f_old = 0
        a = discriminator.predict(output_profiles)
        for v in a:
            if v > 0.5:
                r = r + 1

        a = discriminator.predict(fake_data)
        for v in a:
            if v > 0.5:
                f_new = f_new + 1

        a = discriminator.predict(fake_data_old)
        for v in a:
            if v > 0.5:
                f_old = f_old + 1
        print(str(d) + " discriminator " + str(r) + " : " + str(f_old) + " : " + str(f_new) + " - " + str(len(input_data)))


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def get_autoencoder(input_size, latent_dim, data):
    learning_rate = 0.0001
    autoencoder = build(input_size, latent_dim)
    autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate))
    encoder = autoencoder.get_layer("encoder")
    cell_decoders = {}
    cell_discriminators = {}
    discriminator = make_discriminator_model(input_size)
    for cell in data.cell_types:
        cell_discriminators[cell] = discriminator.get_weights().copy()
    count = 0
    e = 0
    if not os.path.exists("best"):
        os.makedirs("best")
    if not os.path.exists("weights"):
        os.makedirs("weights")
    while e < nb_total_epoch:
        print("Total epoch " + str(e) + " ------------------------------------------------------")
        if e > 0:
            autoencoder_saved = keras.models.load_model("./weights/main_model")
            autoencoder = build(input_size, latent_dim)
            autoencoder.set_weights(autoencoder_saved.get_weights())
            autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate))
            del autoencoder_saved
            discriminator = make_discriminator_model(input_size)
            encoder = autoencoder.get_layer("encoder")

        if e == 0:
            print("Main autoencoder")
            # autoencoder = keras.models.load_model("default_autoencoder")
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
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
        cl = list(data.cell_types)
        random.shuffle(cl)
        if e == nb_total_epoch - 1:
            print("freezing encoder")
            encoder.trainable = False
            decoder.trainable = True
            autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate))
        for cell in cl:
            print(cell)
            tf.random.set_seed(1)
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
            decoder.set_weights(cell_decoders[cell])
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
            tf.random.set_seed(1)
            if e != nb_total_epoch - 1:
                discriminator.set_weights(cell_discriminators[cell])
                z_mean, z_log_var, z = encoder.predict(input_profiles)
                train_step(z, output_profiles, decoder, discriminator, e)
                cell_discriminators[cell] = discriminator.get_weights()
            cell_decoders[cell] = decoder.get_weights()
            gc.collect()
        print("---------------------------------------------------------------\n")
        autoencoder.save("weights/main_model")
        for cell in data.cell_types:
            pickle.dump(cell_decoders[cell], open("weights/" + cell + "_decoder_weights", "wb"))

        # train_cor_sum = 0.0
        # train_count = 0
        # seen_perts = []
        # for i in range(len(data.train_data)):
        #     train_meta_object = data.train_meta[i]
        #     if train_meta_object[1] in seen_perts:
        #         continue
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

        if count > 40:
            e = nb_total_epoch - 2
            count = 0
            for cell in data.cell_types:
                cell_decoders[cell] = pickle.load(open("best/" + cell + "_decoder_weights", "rb"))
            shutil.rmtree('weights')
            shutil.move('best', 'weights')

        # Needed to prevent Keras memory leak
        del autoencoder
        del encoder
        del discriminator
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("---------------------------------------------------------------\n")
        e = e + 1

    autoencoder = keras.models.load_model("weights/main_model")
    return autoencoder, cell_decoders, val_cor
