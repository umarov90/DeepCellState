import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from scipy import stats
import tensorflow as tf
import gc
import numpy as np
import random
import math

nb_total_epoch = 100
nb_autoencoder_epoch = 100
nb_frozen_epoch = 200
batch_size = 16
use_existing = False
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


def make_discriminator_model(input_size):
    inputs = Input(shape=(input_size, 1))
    x = inputs
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Flatten()(x)
    output = Dense(1)(x)
    model = Model(inputs, output, name="discriminator")
    return model


def make_generator_model(input_size):
    inputs = Input(shape=(input_size, 1))
    x = inputs
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    # x = Flatten()(x)
    output = Dense(1)(x)
    model = Model(inputs, output, name="generator")
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


generator_optimizer = tf.keras.optimizers.Adam(0.00001)
discriminator_optimizer = tf.keras.optimizers.Adam(0.00001)


def train_step(generator, generator2, discriminator, input_profiles, target_profiles):
    with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
        reconstruction = generator(input_profiles, training=True)
        reconstruction2 = generator2(reconstruction, training=True)

        correspondence_loss = tf.reduce_mean(
            tf.math.squared_difference(target_profiles, reconstruction))

        reconstruction_loss = tf.reduce_mean(
            tf.math.squared_difference(input_profiles, reconstruction2))

        real_output = discriminator(target_profiles, training=True)
        fake_output = discriminator(reconstruction, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)
        total_loss = correspondence_loss + reconstruction_loss + 0.01 * gen_loss

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                        discriminator.trainable_variables))
    gradients = tape.gradient(total_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def get_generators(input_size, data):
    cells = list(data.cell_types)
    generators = {}
    generators[cells[0]] = make_generator_model(input_size)
    generators[cells[1]] = make_generator_model(input_size)
    discriminators = {}
    discriminators[cells[0]] = make_discriminator_model(input_size)
    discriminators[cells[1]] = make_discriminator_model(input_size)
    count = 0
    e = 0
    while e < nb_total_epoch:
        print("Epoch " + str(e) + " ------------------------------------------------------")
        for pert in data.train_perts:
            cell = random.choice(list(data.cell_types))
            other_cell = list(data.cell_types - {cell})[0]
            input_profile = np.asarray([data.train_data[i]
                                        for i, p in enumerate(data.train_meta) if p[1] == pert and p[0] == other_cell])
            target_profile = np.asarray([data.train_data[i]
                               for i, p in enumerate(data.train_meta) if p[1] == pert and p[0] == cell])
            train_step(generators[cell], generators[other_cell], discriminators[cell], input_profile, target_profile)

        val_cor_sum = 0.0
        val_count = 0
        seen_perts = []
        disc_fake = 0
        disc_real = 0
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
            cell = val_meta_object[0]

            predictions = []
            for p in all_profiles:
                predictions.append(generators[cell].predict(np.asarray([p])))
            special_decoded = np.mean(np.asarray(predictions), axis=0)
            val_cor_sum = val_cor_sum + stats.pearsonr(special_decoded.flatten(), data.val_data[i].flatten())[0]
            if discriminators[cell].predict(special_decoded)[0, 0] > 0.5:
                disc_fake = disc_fake + 1
            if discriminators[cell].predict(np.asarray([data.val_data[i]]))[0, 0] > 0.5:
                disc_real = disc_real + 1

        val_cor = val_cor_sum / val_count
        print("Validation pcc: " + str(val_cor))
        print("Evaluated:" + str(val_count))
        print("Discriminator real correct:" + str(disc_real) + ",  fake wrongly: " + str(disc_fake))

        if e == 0:
            best_val_cor = val_cor
        else:
            if val_cor < best_val_cor:
                count = count + 1
            else:
                best_val_cor = val_cor
                count = 0

        if count > 4:
            break
        e = e + 1


    return generators