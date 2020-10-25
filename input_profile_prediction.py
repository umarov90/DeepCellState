import os
import random

import deepfake

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import numpy as np
from numpy import zeros
from copy import deepcopy
from scipy import stats
import pickle
import pandas as pd
from collections import Counter
from CellData import CellData
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
import re
from tensorflow.python.keras import backend as K


def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


input_size = 978
latent_dim = 64
data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)

# cell_data = CellData("../LINCS/lincs_phase_1_2.tsv", "7", 10)
# pickle.dump(cell_data, open("cell_data.p", "wb"))
cell_data = pickle.load(open("cell_data.p", "rb"))

autoencoder = keras.models.load_model("best_autoencoder_1/main_model/")
cell_decoders = {}
for cell in cell_data.cell_types:
    cell_decoders[cell] = pickle.load(open("best_autoencoder_1/" + cell + "_decoder_weights", "rb"))
encoder = autoencoder.get_layer("encoder")
decoder = autoencoder.get_layer("decoder")
autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])

symbols = np.loadtxt("../gene_symbols.csv", dtype="str")

output_class = 7  # the index of the output class we want to maximize
output = autoencoder.output
input_layer = autoencoder.input
loss = K.mean(output[:, output_class, 0]) - 5.0 * K.mean(K.abs(output))
grads = K.gradients(loss, autoencoder.input)[0]  # the output of `gradients` is a list, just take the first (and only) element

grads = K.l2_normalize(grads)  # normalize the gradients to help having an smooth optimization process
func = K.function([autoencoder.input], [loss, grads])

input_img = np.random.random((1, 978, 1))  # define an initial random image
input_img = np.asarray([cell_data.test_data[0]])
lr = 0.1  # learning rate used for gradient updates
max_iter = 1000  # number of gradient updates iterations
for i in range(max_iter):
    loss_val, grads_val = func([input_img])
    input_img += grads_val * lr  # update the image based on gradients

a = autoencoder.predict(input_img)
input_img[0, output_class, 0] = 0
a2 = autoencoder.predict(input_img)
sum = np.sum(np.abs(input_img))
autoencoder_frozen = deepfake.build_frozen(978, 64)


def copy_weights(model1, source_model):
    for layer in model1.layers:
        if isinstance(layer, Model):
            copy_weights(layer, source_model.get_layer(layer.name))
        elif layer.name != "non_frozen_dense" and layer.name != "non_frozen_dense_aux":
            if len(layer.get_weights()) != 0:
                layer.trainable = False
                layer.set_weights(source_model.get_layer(layer.name).get_weights())


copy_weights(autoencoder_frozen, autoencoder)

non_frozen_layer = autoencoder_frozen.get_layer("non_frozen_dense")

autoencoder_frozen.compile(loss="mse", optimizer=Adam(lr=1e-4))
seen_perts = []
num = 0
gene_to_silence = 100
for i in range(len(cell_data.test_data)):
    if i % 100 == 0:
        print(str(i) + " - ", end="", flush=True)
    test_meta_object = cell_data.test_meta[i]
    if test_meta_object[0] != "MCF7":
        continue

    test_profile = np.asarray([cell_data.test_data[i]])
    # test_profile[0, gene_to_silence] = -1
    input_profile = np.ones((1, 978, 1))
    # input_profile[0, gene_to_silence, 0] = 0

    autoencoder_frozen.fit(input_profile, test_profile, epochs=100)
    weights = non_frozen_layer.get_weights()[0]
    pcc = stats.pearsonr(weights.flatten(), test_profile.flatten())[0]
    print(pcc)
