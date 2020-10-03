import os
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

from CellData import CellData


def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


input_size = 978
latent_dim = 128
data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)

autoencoder = keras.models.load_model("best_autoencoder_1/main_model/")
cell_decoders = {"MCF7": pickle.load(open("best_autoencoder_1/" + "MCF7" + "_decoder_weights", "rb")),
                 "PC3": pickle.load(open("best_autoencoder_1/" + "PC3" + "_decoder_weights", "rb"))}
encoder = autoencoder.get_layer("encoder")
decoder = autoencoder.get_layer("decoder")

symbols = np.loadtxt("../gene_symbols.csv", dtype="str")
cell_data = CellData("../LINCS/lincs_phase_1_2.tsv", "1", 10)

mat = []
for p in cell_data.train_data:
    v = np.abs(encoder.predict(np.asarray([p])))
    mat.append(v)

mat = np.asarray(mat)
max_vec = np.max(mat, axis=1)

importance_scores = zeros((len(cell_decoders), input_size))
importance_list = {}
for key in cell_decoders.keys():  # ["PC3", "MCF7"]:
    print(key + "________________________________________________")
    autoencoder.get_layer("decoder").set_weights(cell_decoders[key])
    for k in range(input_size):
        a = np.zeros((1, input_size, 1))
        a[0, k, 0] = 1
        v = encoder.predict(a)
        p = autoencoder.predict(a)

        max_outout_uo = np.argmax(p)
        print(str(k) + " - " + str(max_outout_uo))
        max_up_ind = np.argmax(v)
        max_down_ind = np.argmin(v)
        print(str(max_up_ind) + " - " + str(max_down_ind))
        av = np.zeros((1, 128))
        av[0, max_up_ind] = 1
        p2 = decoder.predict([a, av])
        pb = decoder.predict([np.zeros((1, input_size, 1)), av])
        up_out = np.argmax(p2)
        a[0, k, 0] = 0.5
        v = encoder.predict(a)
        max_up_ind = np.argmax(v)
        max_down_ind = np.argmin(v)
        print(str(max_up_ind) + " - " + str(max_down_ind))

        a[0, k, 0] = -0.5
        v = encoder.predict(a)
        max_up_ind = np.argmax(v)
        max_down_ind = np.argmin(v)
        print(str(max_up_ind) + " - " + str(max_down_ind))

        a[0, k, 0] = -1
        v = encoder.predict(a)
        max_up_ind = np.argmax(v)
        max_down_ind = np.argmin(v)
        print(str(max_up_ind) + " - " + str(max_down_ind))
        print("---------------------------------------")
