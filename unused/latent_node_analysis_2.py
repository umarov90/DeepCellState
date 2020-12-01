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
from collections import Counter
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

# cell_data = CellData("../LINCS/lincs_phase_1_2.tsv", "1", 10)
# mat = []
# for p in cell_data.train_data:
#     v = encoder.predict(np.asarray([p]))
#     mat.append(v)
#
# mat = np.asarray(mat)
# max_vec = np.squeeze(np.max(mat, axis=0))
# min_vec = np.squeeze(np.min(mat, axis=0))
# pickle.dump(max_vec, open("max_vec.p", "wb"))
# pickle.dump(min_vec, open("min_vec.p", "wb"))

max_vec = pickle.load(open("max_vec.p", "rb"))
min_vec = pickle.load(open("min_vec.p", "rb"))

tf_data = pickle.load(open("tf_data.p", "rb"))
# Hep_G2
# tf_data = {"MCF7": {}, "PC3": {}}
# directory = "/home/user/data/DeepFake/TFS"
# cell_names = set()
# for filename in os.listdir(directory):
#     try:
#         top_genes = {}
#         for cell_type in ["MCF-7", "PC-3"]:
#             df = pd.read_csv(os.path.join(directory, filename), sep="\t", index_col="Target_genes")
#             to_drop = []
#             to_merge = []
#             for col in df.columns:
#                 name = col.split("|")[-1]
#                 cell_names.add(name)
#             for col in df.columns:
#                 if cell_type not in col:
#                     to_drop.append(col)
#                 else:
#                     to_merge.append(col)
#             if len(to_merge) == 0:
#                 continue
#             df = df.drop(to_drop, 1)
#             df[cell_type] = df[to_merge].astype(float).mean(axis=1)
#             df = df.drop(to_merge, 1)
#             df = df[df.index.isin(symbols)]
#             df = df.sort_values(cell_type, ascending=False).head(50)
#             # df = df[(df.T != 0).any()]
#             top_genes[cell_type] = df.index.to_list()
#             # if len(top_genes) != 10:
#             #     continue
#         if len(top_genes.keys()) > 1:
#             for cell_type in ["MCF-7", "PC-3"]:
#                 tf_data[cell_type.replace("-", "")][filename] = top_genes[cell_type]
#     except Exception as e:
#         print(e)
# pickle.dump(tf_data, open("tf_data.p", "wb"))

tf_mapping = {"PC3": [], "MCF7": []}
top_genes_mapping = {"PC3": [], "MCF7": []}
for key in ["PC3", "MCF7"]:  # ["PC3", "MCF7"]:
    print(key + "________________________________________________")
    autoencoder.get_layer("decoder").set_weights(cell_decoders[key])
    best_fit = 0
    for k in range(128):
        top_genes = []
        #mat = []
        for g in np.linspace(min_vec[k], max_vec[k], 20):
            av = np.zeros((1, 128))
            av[0, k] = g
            pb = np.abs(np.squeeze(decoder.predict([np.zeros((1, input_size, 1)), av])))
            #mat.append(pb)
            idx = (-pb).argsort()[:50]
            top_genes.extend(symbols[idx])
        #mat = np.asarray(mat)
        #std_vec = mat.std(axis=0)
        #idx = (-std_vec).argsort()[:50]
        #top_genes = symbols[idx]

        c = Counter(top_genes)
        top_genes_tuples = c.most_common(50)
        top_genes = []
        for x, y in top_genes_tuples:
            top_genes.append(x)

        top_genes_mapping[key].append(top_genes)

        max_len = 0
        tf_gene_num = 0
        top_tf = None
        for tf, tf_top_genes in tf_data[key].items():
            common_genes = list(set(tf_top_genes).intersection(top_genes))
            if len(common_genes) > max_len:
                max_len = len(common_genes)
                top_tf = tf
                tf_gene_num = len(tf_top_genes)
        if max_len > best_fit:
            best_fit = max_len
        if max_len > 5:
            tf_mapping[key].append(top_tf)
        else:
            tf_mapping[key].append("-----")
        print(tf_mapping[key][-1])
    print(key)
print("Done")
for k in range(128):
    common_genes = list(set(top_genes_mapping["PC3"][k]).intersection(top_genes_mapping["MCF7"][k]))
    print(len(common_genes))

