import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import numpy as np
from scipy import stats
from numpy import zeros
from copy import deepcopy
from scipy import stats
import pickle
import pandas as pd
from collections import Counter
from CellData import CellData
from tensorflow.python.keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)
model = "best_autoencoder_12/"
autoencoder = keras.models.load_model(model + "main_model/")
cell_decoders = {"MCF7": pickle.load(open(model + "MCF7" + "_decoder_weights", "rb")),
                 "PC3": pickle.load(open(model +  "PC3" + "_decoder_weights", "rb"))}
encoder = autoencoder.get_layer("encoder")
decoder = autoencoder.get_layer("decoder")

symbols = np.loadtxt("../gene_symbols.csv", dtype="str")

# cell_data = CellData("../LINCS/lincs_phase_1_2.tsv", "1", 10)
# pickle.dump(cell_data, open("cell_data.p", "wb"))
cell_data = pickle.load(open("cell_data.p", "rb"))

top_genes_num = 50
top_targets_num = 50
# tf_data = pickle.load(open("tf_data.p", "rb"))
# tf_sum = pickle.load(open("tf_sum.p", "rb"))
# # Hep_G2
gene_name_conversion = {'HDGFRP3':'HDGFL3', 'HIST1H2BK':'H2BC12', 'HIST2H2BE':'H2BC21', 'PAPD7':'TENT4A',
                            'NARFL':'CIAO3', 'IKBKAP':'ELP1', 'KIAA0196':'WASHC5', 'TOMM70A':'TOMM70',
                            'FAM63A':'MINDY1', 'HN1L':'JPT2', 'PRUNE':'PRUNE1', 'H2AFV':'H2AZ2', 'TMEM110':'STIMATE',
                            'KIAA1033':'WASHC4', 'ADCK3':'COQ8A', 'LRRC16A':'CARMIL1', 'FAM69A':'DIPK1A', 'WRB':'GET1',
                            'TMEM5':'RXYLT1', 'KIF1BP':'KIFBP', 'TMEM2':'CEMIP2', 'ATP5S':'DMAC2L', 'KIAA0907':'KHDC4',
                            'SQRDL':'SQOR', 'FAM57A':'TLCD3A', 'AARS':'AARS1', 'EPRS':'EPRS1'}
gene_name_conversion2 = {}
for key, value in gene_name_conversion.items():
    gene_name_conversion2[value] = key

tf_data = {"MCF7": {}, "Average": {}}
tf_sum = {"MCF7": {}, "Average": {}}
directory = "/home/user/data/DeepFake/TFS"
cell_names = set()
for filename in os.listdir(directory):
    try:
        if filename not in symbols:
            continue
        top_genes = {}
        df = pd.read_csv(os.path.join(directory, filename), sep="\t", index_col="Target_genes")
        avg_columns = []
        mcf7_columns = []
        for col in df.columns:
            if "MCF-7" not in col:
                if "Average" not in col:
                    avg_columns.append(col)
            else:
                mcf7_columns.append(col)
        if len(mcf7_columns) == 0:
            continue
        df["MCF7"] = df[mcf7_columns].astype(float).mean(axis=1)
        df["Average"] = df[avg_columns].astype(float).mean(axis=1)
        df = df.drop(mcf7_columns, 1)
        df = df.drop(avg_columns, 1)
        for i in range(len(df.index)):
            if df.index[i] in gene_name_conversion2.keys():
                df.rename(index={df.index[i]: gene_name_conversion2[df.index[i]]}, inplace=True)
        df = df[df.index.isin(symbols)]
        # remove the TF itself from the top targets
        # if filename in df.index:
        #     df.drop([filename], inplace=True)
        # pick top 50 targets
        tf_data["MCF7"][filename] = df.sort_values("MCF7", ascending=False).head(top_targets_num).index.to_list()
        tf_data["Average"][filename] = df.sort_values("Average", ascending=False).head(top_targets_num).index.to_list()
    except Exception as e:
        print(e)
pickle.dump(tf_data, open("tf_data.p", "wb"))
pickle.dump(tf_sum, open("tf_sum.p", "wb"))
tf_vals = {}
target_vals = {}

for i in range(len(cell_data.train_data)):
    if i % 100 == 0:
        print(str(i) + " - ", end="", flush=True)
    test_meta_object = cell_data.train_meta[i]
    test_profile = np.asarray([cell_data.train_data[i]])
    if test_meta_object[0] != "MCF7":
        continue
    for tf in tf_data["MCF7"].keys():
        tf_vals.setdefault(tf, []).append(test_profile[0, list(symbols.flatten()).index(tf), 0])
        sum_target = 0
        for j in range(min(1, len(tf_data["MCF7"][tf]))):
            target = tf_data["MCF7"][tf][j]
            if target == tf:
                continue
            sum_target = sum_target + test_profile[0, list(symbols.flatten()).index(target), 0]
        target_vals.setdefault(tf, []).append(sum_target)

for tf in tf_data["MCF7"].keys():
    pcc = stats.pearsonr(tf_vals[tf], target_vals[tf])[0]
    print(tf + " - " + str(pcc))

tf_chosen = list(tf_vals.keys())
tf_chosen.sort(key=lambda x: max(tf_vals[x]), reverse=True)
tf_chosen = tf_chosen[:10]
tf_mapping = {"PC3": [], "MCF7": []}
figure_vals_fold = {"MCF7": [], "Average": []}
figure_vals_num = {"MCF7": [], "Average": []}
figure_tfs = []
sum = 0
n = 0
downtf = "STAT1,PRKCQ".split(",")
for key in ["MCF7", "Average"]:  # ["PC3", "MCF7"]:
    print(key + "________________________________________________")
    autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])
    best_fit = 0
    for tf in tf_data[key].keys():
        tf_info = np.asarray(tf_vals[tf])
        if np.max(tf_info) < 0.5:
            continue
        elif tf not in figure_tfs:
            figure_tfs.append(tf)
        output_layer = autoencoder.output
        input_layer = autoencoder.input
        gene = np.where(symbols==tf)[0][0]

        loss = output_layer[:, gene, 0]
        grads = K.gradients(loss, autoencoder.input)[0]

        grads = K.l2_normalize(grads)  # normalize the gradients to help having an smooth optimization process
        func = K.function([autoencoder.input], [loss, grads])

        input_profile = 0.4 * np.random.random((1, 978, 1)) - 0.2
        for i in range(20):
            loss_val, grads_val = func([input_profile])
            input_profile += grads_val

        a = autoencoder.predict(input_profile)
        pb = np.squeeze(a)
        if tf in downtf:
            sign = 1
        else:
            sign = -1
        idx = (sign * pb).argsort()[:top_genes_num]
        top_genes = symbols[idx]
        common_genes = list(set(tf_data[key][tf]).intersection(top_genes) - {tf})
        print(tf + " " + str(len(common_genes)))
        res = len(common_genes) / ((top_targets_num / 978) * top_genes_num)
        figure_vals_fold[key].append(res)
        figure_vals_num[key].append(len(common_genes))
        sum = sum + res
        n = n + 1
    print(key)
    print(sum/n)
    print("Done")

figure_data = np.stack([figure_vals_num["Average"], figure_vals_fold["Average"],
                        figure_vals_num["MCF7"], figure_vals_fold["MCF7"]], axis=0).transpose()
df = pd.DataFrame(data=figure_data, index=figure_tfs, columns=["Average_num", "Average_fold", "MCF7_num", "MCF7_fold"])
df.index.name = "TF"
df.to_csv("../tf_analysis_data.csv")
# for k in range(128):
#     common_genes = list(set(top_genes_mapping["PC3"][k]).intersection(top_genes_mapping["MCF7"][k]))
#     print(len(common_genes))

