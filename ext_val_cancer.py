import gc
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use("Agg")
import cmapPy.pandasGEXpress.parse_gctx as pg
import cmapPy.pandasGEXpress.subset_gctoo as sg
import pandas as pd
import os
from tensorflow.python.keras.optimizers import Adam
import deepfake

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from scipy import stats
from tensorflow import keras
import pickle
from random import randint
from numpy import inf
import utils1
from CellData import CellData
from tensorflow.python.keras import backend as K
import tensorflow as tf


tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config1 = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def find_closest_corr(train_data, meta, input_profile, cell):
    best_corr = -1
    best_ind = -1
    for i, p in enumerate(train_data):
        if meta[i][0] != cell:
            continue
        p_corr = stats.pearsonr(p.flatten(), input_profile.flatten())[0]
        if p_corr > best_corr:
            best_corr = p_corr
            best_ind = i
    # best_corr = stats.pearsonr(train_data[best_ind].flatten(), test_profile.flatten())[0]
    return best_corr, meta[best_ind]


def read_profile(file):
    df = pd.read_csv(file, sep=",")
    profiles_trt = []
    profiles_ctrl = []
    for i in range(1, len(df.columns)):
        profile = []
        for g in genes:
            if len(df[(df['Gene_Symbol'] == g)]) != 0:
                profile.append(df[(df['Gene_Symbol'] == g)][df.columns[i]].tolist()[0])
            else:
                profile.append(0)
        profile = np.asarray(profile)
        profile = profile + 20
        # profile = 1000000 * (profile / np.max(profile))
        if df.columns[i].startswith("T"):  # in trt df[(df['Gene_Symbol'] == "HMGCR")][df.columns[i]]
            profiles_trt.append(profile)
        else:
            profiles_ctrl.append(profile)
    # all_profiles = profiles_trt.copy()
    # all_profiles.extend(profiles_ctrl)
    # all_profiles = np.asarray(all_profiles)
    # all_profiles = stats.zscore(all_profiles, axis=0)
    # all_profiles[:len(profiles_trt)]
    trt_profile = np.mean(np.asarray(profiles_trt), axis=0)
    ctrl_profile = np.mean(np.asarray(profiles_ctrl), axis=0)
    utils1.draw_one_profiles([trt_profile], len(genes), file + "trt_profiles.png")
    utils1.draw_one_profiles([ctrl_profile], len(genes), file + "ctrl_profiles.png")
    profile = np.zeros(trt_profile.shape)
    for i in range(len(genes)):
        if ctrl_profile[i] != 0 and trt_profile[i] != 0:
            #if trt_profile[i] > 10 and ctrl_profile[i] > 10:
            try:
                profile[i] = math.log(trt_profile[i] / ctrl_profile[i])
            except Exception as e:
                print(e)
    # profile = 2 * (profile - np.min(profile)) / (np.max(profile) - np.min(profile)) - 1
    profile = profile / max(np.max(profile), abs(np.min(profile)))
    utils1.draw_one_profiles([profile], len(genes), file + "profile.png")
    profile = np.expand_dims(profile, axis=-1)
    # profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile))
    return profile


def get_profile(data, meta_data, test_cell, test_pert):
    pert_list = [p[1] for p in meta_data if
                 p[0][0] == test_cell and p[0][
                     1] == test_pert]  # and p[0][2] == test_pert[2] and p[0][3] == test_pert[3]
    if len(pert_list) > 0:
        random_best = randint(0, len(pert_list) - 1)
        mean_profile = np.mean(np.asarray(data[pert_list]), axis=0, keepdims=True)
        return random_best, np.asarray([data[pert_list[random_best]]]), mean_profile, data[pert_list]
    else:
        return -1, None, None, None


def get_intersection(a, b, top_genes):
    predicted_gene_scores = []
    for i in range(978):
        predicted_gene_scores.append([genes[i], a[i]])
    predicted_gene_scores = sorted(predicted_gene_scores, key=lambda x: x[1], reverse=True)
    predicted_gene_scores = predicted_gene_scores[:top_genes]
    gene_scores = []
    for i in range(978):
        gene_scores.append([genes[i], b[i]])
    gene_scores = sorted(gene_scores, key=lambda x: x[1], reverse=True)
    gene_scores = gene_scores[:top_genes]

    top_gt = set([p[0] for p in gene_scores])
    top_predicted = set([p[0] for p in predicted_gene_scores])
    z = top_gt.intersection(top_predicted)
    return len(z)


def to_profile(df_data, cell, pert):
    indexes_trt = [i for i in range(len(meta)) if meta[i][0] == cell and
                   meta[i][1] == pert and not meta[i][2].startswith("0") and meta[i][3] == "24h"]
    indexes_ctrl = [i for i in range(len(meta)) if meta[i][0] == cell and
                    meta[i][1] == pert and meta[i][2].startswith("0")]

    trt_data = df_data.iloc[:, indexes_trt].mean(axis=1)[genes].values
    ctrl_data = df_data.iloc[:, indexes_ctrl].mean(axis=1)[genes].values
    profile = np.zeros(978)
    for i in range(len(profile)):
        if not np.isnan(trt_data[i]) and not np.isnan(ctrl_data[i]):
            try:
                profile[i] = math.log(trt_data[i] / ctrl_data[i])
            except Exception as e:
                print(e)
    profile = profile / max(np.max(profile), abs(np.min(profile)))
    # utils1.draw_one_profiles([profile], len(genes), file + "profile.png")
    profile = np.expand_dims(profile, axis=-1)
    return profile


data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)

genes = np.loadtxt("../gene_symbols.csv", dtype="str")
input_file = "../data/GSE116436_series_matrix.txt"
df_data = pd.read_csv(input_file, sep="\t", comment='!', index_col="ID_REF")
df_gpl = pd.read_csv("../data/GPL571-17391.txt", sep="\t", comment='#', index_col="ID")
affy_dict = df_gpl["Gene Symbol"].to_dict()
missed = 0
count = 0
seen = []
for key, value in affy_dict.items():
    names = str(value).split(" /// ")
    for n in names:
        if n in genes:
            if n not in seen:
                count = count + 1
                seen.append(n)
            affy_dict[key] = n
            break
    else:
        missed = missed + 1
        affy_dict[key] = min(names, key=len)
s = df_data.index.to_series()
df_data.index = s.map(affy_dict).fillna(s)
df_data = df_data[df_data.index.isin(genes)]
df_data = df_data.groupby(df_data.index).sum()

with open(input_file, 'r') as file:
    for line in file:
        if line.startswith("!Sample_title"):
            meta = line
meta = meta.replace('\n', '').replace('"', '')
meta = meta.split("\t")
del meta[0]
pert_ids = []
for i in range(len(meta)):
    meta[i] = meta[i].split("_")
    if meta[i][1] not in pert_ids:
        pert_ids.append(meta[i][1])


#cell_data = CellData("../LINCS/lincs_phase_1_2.tsv", "1", 10)
autoencoder = keras.models.load_model("best_autoencoder_1/main_model")
cell_decoders = {}
cell_decoders["MCF7"] = pickle.load(open("best_autoencoder_1/" + "MCF7" + "_decoder_weights", "rb"))
cell_decoders["PC3"] = pickle.load(open("best_autoencoder_1/" + "PC3" + "_decoder_weights", "rb"))
autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])
# print("Baseline: " + str(baseline_corr))

# closest_cor, info = find_closest_corr(cell_data.train_data, cell_data.train_meta, df_pc3, "PC3")
# print(closest_cor)
# print(info)
# closest_cor, info = find_closest_corr(cell_data.train_data, cell_data.train_meta, df_mcf7, "MCF7")
# print(closest_cor)
# print(info)

baseline_corr = 0
our_corr = 0
input_data = []
output_data = []
bdata = []
ddata = []
cdata = []
for p in pert_ids:
    df_mcf7 = to_profile(df_data, "MCF7", p)
    df_pc3 = to_profile(df_data, "PC-3", p)
    input_data.append(df_pc3)
    output_data.append(df_mcf7)
    baseline_corr = baseline_corr + stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0]
    decoded = autoencoder.predict(np.asarray([df_pc3]))
    # print(get_intersection(decoded, df_mcf7, 50))
    our_corr = our_corr + stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0]
    print(p + ":" + str(stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0])
          + " : " + str(stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0]))
    bdata.append(stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0])
    ddata.append(stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0])

pickle.dump(pert_ids, open("pert_ids.p", "wb"))
pickle.dump(bdata, open("bdata.p", "wb"))
pickle.dump(ddata, open("ddata.p", "wb"))

baseline_corr = baseline_corr / len(pert_ids)
our_corr = our_corr / len(pert_ids)
print("Baseline: " + str(baseline_corr))
print("DeepCellState: " + str(our_corr))

tcorr = 0
for i in range(len(pert_ids)):
    test_input = input_data[i]
    test_output = output_data[i]
    autoencoder = keras.models.load_model("best_autoencoder_1/main_model")
    cell_decoders = {}
    cell_decoders["MCF7"] = pickle.load(open("best_autoencoder_1/" + "MCF7" + "_decoder_weights", "rb"))
    cell_decoders["PC3"] = pickle.load(open("best_autoencoder_1/" + "PC3" + "_decoder_weights", "rb"))
    autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])

    input_tr = np.delete(np.asarray(input_data), i, axis=0)
    output_tr = np.delete(np.asarray(output_data), i, axis=0)
    # autoencoder.trainable = True
    autoencoder.fit(input_tr, output_tr, epochs=5, batch_size=1)
    decoded = autoencoder.predict(np.asarray([test_input]))
    # print(get_intersection(decoded.flatten(), test_output, 50))
    corr = stats.pearsonr(decoded.flatten(), test_output.flatten())[0]
    cdata.append(corr)
    tcorr = tcorr + corr

     # Needed to prevent Keras memory leak
    del autoencoder
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()

tcorr = tcorr / len(pert_ids)
print("DeepCellState*: " + str(tcorr))
pickle.dump(cdata, open("cdata.p", "wb"))