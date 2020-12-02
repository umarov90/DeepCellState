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
from scipy.stats import zscore

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from scipy import stats
from tensorflow import keras
import pickle
from random import randint
from numpy import inf
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
    # profile = profile / max(np.max(profile), abs(np.min(profile)))
    # utils1.draw_one_profiles([profile], len(genes), file + "profile.png")
    profile = np.expand_dims(profile, axis=-1)
    return profile


os.chdir(open("../data_dir").read().strip())

genes = np.loadtxt("gene_symbols.csv", dtype="str")
input_file = "data/GSE116436_series_matrix.txt"
df_data = pd.read_csv(input_file, sep="\t", comment='!', index_col="ID_REF")
df_gpl = pd.read_csv("data/GPL571-17391.txt", sep="\t", comment='#', index_col="ID")
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
model = "sub2/best_autoencoder_ext_val/"
autoencoder = keras.models.load_model(model + "main_model/")
cell_decoders = {"MCF7": pickle.load(open(model + "MCF7" + "_decoder_weights", "rb")),
                 "PC3": pickle.load(open(model +  "PC3" + "_decoder_weights", "rb"))}
autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])

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
    # baseline_corr = baseline_corr + stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0]
    # decoded = autoencoder.predict(np.asarray([df_pc3]))
    # # print(get_intersection(decoded, df_mcf7, 50))
    # our_corr = our_corr + stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0]
    # print(p + ":" + str(stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0])
    #       + " : " + str(stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0]))
    # bdata.append(stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0])
    # ddata.append(stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0])

for i in range(len(input_data)):
    output_data[i] = (output_data[i] - np.mean(np.asarray(output_data), axis=0)) / np.std(np.asarray(output_data), axis=0)
    input_data[i] = (input_data[i] - np.mean(np.asarray(input_data), axis=0)) / np.std(np.asarray(input_data), axis=0)
    output_data[i][np.isnan(output_data[i])] = 0
    input_data[i][np.isnan(input_data[i])] = 0


# for i, p in enumerate(pert_ids):
#     output_data[i] = output_data[i] / np.max(np.abs(np.asarray(output_data)))
#     input_data[i] = input_data[i] / np.max(np.abs(np.asarray(input_data)))

# p2 = np.mean(np.asarray(output_data), axis=0)
# p2 = matrix.std(axis=0)
# utils1.draw_vectors([p1, p2], "ext_val_info.png", names=["Mean", "SD"])
for i, p in enumerate(pert_ids):
    df_mcf7 = output_data[i]
    df_pc3 = input_data[i]
    baseline_corr = baseline_corr + stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0]
    decoded = autoencoder.predict(np.asarray([df_pc3]))
    # print(get_intersection(decoded, df_mcf7, 50))
    our_corr = our_corr + stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0]
    print(p + ":" + str(stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0])
          + " : " + str(stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0]))
    bdata.append(stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0])
    ddata.append(stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0])

baseline_corr = baseline_corr / len(pert_ids)
our_corr = our_corr / len(pert_ids)
print("Baseline: " + str(baseline_corr))
print("DeepCellState: " + str(our_corr))
exit()
tcorr = 0
tcorrb = 0
for i in range(len(pert_ids)):
    test_input = input_data[i]
    test_output = output_data[i]
    autoencoder = keras.models.load_model(model + "main_model/")
    cell_decoders = {}
    cell_decoders["MCF7"] = pickle.load(open(model + "MCF7" + "_decoder_weights", "rb"))
    cell_decoders["PC3"] = pickle.load(open(model + "PC3" + "_decoder_weights", "rb"))
    autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])

    input_tr = np.delete(np.asarray(input_data), i, axis=0)
    output_tr = np.delete(np.asarray(output_data), i, axis=0)
    # autoencoder.trainable = True
    autoencoder.fit(input_tr, output_tr, epochs=25, batch_size=3)
    decoded = autoencoder.predict(np.asarray([test_input]))
    # print(get_intersection(decoded.flatten(), test_output, 50))
    corr = stats.pearsonr(decoded.flatten(), test_output.flatten())[0]
    cdata.append(corr)
    tcorr = tcorr + corr

    autoencoder = deepfake.build(978, 64)
    autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-4))
    input_tr = np.delete(np.asarray(input_data), i, axis=0)
    output_tr = np.delete(np.asarray(output_data), i, axis=0)
    autoencoder.fit(input_tr, output_tr, epochs=25, batch_size=3)
    decoded = autoencoder.predict(np.asarray([test_input]))
    corr = stats.pearsonr(decoded.flatten(), test_output.flatten())[0]
    # cdata.append(corr)
    tcorrb = tcorrb + corr

    # Needed to prevent Keras memory leak
    del autoencoder
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()

tcorr = tcorr / len(pert_ids)
print("DeepCellState*: " + str(tcorr))
tcorrb = tcorrb / len(pert_ids)
print("DeepCellState*b: " + str(tcorrb))

df = pd.DataFrame(list(zip(bdata, ddata, cdata)),
                  columns=['Baseline', 'DeepCellState', "DeepCellState*"], index=pert_ids)
df.to_csv("figures_data/cancer_drugs.tsv", sep="\t")