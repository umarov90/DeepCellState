import gc
import math
import pandas as pd
import os
import deepfake
import numpy as np
from scipy import stats
from tensorflow import keras
import pickle
import tensorflow as tf
from scipy.stats import ttest_ind
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config1 = tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
    profile = np.expand_dims(profile, axis=-1)
    return profile


os.chdir(open("data_dir").read().strip())

genes = np.loadtxt("data/gene_symbols.csv", dtype="str")
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


model = "best_autoencoder_ext_val/"
autoencoder = keras.models.load_model(model + "main_model/")
cell_decoders = {"MCF7": pickle.load(open(model + "MCF7" + "_decoder_weights", "rb")),
                 "PC3": pickle.load(open(model + "PC3" + "_decoder_weights", "rb"))}
autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])

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

for i in range(len(input_data)):
    output_data[i] = (output_data[i] - np.mean(np.asarray(output_data), axis=0)) / np.std(np.asarray(output_data), axis=0)
    input_data[i] = (input_data[i] - np.mean(np.asarray(input_data), axis=0)) / np.std(np.asarray(input_data), axis=0)
    output_data[i][np.isnan(output_data[i])] = 0
    input_data[i][np.isnan(input_data[i])] = 0


for i, p in enumerate(pert_ids):
    df_mcf7 = output_data[i]
    df_pc3 = input_data[i]
    baseline_corr = baseline_corr + stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0]
    decoded = autoencoder.predict(np.asarray([df_pc3]))
    our_corr = our_corr + stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0]
    print(p + ":" + str(stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0])
          + " : " + str(stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0]))
    bdata.append(stats.pearsonr(df_pc3.flatten(), df_mcf7.flatten())[0])
    ddata.append(stats.pearsonr(decoded.flatten(), df_mcf7.flatten())[0])

baseline_corr = baseline_corr / len(pert_ids)
our_corr = our_corr / len(pert_ids)
print("Baseline: " + str(baseline_corr))
print("DeepCellState: " + str(our_corr))
print("Improvement: " + str(our_corr/baseline_corr))
# exit()
tcorr = 0
tcorrb = 0
for i in range(len(pert_ids)):
    test_input = input_data[i]
    test_output = output_data[i]
    autoencoder_w = keras.models.load_model(model + "main_model/")
    autoencoder_w.get_layer("decoder").set_weights(pickle.load(open(model + "MCF7" + "_decoder_weights", "rb")))
    input_tr = np.delete(np.asarray(input_data), i, axis=0)
    output_tr = np.delete(np.asarray(output_data), i, axis=0)
    autoencoder = deepfake.build(978, 128, regul_stren=0)
    autoencoder.set_weights(autoencoder_w.get_weights())
    autoencoder.compile(loss="mse", optimizer=Adam(lr=1e-5))
    autoencoder.fit(input_tr, output_tr, epochs=50, batch_size=1)
    decoded = autoencoder.predict(np.asarray([test_input]))
    corr = stats.pearsonr(decoded.flatten(), test_output.flatten())[0]
    cdata.append(corr)
    tcorr = tcorr + corr
    print(corr)

    # Needed to prevent Keras memory leak
    del autoencoder
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()

tcorr = tcorr / len(pert_ids)
print("DeepCellState*: " + str(tcorr))
t, p = ttest_ind(bdata, ddata)
print("DeepCellState p: " + str(p))
t, p = ttest_ind(bdata, cdata)
print("DeepCellState* p: " + str(p))
for val in cdata:
    print(val)
df = pd.DataFrame(list(zip(bdata, ddata, cdata)),
                  columns=['Baseline', 'DeepCellState', "DeepCellState*"], index=pert_ids)
df.to_csv("figures_data/cancer_drugs.tsv", sep="\t")